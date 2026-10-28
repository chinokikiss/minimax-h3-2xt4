"""Scoped qualified Video VAE CUDA operators; never enable a global backend."""
from contextlib import contextmanager
import contextvars
import inspect
from functools import lru_cache
import threading

import torch

_active=contextvars.ContextVar('h3_video_cuda_linears',default=False)
_contexts=contextvars.ContextVar('h3_video_cuda_contexts',default=())
_lock=threading.RLock()
_shapes={(6144,2048,'rms_norm'),(2048,2048,None),
         (16384,2048,'rms_norm'),(2048,8192,'swiglu')}


@lru_cache(maxsize=1)
def _backend():
    import comfy_kitchen as ck
    from importlib.metadata import version
    if version('comfy-kitchen')!='0.2.37':raise RuntimeError('Pinned Kitchen 0.2.37 required')
    return ck.backends.cuda


def int8_linear(original,*args,**kwargs):
    if not _active.get():return original(*args,**kwargs)
    import comfy_kitchen as ck
    bound=inspect.signature(ck.int8_linear).bind(*args,**kwargs);bound.apply_defaults()
    values=dict(bound.arguments);x=values['x'];w=values['weight']
    eligible=(_active.get() and not torch.is_grad_enabled() and x.is_cuda
        and x.dtype==torch.float16 and values['out_dtype']==torch.float16
        and w.dtype==torch.int8 and w.ndim==2 and w.device==x.device
        and values['convrot'] and values['convrot_groupsize']==256
        and (w.shape[0],w.shape[1],values['input_act']) in _shapes
        and x.numel()//x.shape[-1]>=128
        and torch.cuda.get_device_capability(x.device)==(7,5))
    if eligible:
        backend=_backend()
        if backend._EXT_AVAILABLE:return backend.int8_linear(**values)
    return original(*args,**kwargs)


class _Proxy:
    def __init__(self,original):self.original=original;self.users=0
    def __getattr__(self,name):return getattr(self.original,name)
    def int8_linear(self,*args,**kwargs):return int8_linear(self.original.int8_linear,*args,**kwargs)
    def rms_rope_split_half(self,*args,**kwargs):return self._native('rms_rope_split_half',*args,**kwargs)
    def rms_rope_split_half_(self,*args,**kwargs):return self._native('rms_rope_split_half_',*args,**kwargs)
    def group_norm_silu_pad3d(self,*args,**kwargs):return self._native('group_norm_silu_pad3d',*args,**kwargs)

    def _native(self,name,*args,**kwargs):
        original=getattr(self.original,name)
        if not _active.get() or torch.is_grad_enabled():return original(*args,**kwargs)
        import comfy_kitchen as ck
        bound=inspect.signature(getattr(ck,name)).bind(*args,**kwargs);bound.apply_defaults();values=dict(bound.arguments)
        x=values['q'] if name.startswith('rms_rope') else values['x']
        if not x.is_cuda or x.dtype!=torch.float16 or torch.cuda.get_device_capability(x.device)!=(7,5):
            return original(*args,**kwargs)
        if name.startswith('rms_rope'):
            eligible=(x.shape[-1]==64 and values['k'].dtype==torch.float16 and values['rot_dim']==48)
        else:eligible=x.ndim==5 and x.shape[1] in (128,1024) and values['num_groups']==32
        backend=_backend()
        if eligible and backend._EXT_AVAILABLE:return getattr(backend,name)(**values)
        return original(*args,**kwargs)


@contextmanager
def scope():
    import comfy.quant_ops
    with _lock:
        if not isinstance(comfy.quant_ops.ck,_Proxy):comfy.quant_ops.ck=_Proxy(comfy.quant_ops.ck)
        proxy=comfy.quant_ops.ck;proxy.users+=1
    token=_active.set(True)
    try:yield
    finally:
        _active.reset(token)
        with _lock:
            proxy.users-=1
            if proxy.users==0 and comfy.quant_ops.ck is proxy:comfy.quant_ops.ck=proxy.original


def _enter(module,args):
    manager=scope();manager.__enter__();_contexts.set((*_contexts.get(),manager))


def _exit(module,args,output):
    stack=_contexts.get()
    if stack:
        _contexts.set(stack[:-1]);stack[-1].__exit__(None,None,None)


def install(model):
    for name in ('encoder','decoder'):
        component=getattr(model,name,None)
        if component is not None and not getattr(component,'_h3_cuda_operators',False):
            component.register_forward_pre_hook(_enter)
            component.register_forward_hook(_exit,always_call=True)
            component._h3_cuda_operators=True
    return model
