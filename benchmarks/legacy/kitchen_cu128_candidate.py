"""Explicit, experimental cu128 calls into the pinned Kitchen CUDA backend.

Registration does not enable ``cuda`` or change registry priority. Ordinary
ComfyUI/TP calls keep their validated dispatch. There is no production startup
hook: individual operations need physical SM75 qualification before promotion.
"""
import inspect

ALIAS='h3_cu128_candidate_cuda'
OPERATIONS=frozenset(('int8_linear','apply_rope_split_half','rms_rope_split_half',
                      'rms_rope_split_half_','group_norm_silu_pad3d','fp16_conv3d'))


def _register(ck,operations):
    operations=frozenset(operations)
    if not operations or not operations<=OPERATIONS:raise ValueError('Explicit H3 operation allowlist required')
    backend=ck.backends.cuda
    constraints=backend._build_constraints()
    if not operations<=constraints.keys():raise RuntimeError('Pinned CUDA backend lacks requested operations')
    ck.registry.register(ALIAS,backend,{name:constraints[name] for name in operations})


class Candidate:
    """Strict per-operation dispatch; unsupported dtype/layout raises."""
    def __init__(self,operations):
        import torch
        import comfy_kitchen as ck
        from importlib.metadata import version
        if torch.__version__!='2.11.0+cu128' or torch.version.cuda!='12.8':
            raise RuntimeError('Candidate is restricted to native Kaggle Torch 2.11.0+cu128')
        if version('comfy-kitchen')!='0.2.37' or version('tilelang')!='0.1.13':
            raise RuntimeError('Preserved core pins required')
        if not torch.cuda.is_available() or torch.cuda.get_device_capability()!=(7,5):
            raise RuntimeError('Physical SM75 required; local SM120 is not qualification')
        if not ck.backends.cuda._EXT_AVAILABLE:raise RuntimeError('Pinned CUDA extension unavailable')
        self.ck=ck;self.operations=frozenset(operations)
        self.cuda_state=ck.registry.list_backends().get('cuda')
        _register(ck,self.operations)

    def call(self,name,*args,**kwargs):
        if name not in self.operations:raise ValueError('Operation not allowed: '+name)
        bound=inspect.signature(getattr(self.ck,name)).bind(*args,**kwargs)
        bound.apply_defaults()
        values=dict(bound.arguments)
        # Preserve the public wrapper's BF16 default; the backend function's
        # own None default would instead choose x.dtype.
        if name=='int8_linear' and values['out_dtype'] is None:
            import torch
            values['out_dtype']=torch.bfloat16
        if name=='fp16_conv3d':
            # The public wrapper consumes out and canonicalizes stride before
            # dispatching to the five-argument backend function.
            if values.pop('out',None) is not None:
                raise ValueError('Candidate fp16_conv3d does not qualify the separate out variant')
            stride=values['stride']
            values['stride']=[stride]*3 if isinstance(stride,int) else list(stride)
        impl=self.ck.registry.get_implementation(name,backend=ALIAS,kwargs=values)
        return impl(**values)

    def unchanged_global_cuda(self):
        return self.cuda_state==self.ck.registry.list_backends().get('cuda')
