"""Checkpoint-complete text-only TE and audiovisual DiT native entry points.

Placement wrappers change device ownership only. PP schedules independent
requests; no sequence chunks or altered attention domains are introduced.
"""
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor
import dataclasses
import json
from pathlib import Path
import struct
import threading

import retest
import torch
from torch import nn

PLACEMENT=threading.local()


def auxiliary_state(kind,path):
    prefix='model.layers.' if kind=='te' else 'blocks.'
    if kind=='te': return {'embed_tokens.weight':retest.state(path,'model.embed_tokens.')['weight']}
    # Windows cannot mmap the entire checkpoint without exhausting commit.
    if __import__('os').name!='nt':
        from safetensors import safe_open
        with safe_open(path,framework='pt',device='cpu') as f:
            return {k:f.get_tensor(k) for k in f.keys() if not k.startswith(prefix)}
    dtypes={'F32':torch.float32,'F16':torch.float16,'BF16':torch.bfloat16,'I8':torch.int8,'U8':torch.uint8}
    result={}
    with Path(path).open('rb') as stream:
        n=struct.unpack('<Q',stream.read(8))[0];h=json.loads(stream.read(n))
        for k,v in h.items():
            if k=='__metadata__' or k.startswith(prefix):continue
            a,b=v['data_offsets'];stream.seek(8+n+a)
            result[k]=torch.frombuffer(bytearray(stream.read(b-a)),dtype=dtypes[v['dtype']]).reshape(v['shape'])
    return result


def skeleton(kind,path,device):
    import comfy.ops
    dtype=torch.float16 if kind=='te' else torch.float32
    ops=comfy.ops.mixed_precision_ops({'mixed_ops':True},dtype)
    if kind=='te':
        from comfy.text_encoders.llama import Llama2_,Qwen3VL_32BConfig
        c=Qwen3VL_32BConfig();model=Llama2_(dataclasses.replace(c,num_hidden_layers=0),device='cpu',dtype=dtype,ops=ops);model.config=c
    else:
        from comfy.ldm.minimax.model import MiniMaxH3Model
        model=MiniMaxH3Model(num_layers=0,adaln_curve_grid=1025,time_embed_dim=8,dtype=dtype,device='cpu',operations=ops)
    sd=auxiliary_state(kind,path);model.load_state_dict(sd,strict=True);del sd
    return model.to(device).eval().requires_grad_(False)


def move(value,device,cache=True):
    if isinstance(value,torch.Tensor):
        target_stream=torch.cuda.current_stream(device) if torch.device(device).type=='cuda' else None
        key=(id(value),str(device),target_stream.cuda_stream if target_stream else None)
        storage=getattr(PLACEMENT,'cache',None) if cache else None
        if storage is not None and key in storage:return storage[key][1]
        result=value.to(device,non_blocking=True)
        if result.is_cuda:result.record_stream(torch.cuda.current_stream(result.device))
        if storage is not None:storage[key]=(value,result)
        return result
    if isinstance(value,list):return [move(v,device) for v in value]
    if isinstance(value,tuple):return tuple(move(v,device) for v in value)
    # transformer_options and other immutable metadata remain shared per call.
    return value


class PlacedBlock(nn.Module):
    def __init__(self,block,kind,device,index,runner=None):
        super().__init__();self.block=block;self.kind=kind;self.device=torch.device(device);self.index=index;self.runner=runner

    def forward(self,x,*args,**kwargs):
        if self.runner is not None and self.index==self.runner.split:self.runner.boundary()
        with torch.cuda.device(self.device),torch.profiler.record_function(f'layer::{self.index}'):
            x=move(x,self.device,cache=False);args=move(args,self.device)
            kwargs={k:move(v,self.device) for k,v in kwargs.items()}
            if self.kind=='te':
                from comfy.ldm.modules.attention import optimized_attention_for_device
                kwargs['optimized_attention']=optimized_attention_for_device(self.device,mask=kwargs.get('attention_mask') is not None,small_input=True)
            return self.block(x,*args,**kwargs)


class PlacedFinal(nn.Module):
    def __init__(self,inner,device):super().__init__();self.inner=inner;self.device=torch.device(device)
    def forward(self,x,*args,**kwargs):
        with torch.cuda.device(self.device):return self.inner(move(x,self.device,cache=False),*move(args,self.device),**{k:move(v,self.device) for k,v in kwargs.items()})


class TPAdapter(nn.Module):
    def __init__(self,inner,kind):super().__init__();self.inner=inner;self.kind=kind
    def forward(self,x,*args,**kwargs):
        if self.kind=='te':
            if kwargs.get('past_key_value') is not None:raise ValueError('This study measures prefill, without KV decoding')
            return self.inner(x,kwargs.get('attention_mask'),kwargs.get('freqs_cis')),None
        attention=kwargs.pop('attention',None)
        if attention is not None:raise ValueError('Custom attention is outside the study')
        return self.inner(x,*args,**kwargs)


class StreamedBlock(nn.Module):
    """Single-device correctness reference, never reported as resident latency."""
    def __init__(self,kind,path,index,device):super().__init__();self.kind=kind;self.path=path;self.index=index;self.device=device
    def forward(self,*args,**kwargs):
        with torch.cuda.device(self.device):
            block=retest.load_native_block(self.kind,self.path,self.index,self.device)
            out=block(*args,**kwargs);torch.cuda.synchronize(self.device);del block
            return out


def load_full(kind,path,mode,rank=0,split=25,runner=None,devices=('cuda:0','cuda:1'),layers=50,offload_dit_aux=False):
    device=devices[rank] if mode=='tp' else devices[0]
    model=skeleton(kind,path,device)
    if kind=='dit' and offload_dit_aux:stage_dit_aux(model,device,shared=mode=='tp',rank=rank)
    values=[]
    for index in range(layers):
        if mode=='streamed':values.append(StreamedBlock(kind,path,index,device));continue
        if mode=='tp':
            from patches.tp_qwen3vl import TPTransformerBlockINT8
            from patches.tp_dit import TPDiTBlockINT8
            native=retest.load_native_block(kind,path,index,'cpu')
            cls=TPTransformerBlockINT8 if kind=='te' else TPDiTBlockINT8
            block=cls(rank=rank,world_size=2,device=device).to(device).eval().requires_grad_(False)
            block.load_from_full_block(native);del native
            values.append(TPAdapter(block,kind))
        else:
            target=devices[0 if index<split else 1]
            block=retest.load_native_block(kind,path,index,target)
            values.append(PlacedBlock(block,kind,target,index,runner))
        if index%10==9:print('LOADED',kind,mode,index+1,flush=True)
    setattr(model,'layers' if kind=='te' else 'blocks',nn.ModuleList(values))
    if kind=='dit' and mode not in ('tp','streamed'):model.final_layer=PlacedFinal(model.final_layer.to(devices[1]),devices[1])
    return model


def stage_dit_aux(model,device,shared=False,rank=0,comm=None):
    """Stage native preprocessing weights; keep their lifetime outside blocks.

    In TP only rank zero preprocesses, then broadcasts the native packed
    activation. The native time curves, RoPE and output heads remain intact.
    Weight staging and broadcast remain inside the timed native forward.
    """
    import torch.distributed as distributed
    import types
    comm=comm or distributed
    names=('condition_proj','token_refiner','video_patch_proj','audio_patch_proj')
    modules=[getattr(model,n) for n in names]
    for m in modules:m.cpu()
    # Keep immutable CPU parameters, avoiding a fresh multi-GB device-to-host
    # allocation on every step. Temporary parameter wrappers share CPU storage
    # until Module.to creates the CUDA copy; native operator dispatch is intact.
    owners=[m for parent in modules for m in parent.modules()]
    parameters=[(m,dict(m._parameters)) for m in owners]
    buffers=[(m,dict(m._buffers)) for m in owners]
    original=model._embed_and_pack
    def packed(self,video_x,audio_x,context,layout,payload,transformer_options):
        if not shared or rank==0:
            for m,values in parameters:
                m._parameters={n:nn.Parameter(p.detach(),requires_grad=p.requires_grad) if p is not None else None for n,p in values.items()}
            for m in modules:m.to(device)
            try:h=original(video_x,audio_x,context,layout,payload,transformer_options)
            finally:
                # Fence before releasing temporary CUDA parameters. The same
                # native CPU tensors are restored, with no copy back to host.
                torch.cuda.current_stream(device).synchronize()
                for m,values in parameters:m._parameters=dict(values)
                for m,values in buffers:m._buffers=dict(values)
        else:h=torch.empty((layout.seq_len,self.hidden_size),device=device,dtype=context.dtype)
        if shared:comm.broadcast(h,src=0)
        return h
    model._embed_and_pack=types.MethodType(packed,model)


def request(kind,length,seed,device):
    if kind=='te':
        # Deterministic synthetic token IDs, genuine checkpoint embedding lookup.
        ids=((torch.arange(length,dtype=torch.int64)*7919+seed)%151643)[None].to(device)
        return (ids,),{'dtype':torch.float16}
    from fixture_inputs import block_input
    # Packed native sequence = 48 text + 16 audio + video patch tokens.
    dimensions={68:(4,4),512:(32,56),1024:(48,80),4096:(96,168)}
    h,w=dimensions[length]
    text=block_input('te',48,seed)['input'].float().to(device)
    # Use a portable counter as a source, reshape only the required values.
    values=block_input('te',max(1,(24*h*w+32*16+5119)//5120),seed)['input'].float().flatten().to(device)
    video=values[:24*h*w].reshape(1,24,1,h,w).clone()
    audio=values[24*h*w:24*h*w+32*16].reshape(1,32,2,8).clone()
    return ([video,audio],torch.tensor([625.0],device=device),text),{'transformer_options':{}}


def forward(kind,model,data):
    args,kwargs=data
    options=dict(kwargs)
    if 'transformer_options' in options:options['transformer_options']=dict(options['transformer_options'])
    PLACEMENT.cache={}
    try:out=model(*args,**options)
    finally:PLACEMENT.cache=None
    return out[0] if kind=='te' else torch.cat([v.flatten() for v in out])


class PipelineRunner:
    """Two stage inference pipeline with batch-1 independent microrequests.

    Native forwards retain all preprocessing and output logic. A boundary
    hands GPU ownership to stage 1. Per-device streams serialize a stage,
    while different devices execute independent microrequests concurrently.
    """
    def __init__(self,kind,split=25,devices=('cuda:0','cuda:1')):
        self.kind=kind;self.split=split;self.devices=devices
        self.local=threading.local();self.locks=[threading.Lock(),threading.Lock()]
        self.streams=[torch.cuda.Stream(device=d) for d in devices]
        self.events=[];self.event_lock=threading.Lock();self.model=None

    def boundary(self):
        end=torch.cuda.Event(enable_timing=True);end.record(self.streams[0]);self.local.stage0_end=end
        self.local.context.__exit__(None,None,None)
        self.locks[0].release();self.local.locked=None
        self.locks[1].acquire();self.local.locked=1
        self.local.context=torch.cuda.stream(self.streams[1]);self.local.context.__enter__()
        # Cross-device event dependency also works in local semantic simulation.
        self.streams[1].wait_event(end)
        start=torch.cuda.Event(enable_timing=True);start.record(self.streams[1]);self.local.stage1_start=start

    def one(self,index,data):
        self.locks[0].acquire();self.local.locked=0
        self.local.context=torch.cuda.stream(self.streams[0]);self.local.context.__enter__()
        start=torch.cuda.Event(enable_timing=True);start.record(self.streams[0]);self.local.start=start
        try:
            with torch.cuda.device(self.devices[0]),torch.inference_mode(),torch.profiler.record_function(f'PP::microbatch{index}'):
                result=forward(self.kind,self.model,data)
                end=torch.cuda.Event(enable_timing=True);end.record(self.streams[1])
                with self.event_lock:self.events.append((index,start,self.local.stage0_end,self.local.stage1_start,end))
                return result
        finally:
            self.local.context.__exit__(None,None,None)
            if self.local.locked is not None:self.locks[self.local.locked].release()

    def __call__(self,requests):
        self.events=[]
        # Inputs are GPU resident on the default stream. Fence once before use.
        for stream in self.streams:stream.wait_stream(torch.cuda.default_stream(stream.device))
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures=[pool.submit(self.one,i,data) for i,data in enumerate(requests)]
            results=[f.result() for f in futures]
        for d in set(self.devices):torch.cuda.synchronize(d)
        # The return is consumed on the caller's stream. The fence above must
        # precede stacking outputs produced on the stage-1 stream.
        return torch.stack(results)

    def breakdown(self):
        values=sorted(self.events)
        s0=[a.elapsed_time(b) for _,a,b,c,d in values];s1=[c.elapsed_time(d) for _,a,b,c,d in values]
        completions=[values[0][4].elapsed_time(v[4]) for v in values]
        ordered=sorted(completions)
        return {'stage0_ms':s0,'stage1_including_transfer_and_heads_ms':s1,
                'completion_offsets_ms':completions,
                'completion_microrequest_order':sorted(range(len(values)),key=lambda i:completions[i]),
                'steady_completion_interval_ms':[(b-a) for a,b in zip(ordered,ordered[1:])],
                'microbatch_size':1,'microbatches':len(values),
                'ideal_balanced_bubble_fraction':1/(len(values)+1),
                'note':'Fill/drain included in measured batch wall time. Ideal bubble is a scheduling reference, not measured overhead; stage1 includes boundary transfers.'}
