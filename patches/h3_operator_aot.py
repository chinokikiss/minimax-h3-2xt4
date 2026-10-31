"""Prepare all selected H3 source-compiled operators by stage and exact shape.

This moves cold compilation into preparation; it does not erase startup cost.
TileLang retains its own source/target keyed binary cache. No dummy GEMMs,
collectives, model loading or full-sized activations are needed here.
"""
from functools import lru_cache
import hashlib
from importlib import import_module
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import time
import torch


@lru_cache(maxsize=2)
def environment(include_tilelang=True):
    compiler=os.environ.get('CUDACXX') or shutil.which('nvcc')
    if not compiler:raise RuntimeError('Pinned nvcc required for operator preparation')
    compiler_version=subprocess.check_output([compiler,'--version'],text=True,timeout=15)
    if 'release 12.8' not in compiler_version:raise RuntimeError('Preserved CUDA 12.8 compiler required')
    return dict(compiler=compiler,compiler_version=compiler_version,python=platform.python_version(),
        torch=str(torch.__version__),cuda=torch.version.cuda,
        tilelang=version('tilelang') if include_tilelang else None,arch='sm_75')


def plan(rows,rank,policy,stage='dit',*,norm_epsilon=1e-5/4096):
    if type(rows) is not int or rows<1 or rank not in (0,1):raise ValueError('Positive packed length and TP2 rank required')
    if stage not in ('te','dit','video'):raise ValueError('Unknown H3 preparation stage')
    calls=[]
    def add(module,name,*args):calls.append((module,name,args))
    if stage=='video':return calls  # Selected VAE kernels ship in the pinned Kitchen binary.
    if stage=='te' or policy.column_linear_backend=='cuda_int8_tilelang':
        for width in ((512,4096,12800) if stage=='te' else (10752,14336)):
            add('tilelang_column_epilogue','kernel',rows,width,False,'float16','sm_75')
    out_width=5120 if stage=='te' else 5376
    attn_width=4096 if stage=='te' else 3584
    mlp_width=12800 if stage=='te' else 7168
    if policy.row_linear_backend=='tilelang_scaled_fp16':
        from patches.tilelang_transport_tradeoffs import shift_for
        add('tilelang_row_quant','packing_kernel',rows,attn_width,2,rank)
        add('tilelang_row_quant','maximum_kernel',rows,attn_width,attn_width*2,rank*attn_width)
        add('tilelang_row_quant','quantizer',rows,attn_width,attn_width*2,rank*attn_width,'fast')
        if policy.fc2_quantizer=='fused_stored':
            add('tilelang_swiglu_quant','fused_kernel',rows,mlp_width,'float16','sm_75','stored')
            add('tilelang_row_quant','quantizer',rows,mlp_width,mlp_width,0,'fast')
        for local_k in (attn_width,mlp_width):
            shift=shift_for(local_k)
            add('tilelang_transport_tradeoffs','fp16_codec',rows,out_width,shift)
            add('tilelang_transport_p2','epilogue',rows,out_width,'fp16',shift)
    if stage=='dit' and getattr(policy,'dit_rms_rope','native')=='tilelang_fp32':
        # Native H3 rotates three axes, each with 16 frequency pairs: 96/128.
        add('tilelang_fp32_rms_rope','kernel',rows,28,128,96,'float16','float16',1e-6,'sm_75','float16')
    if stage=='dit':
        if getattr(policy,'dit_fused_norm',False):add('tilelang_dit_fusion','norm_kernel',rows,5376,9,float(norm_epsilon))
        if getattr(policy,'dit_fused_gate',False):add('tilelang_dit_fusion','gate_kernel',rows,5376,9)
        if getattr(policy,'dit_fused_row_gate',False) and policy.row_linear_backend=='tilelang_scaled_fp16':
            from patches.tilelang_transport_tradeoffs import shift_for
            for width in (3584,7168):add('tilelang_dit_fusion','row_gate_kernel',rows,5376,9,shift_for(width))
    return calls


@lru_cache(maxsize=32)
def prepare(rows,rank,policy,stage='dit',*,norm_epsilon=1e-5/4096,sol=False):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()!=(7,5):
        return dict(status='unsupported_device',rows=rows,rank=rank)
    calls=plan(rows,rank,policy,stage,norm_epsilon=norm_epsilon)
    if calls and version('tilelang')!='0.1.13':raise RuntimeError('Pinned TileLang required')
    build_environment=environment(bool(calls));start=time.monotonic();steps=[];sources={}
    for module,name,args in calls:
        mod=import_module('patches.'+module);fn=getattr(mod,name)
        before=fn.cache_info();begin=time.monotonic();fn(*args);after=fn.cache_info()
        steps.append(dict(module=module,function=name,args=args,seconds=time.monotonic()-begin,
            compiled=after.misses>before.misses))
        sources[module]=hashlib.sha256(Path(mod.__file__).read_bytes()).hexdigest()
    sol_library=None
    if sol:
        from patches import sol_attention
        begin=time.monotonic();sol_library=sol_attention.prepare_native()
        steps.append(dict(module='sol_attention',function='prepare_native',args=(),seconds=time.monotonic()-begin,
            compiled_or_loaded=True,library=sol_library))
        sources['sol_attention']=hashlib.sha256(Path(sol_attention.__file__).read_bytes()).hexdigest()
        for path in sorted(sol_attention.NATIVE.rglob('*')):
            if path.is_file():
                sources['sol_native/'+path.relative_to(sol_attention.NATIVE).as_posix()]=hashlib.sha256(path.read_bytes()).hexdigest()
    key=dict(stage=stage,rows=rows,rank=rank,policy=vars(policy),sol=bool(sol),sources=sources,environment=build_environment)
    return dict(status='prepared',stage=stage,rows=rows,rank=rank,seconds=time.monotonic()-start,steps=steps,
        identity_sha256=hashlib.sha256(json.dumps(key,sort_keys=True).encode()).hexdigest(),
        environment=key['environment'],
        sol_precompiled=bool(sol),sol_library=sol_library,
        complete_default_custom_coverage=((stage=='video' or (policy.row_linear_backend=='tilelang_scaled_fp16'
            and policy.fc2_quantizer=='fused_stored' and policy.te_attention in ('kitchen_causal','native')
            and policy.dit_attention in ('kitchen','sol','native'))) and (not sol or sol_library is not None)),
        prebuilt='Torch/cuBLAS/NCCL and Kitchen binaries are shipped compiled; selected TileLang and Sol source operators are AOT-prepared here; first-use execution warmup is separate',
        semantics='Preparation compilation only; startup cost is reported separately')
