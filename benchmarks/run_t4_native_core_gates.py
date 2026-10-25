"""Physical TP2 operator and residency gates for the native Kaggle Torch profile."""
import argparse
from dataclasses import replace
from datetime import timedelta
import json
import os
from pathlib import Path
import statistics
import sys
import time
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'ComfyUI'),str(ROOT/'benchmarks')]
import torch
import torch.distributed as dist


def measure(function,repeats):
    started=time.perf_counter();output=function();torch.cuda.synchronize()
    first=1000*(time.perf_counter()-started);del output
    for _ in range(2):output=function();del output
    torch.cuda.synchronize();samples=[];torch.cuda.reset_peak_memory_stats()
    for _ in range(repeats):
        torch.cuda.synchronize();start=time.perf_counter();output=function();torch.cuda.synchronize()
        samples.append(1000*(time.perf_counter()-start));del output
    return dict(first_call_including_compilation_ms=first,samples_ms=samples,median_ms=statistics.median(samples),
                peak_allocated_bytes=torch.cuda.max_memory_allocated())


def attention_cases(repeats,full):
    import comfy_kitchen as ck
    from patches import h3_operator_policy as policy
    records=[];block=SimpleNamespace(operator_policy=policy.H3OperatorPolicy())
    torch.manual_seed(20261021)
    for dtype in (torch.float32,torch.float16):
        q=torch.randn(1,32,353,128,device='cuda',dtype=dtype)*.2
        k=torch.randn(1,4,353,128,device='cuda',dtype=dtype)*.2;v=torch.randn_like(k)
        mask=torch.ones(353,353,device='cuda',dtype=torch.bool).tril()
        run=lambda:policy.te_attention(block,q,k,v,mask)
        actual=run()
        from test_h3_operator_integration import masked_reference
        from patches import tilelang_grouped_tp_kitchen_v2 as dense
        reference=masked_reference(dense.prepare(q,k,v),mask).to(actual.dtype)
        torch.testing.assert_close(actual,reference,rtol=.006,atol=.003)
        records.append(dict(component='te',q=list(q.shape),k=list(k.shape),dtype=str(dtype),
            scheme='tilelang_group32',prepared_reference='passed',timing=measure(run,repeats)))
        del q,k,v,mask,actual,reference
    length=6814 if full else 513
    heads=28
    q=torch.randn(1,heads,length,128,device='cuda',dtype=torch.float16)*.2
    k=torch.randn_like(q)*.2;v=torch.randn_like(q)
    prefix=874 if full else 57
    options={'minimax_h3_layout':SimpleNamespace(seq_len=length,segments=[(0,prefix,'text'),(prefix,length,'video')])}
    for mode in ('kitchen','tilelang_dense','tilelang_drop','tilelang_scalar'):
        block.operator_policy=replace(block.operator_policy,dit_attention=mode)
        run=lambda:policy.dit_attention(block,q,k,v,options)
        actual=run();assert actual.shape==(1,length,heads*128) and torch.isfinite(actual).all()
        records.append(dict(component='dit',q=list(q.shape),dtype=str(q.dtype),scheme=mode,
            synthetic=True,finite=True,sparse_quality='Deferred; no semantic/distribution or L1/L2 claim',
            timing=measure(run,repeats)))
    return records


def residency_case(rank):
    from patches.weight_residency import WeightSpec,ResidencyWindow
    size=16*2**20
    def prepare(index):
        module=torch.nn.Module();module.register_buffer('w',torch.full((size//2,),index,dtype=torch.float16));return module
    specs=[WeightSpec(i,lambda i=i:prepare(i),size,prepare_peak_bytes=size) for i in range(3)]
    window=ResidencyWindow(specs,[0,1,2],torch.device('cuda',rank),max_weight_bytes=2*size,
        pinned_bytes=4*size,lookahead=2,adaptive=True,reserve_bytes=512*2**20)
    # A deployment-sized FP16 residual workspace coexists with the window.
    workspace=torch.empty(6814,5376,device='cuda',dtype=torch.float16)
    try:
        window._adapt_gpu_lookahead();assert window.lookahead==2
        free,_=torch.cuda.mem_get_info()
        memory=torch.cuda.memory_stats()
        free+=max(0,memory.get('reserved_bytes.all.current',0)-memory.get('active_bytes.all.current',0))
        window.reserve_bytes=max(0,int(free)-24*2**20)
        window._adapt_gpu_lookahead();assert window.lookahead==1
        window.reserve_bytes=512*2**20;window._adapt_gpu_lookahead();assert window.lookahead==2
        for key in (0,1,2):
            with window.use(key) as module:assert bool((module.w==key).all())
        window.finish();torch.cuda.synchronize()
        return dict(status='passed',rank=rank,workspace_shape=list(workspace.shape),workspace_dtype=str(workspace.dtype),
            admission_depths=[2,1,2],weight_bytes_per_group=size,stats=window.stats,
            scope='Physical device events/admission with bounded synthetic modules; not full TE throughput')
    finally:window.close();del workspace


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--full-shapes',action='store_true');parser.add_argument('--repeats',type=int,default=7)
    parser.add_argument('--section',choices=['all','attention','rows','residency'],default='all')
    args=parser.parse_args();sys.argv=sys.argv[:1]
    rank=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(rank)
    if torch.cuda.device_count()!=2 or torch.cuda.get_device_capability()!=(7,5) or 'T4' not in torch.cuda.get_device_name():
        raise RuntimeError('Two actual T4s required')
    from patches.legacy_runtime import validate_packages,validate_source,validate_legacy_attention_api
    validate_packages();validate_source(ROOT/'ComfyUI');api=validate_legacy_attention_api()
    import importlib.metadata as md
    assert md.version('comfy-kitchen')=='0.2.37' and md.version('tilelang')=='0.1.13'
    torch.set_grad_enabled(False);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    dist.init_process_group('nccl',timeout=timedelta(seconds=120))
    report=dict(status='running',rank=rank,torch=torch.__version__,cuda=torch.version.cuda,legacy_api=api,
        gpu=torch.cuda.get_device_name(),capability=[7,5],uuid=str(torch.cuda.get_device_properties(rank).uuid),records=[])
    output=args.output.with_name(args.output.stem+f'_rank{rank}.json');output.parent.mkdir(parents=True,exist_ok=True)
    def save():output.write_text(json.dumps(report,indent=2))
    try:
        if args.section in ('all','attention'):
            report['records'].extend(attention_cases(args.repeats,args.full_shapes));save()
            dist.barrier()
        if args.section in ('all','rows'):
            from test_h3_scaled_fp16_row import rank_cases
            from patches import optimized_tp
            optimized_tp.enable('scalemax')
            report['records'].append(dict(component='scaled_fp16_row_four_sites',cases=rank_cases(dist,rank,torch.device('cuda',rank))))
            save();dist.barrier()
            if args.full_shapes:
                rows=dict(dit_o=6814,dit_fc2=6814,te_o=353,te_down=353)
                report['records'].append(dict(component='scaled_fp16_row_deployment_shapes',
                    scope='Native FP32 rotation control isolates transport; owned_fp32 remains a separately qualified mode',
                    cases=rank_cases(dist,rank,torch.device('cuda',rank),rows_by_kind=rows,fc2_quantizer='native',repeats=5)))
                save();dist.barrier()
        if args.section in ('all','residency'):
            report['records'].append(dict(component='adaptive_lookahead',**residency_case(rank)))
        report['status']='passed'
    except BaseException as error:
        report.update(status='failed',error=repr(error));raise
    finally:save();dist.destroy_process_group()


if __name__=='__main__':main()
