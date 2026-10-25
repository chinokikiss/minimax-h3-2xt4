"""Local operator preview only; no model construction, NFE or Kaggle calls.

Use the exact pre-edit column source as a baseline and immutable trained base
weight slices. Seeded activations are shape fixtures, not captured H3 states.
Experimental kernels are invoked directly to preview them on the local GPU;
production dispatch remains gated to SM75 and is not qualified by this run.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'ComfyUI')]
import torch
from patches.legacy_runtime import validate_packages,validate_source
from patches.dit_column_linear import _eager_column
from patches.windowed_tp import CheckpointReader


def sha_tensor(tensor):
    return hashlib.sha256(memoryview(tensor.contiguous().view(torch.uint8).numpy()).cast('B')).hexdigest()


@torch.no_grad()
def run(args):
    packages=validate_packages();validate_source(ROOT/'ComfyUI')
    source=args.prior.read_bytes()
    if hashlib.sha256(source).hexdigest()!='437741a15f536925563453eb7463c472a297f3f2c8c7983deed1bd78f1125ebf':
        raise ValueError('The baseline must be the exact pre-edit column source')
    spec=importlib.util.spec_from_file_location('prior_column',args.prior)
    prior=importlib.util.module_from_spec(spec);spec.loader.exec_module(prior)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.manual_seed(20261024)
    reader=CheckpointReader(args.checkpoint)
    receipt=dict(status='running',device=torch.cuda.get_device_name(),
        capability=torch.cuda.get_device_capability(),torch=torch.__version__,cuda=torch.version.cuda,
        kitchen=packages['comfy-kitchen'],tilelang=packages['tilelang'],
        snapshot_packages_checked=len(packages),source_frozen=True,
        workload='Trained base TP2 weight slices plus seeded FP16 shape fixtures; no full pipeline',
        baseline_sha256=hashlib.sha256(source).hexdigest(),cases=[],epilogue=[])
    def save():args.output.write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    save()
    from patches.tilelang_column_epilogue import dequantize
    # Deliberately exercise INT32->FP32 rounding, non-power-of-two scales,
    # negative accumulators and FP16 range behavior separately from INT8 GEMM.
    values=torch.tensor([-2147483647,-16777217,-3,0,3,16777217,2147483647],device='cuda',dtype=torch.int32)
    acc=values.repeat(17,1).contiguous()
    row_scale=torch.logspace(-8,2,17,device='cuda').reshape(-1,1)
    for scalar in (True,False):
        scale=torch.tensor([.314159],device='cuda') if scalar else torch.linspace(.01,.9,7,device='cuda')
        for dtype in (torch.float16,torch.bfloat16,torch.float32):
            expected=(acc.float()*(row_scale*scale)).to(dtype)
            actual=dequantize(acc,row_scale,scale,dtype)
            exact=torch.equal(expected.contiguous().view(torch.uint8),actual.contiguous().view(torch.uint8))
            receipt['epilogue'].append(dict(scalar_scale=scalar,dtype=str(dtype),byte_exact=exact))
            save()
            if not exact:raise AssertionError('Column epilogue rounding contract failed')
    del acc,actual,expected,row_scale,scale
    for projection,rows in [('attn.qkv_proj',[(0,3584),(7168,10752),(14336,17920)]),
                            ('mlp.fc1',[(0,7168),(14336,21504)])]:
        prefix='blocks.0.'+projection
        weight=reader.read_tensor(prefix+'.weight',rows=rows)
        scale=reader.read_tensor(prefix+'.weight_scale',rows=rows)
        identity=dict(weight_sha256=sha_tensor(weight),scale_sha256=sha_tensor(scale))
        weight,scale=weight.cuda(),scale.cuda()
        for count in args.rows:
            x=(torch.randn(count,5376,device='cuda',dtype=torch.float16)*.2)
            options=dict(convrot=True,convrot_groupsize=256,out_dtype=torch.float16)
            reference=prior._eager_column(x,weight,scale,**options)
            backends={'prior':lambda:prior._eager_column(x,weight,scale,**options),
                      **{name:(lambda name=name:_eager_column(x,weight,scale,backend=name,**options))
                         for name in ('preallocated','tilelang_exact','cuda_int8_tilelang')}}
            case=dict(projection=projection,shape=[count,weight.shape[0],5376],**identity,backends={})
            receipt['cases'].append(case);save()
            for name,fn in backends.items():
                begin=time.perf_counter();output=fn();torch.cuda.synchronize()
                compile_warm_seconds=time.perf_counter()-begin
                exact=torch.equal(output.contiguous().view(torch.uint8),reference.contiguous().view(torch.uint8));del output
                result=dict(byte_exact=exact,compile_warm_seconds=compile_warm_seconds,measurements=[])
                case['backends'][name]=result;save()
                if not exact:raise AssertionError(f'Column mismatch: {projection} {count} {name}')
            for repetition in range(3):
                order=list(backends) if repetition%2==0 else list(reversed(backends))
                for name in order:
                    torch.cuda.synchronize();base=torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats()
                    start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
                    begin=time.perf_counter();start.record();output=backends[name]();end.record();end.synchronize()
                    wall=time.perf_counter()-begin;peak=torch.cuda.max_memory_allocated()
                    case['backends'][name]['measurements'].append(dict(repetition=repetition,
                        cuda_ms=start.elapsed_time(end),host_seconds=wall,peak_allocated=peak,
                        incremental_peak_bytes=peak-base))
                    del output;save()
            for result in case['backends'].values():
                result['median_cuda_ms']=statistics.median(r['cuda_ms'] for r in result['measurements'])
                result['max_incremental_peak_bytes']=max(r['incremental_peak_bytes'] for r in result['measurements'])
            print(json.dumps(dict(shape=case['shape'],projection=projection,
                backends={key:dict(cuda_ms=round(value['median_cuda_ms'],3),
                                  peak_gib=round(value['max_incremental_peak_bytes']/2**30,3))
                          for key,value in case['backends'].items()})),flush=True)
            del x,reference;torch.cuda.empty_cache()
        del weight,scale;torch.cuda.empty_cache()
    receipt.update(status='complete',physical_t4_qualification=False,
                   precision_scope='Byte equality for these local fixtures only; no full-generation claim')
    save()


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--prior',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--rows',type=int,nargs='+',default=[261,4096,51220])
    args=parser.parse_args()
    run(args)
