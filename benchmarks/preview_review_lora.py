"""Local, seeded QKV online-LoRA geometry/peak preview; no model or pipeline."""
import argparse
import json
from pathlib import Path
import statistics
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'ComfyUI')]
import torch
from patches.h3_lora import NativeAlignedLoRA
from patches.legacy_runtime import validate_packages,validate_source


@torch.no_grad()
def run(args):
    validate_packages();validate_source(ROOT/'ComfyUI')
    torch.backends.cuda.matmul.allow_tf32=False;torch.manual_seed(20261024)
    terms=[(torch.randn(128,5376)*.02,torch.randn(21504,128)*.01,strength) for strength in (.7,-.25)]
    result=dict(status='running',device=torch.cuda.get_device_name(),capability=torch.cuda.get_device_capability(),
                torch=torch.__version__,scope='Seeded rank128 two-adapter QKV; actual TP2 geometry; local operator only',
                cases=[],physical_t4_qualification=False)
    def save():args.output.write_text(json.dumps(result,indent=2),encoding='utf-8')
    save()
    for rank in (0,1):
        rows=[(c*7168+rank*3584,c*7168+(rank+1)*3584) for c in range(3)]
        full=NativeAlignedLoRA(terms,rows=rows).cuda();full.fp32_compute=True
        selected=NativeAlignedLoRA([(a,torch.cat([b[lo:hi] for lo,hi in rows]),s) for a,b,s in terms]).cuda()
        selected.fp32_compute=True
        for count in args.rows:
            x=torch.randn(count,5376,device='cuda',dtype=torch.float16)*.2
            base=torch.randn(count,10752,device='cuda',dtype=torch.float16)*.2
            expected=full.add_to(x,base.clone())
            case=dict(rank=rank,rows=count,backends={})
            result['cases'].append(case)
            for name,adapter in [('full',full),('owned',selected)]:
                output=adapter.add_to(x,base.clone());torch.cuda.synchronize()
                exact=torch.equal(output.contiguous().view(torch.uint8),expected.contiguous().view(torch.uint8));del output
                case['backends'][name]=dict(byte_exact=exact,measurements=[])
            for repetition in range(3):
                order=[('full',full),('owned',selected)]
                if repetition%2:order.reverse()
                for name,adapter in order:
                    torch.cuda.synchronize();initial=torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats()
                    start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
                    begin=time.perf_counter();start.record();output=adapter.add_to(x,base.clone());end.record();end.synchronize()
                    case['backends'][name]['measurements'].append(dict(cuda_ms=start.elapsed_time(end),
                        host_seconds=time.perf_counter()-begin,incremental_peak_bytes=torch.cuda.max_memory_allocated()-initial))
                    del output;save()
            for value in case['backends'].values():
                value['median_cuda_ms']=statistics.median(r['cuda_ms'] for r in value['measurements'])
                value['max_incremental_peak_bytes']=max(r['incremental_peak_bytes'] for r in value['measurements'])
            print(json.dumps(dict(rank=rank,rows=count,backends={key:dict(exact=value['byte_exact'],
                cuda_ms=round(value['median_cuda_ms'],3),peak_gib=round(value['max_incremental_peak_bytes']/2**30,3))
                for key,value in case['backends'].items()})),flush=True)
            del x,base,expected;torch.cuda.empty_cache()
        del full,selected;torch.cuda.empty_cache()
    result.update(status='complete',all_outputs_byte_exact=all(c['backends']['owned']['byte_exact'] for c in result['cases']))
    save()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--rows',type=int,nargs='+',default=[261,4096,51220]);args=parser.parse_args()
    run(args)
