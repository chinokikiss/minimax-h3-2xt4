"""Paired trained column GEMMs: identical full shapes/arithmetic, no generation.

Local runs are previews only. --require-t4 checks the target before allocating
test tensors. Timing is warmed separately from the instrumented DiT profile.
"""
import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import statistics
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
import comfy_kitchen as ck
from patches.dit_column_linear import column_linear
from patches.windowed_tp import CheckpointReader


def measure(fn,samples):
    torch.cuda.synchronize();start=time.perf_counter();out=fn();torch.cuda.synchronize()
    first=time.perf_counter()-start;del out
    for _ in range(2):out=fn();del out
    elapsed=[];peaks=[]
    for _ in range(samples):
        torch.cuda.synchronize();before=torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats()
        start=time.perf_counter();out=fn();torch.cuda.synchronize()
        elapsed.append((time.perf_counter()-start)*1000)
        peaks.append(torch.cuda.max_memory_allocated()-before);del out
    return dict(first_call_seconds=first,warmups=2,samples_ms=elapsed,
                median_ms=statistics.median(elapsed),peak_increment_bytes=peaks)


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--tokens',type=int,default=51220)
    p.add_argument('--samples',type=int,default=3);p.add_argument('--require-t4',action='store_true')
    args=p.parse_args();assert args.tokens>0 and args.samples>0
    torch.set_grad_enabled(False);torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    device=dict(name=torch.cuda.get_device_name(),capability=list(torch.cuda.get_device_capability()))
    if args.require_t4:assert device['capability']==[7,5] and 'T4' in device['name']
    state=dict(status='starting',device=device,torch=torch.__version__,cuda=torch.version.cuda,
        kitchen=version('comfy-kitchen'),tilelang=version('tilelang'),t4_qualified=False,
        scope='Operator-only trained base weights, seeded inputs; not real captured activations or E2E',
        changes='Lifetimes only; full GEMM, scale loop and final cat retained',rows=args.tokens,operators={},
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    args.output.parent.mkdir(parents=True,exist_ok=True)
    def save():args.output.write_text(json.dumps(state,indent=2))
    save()
    try:
        gen=torch.Generator().manual_seed(20261023)
        x=torch.randn(args.tokens,5376,generator=gen,dtype=torch.float16).cuda()
        checkpoint=CheckpointReader(args.checkpoint)
        for name,stem,components in [('FC1','mlp.fc1',2),('QKV','attn.qkv_proj',3)]:
            key='blocks.0.'+stem
            component=checkpoint.header[key+'.weight']['shape'][0]//components
            owned=[(i*component,i*component+component//2) for i in range(components)]
            weight=checkpoint.read_tensor(key+'.weight',rows=owned)
            scale=checkpoint.read_tensor(key+'.weight_scale',rows=owned)
            weight,scale=weight.cuda(),scale.cuda()
            entry=state['operators'][name]=dict(shape=[args.tokens,5376,weight.shape[0]],
                weight_sha256=hashlib.sha256(weight.cpu().numpy().tobytes()).hexdigest())
            with ck.registry.use_backend('eager'):
                baseline=lambda:ck.int8_linear(x,weight,scale,None,torch.float16,convrot=True,convrot_groupsize=256)
                candidate=lambda:column_linear(x,weight,scale,out_dtype=torch.float16)
                torch.cuda.empty_cache();entry['baseline']=measure(baseline,args.samples)
                expected=baseline().cpu()
                torch.cuda.empty_cache();entry['lifetime']=measure(candidate,args.samples)
                actual=candidate();unequal=nonfinite=0
                for start in range(0,args.tokens,2048):
                    part=actual[start:start+2048].cpu();reference=expected[start:start+2048]
                    unequal+=int(torch.count_nonzero(part.view(torch.int16)!=reference.view(torch.int16)))
                    nonfinite+=int(torch.count_nonzero(~torch.isfinite(part)))
                    del part,reference
                entry.update(unequal_storage_values=unequal,nonfinite_values=nonfinite)
                del actual,expected
            del weight,scale;save()
            print('DIT_COLUMN_LIFETIME',name,json.dumps(entry),flush=True)
        accepted=all(r['unequal_storage_values']==0 and r['nonfinite_values']==0 for r in state['operators'].values())
        state.update(status='passed' if accepted else 'numerical_gate_failed',t4_qualified=args.require_t4 and accepted)
        save();assert accepted
    except BaseException as error:
        state.update(status='failed',error=repr(error));save();raise


if __name__=='__main__':main()
