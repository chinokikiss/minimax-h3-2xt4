"""Trained TE weight slices, seeded inputs and shared eager preparation preview."""
import argparse
import json
from pathlib import Path
import statistics
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'ComfyUI')]
import torch
import comfy_kitchen as ck
from patches.dit_column_linear import shared_column_linears
from patches.legacy_runtime import validate_packages,validate_source
from patches.windowed_tp import CheckpointReader


@torch.no_grad()
def run(args):
    validate_packages();validate_source(ROOT/'ComfyUI')
    torch.backends.cuda.matmul.allow_tf32=False;torch.manual_seed(20261024)
    reader=CheckpointReader(args.checkpoint)
    record=dict(status='running',device=torch.cuda.get_device_name(),capability=torch.cuda.get_device_capability(),
        torch=torch.__version__,scope='Forced pinned eager path; trained TE TP2 base slices plus seeded FP16 input',
        physical_t4_qualification=False,cases=[])
    def save():args.output.write_text(json.dumps(record,indent=2),encoding='utf-8')
    x=torch.randn(1,1182,5120,device='cuda',dtype=torch.float16)*.2
    for group,columns in [('qkv',[('self_attn.q_proj',4096),('self_attn.k_proj',512),('self_attn.v_proj',512)]),
                          ('gate_up',[('mlp.gate_proj',12800),('mlp.up_proj',12800)])]:
        weights=[]
        for name,width in columns:
            prefix='model.layers.0.'+name
            weights.append(tuple(reader.read_tensor(prefix+'.'+suffix,rows=[(0,width)]).cuda()
                                 for suffix in ('weight','weight_scale')))
        with ck.registry.use_backend('eager'):
            def independent():
                return tuple(ck.int8_linear(x.view(-1,5120),w,s,None,x.dtype,convrot=True,convrot_groupsize=256)
                             .view(1,1182,w.shape[0]) for w,s in weights)
            def shared():return shared_column_linears(x,weights)
            expected=independent();actual=shared()
            exact=all(torch.equal(a.contiguous().view(torch.uint8),b.contiguous().view(torch.uint8))
                      for a,b in zip(actual,expected));del actual
            if not exact:raise AssertionError('Shared TE preparation changed output')
            case=dict(group=group,rows=1182,output_widths=[n for _,n in columns],byte_exact=exact,
                      preparation_calls=dict(independent=len(weights),shared=1),backends={})
            record['cases'].append(case)
            functions={'independent':independent,'shared':shared}
            for name,fn in functions.items():
                output=fn();torch.cuda.synchronize();del output
                case['backends'][name]=dict(measurements=[])
            for repetition in range(3):
                order=list(functions) if repetition%2==0 else list(reversed(functions))
                for name in order:
                    torch.cuda.synchronize();initial=torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats()
                    begin=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
                    begin.record();output=functions[name]();end.record();end.synchronize()
                    case['backends'][name]['measurements'].append(dict(cuda_ms=begin.elapsed_time(end),
                        incremental_peak_bytes=torch.cuda.max_memory_allocated()-initial))
                    del output
            for value in case['backends'].values():
                value['median_cuda_ms']=statistics.median(r['cuda_ms'] for r in value['measurements'])
            print(json.dumps(case),flush=True);save()
            del expected,weights;torch.cuda.empty_cache()
    record['status']='complete';save()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();run(args)
