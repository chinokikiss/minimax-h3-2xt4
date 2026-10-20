"""Native learned FFN projection: reduction accuracy and local memory only."""
import argparse
import hashlib
import json
from pathlib import Path
import simulate_weight_residency as setup
import torch
from logical_dual_gpu import LogicalDualGPU,isolate_cublas_workspaces
from patches.parallel_int8 import quantize,dequantize
from patches.windowed_tp import CheckpointReader
from patches.optimized_tp import enable,disable
from patches.quantized_collectives import int8_sum
import retest


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--rows',nargs='+',type=int,default=[257,512,8192])
    args=parser.parse_args();root=Path(__file__).resolve().parents[1];cases=[]
    import comfy_kitchen as ck
    import patches.optimized_tp as tp
    ck.backends.cuda._max_dynamic_shared_memory_per_block=lambda _:49152
    torch.set_grad_enabled(False);torch.backends.cuda.matmul.allow_tf32=False;isolate_cublas_workspaces()
    reader=CheckpointReader(setup.PATHS['dit'])
    weights={n:reader.read_tensor('blocks.0.mlp.'+n).cuda() for n in
             ('fc1.weight','fc1.weight_scale','fc2.weight','fc2.weight_scale')}
    comm=LogicalDualGPU(180)
    for label,parts in [('overflow',[torch.full((3,7),1.e4,device='cuda')]*2),
                        ('zero',[torch.zeros(3,7,device='cuda')]*2),
                        ('cancellation',[torch.full((3,7),1.e4,device='cuda'),torch.full((3,7),-1.e4,device='cuda')])]:
        result=comm.run([lambda r=r:int8_sum(parts[r],comm,2) for r in (0,1)])
        cases.append(dict(label=label,passed=torch.equal(result[0],parts[0]+parts[1]) and torch.equal(result[0],result[1])))
    enable('scalemax')
    for count in args.rows:
        torch.manual_seed(123);x=torch.randn(count,5376,device='cuda')
        packed=ck.int8_linear(x,weights['fc1.weight'],weights['fc1.weight_scale'],None,torch.float32,convrot=True,convrot_groupsize=256)
        q,s=quantize(packed,input_act='swiglu')
        reference=dequantize(tp.native_int_mm(q,weights['fc2.weight']),s,weights['fc2.weight_scale'],torch.float32)
        gate,up=packed.chunk(2,-1)
        inputs=[torch.cat((gate.chunk(2,-1)[r],up.chunk(2,-1)[r]),-1).contiguous() for r in (0,1)]
        shards=[w.contiguous() for w in weights['fc2.weight'].chunk(2,-1)]
        for mode,reduction in [('scalemax','native_int32'),('scalemax_streamed','native_int32'),
                               ('scalemax_streamed','fp32'),('scalemax_streamed','int8')]:
            enable(mode,swiglu_reduction=reduction);comm=LogicalDualGPU(180);torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats();start=torch.cuda.memory_allocated()
            outputs=comm.run([lambda r=r:tp.row_linear(inputs[r],shards[r],weights['fc2.weight_scale'],comm,2,input_act='swiglu') for r in (0,1)])
            peak=torch.cuda.max_memory_allocated()-start
            score=retest.metrics(outputs[0],reference)
            row=dict(rows=count,mode=mode,reduction=reduction,**score,peak_delta_bytes=peak,
                     replicas_equal=torch.equal(outputs[0],outputs[1]),required=reduction=='native_int32')
            row['passed']=row['is_bit_exact'] if row['required'] else retest.passes(score,dict(rel_l2_max=1e-5,cosine_min=.999999,max_abs_max=.005))
            cases.append(row);print('REDUCTION',count,mode,reduction,'exact',score['is_bit_exact'],'rel',score['rel_l2'],'MiB',round(peak/2**20,2),flush=True)
            del outputs
        del x,packed,q,s,reference,gate,up,inputs,shards;torch.cuda.empty_cache()
    disable()
    result=dict(passed=all(c['passed'] for c in cases if c.get('required',True)),physical_t4=False,
        scope='Learned Native H3 block0 FC1→SwiGLU→FC2, logical ranks on 5060 Ti. No T4 latency claim.',
        cases=cases,source_sha256={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in
            ('patches/optimized_tp.py','patches/quantized_collectives.py','benchmarks/qualify_h3_reduction.py')})
    (root/'semantic_results/h3_optional_reduction.json').write_text(json.dumps(result,indent=2))
    if not result['passed']:raise RuntimeError('Native or overflow check failed')


if __name__=='__main__':main()
