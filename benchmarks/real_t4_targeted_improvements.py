"""Re-benchmark two targeted candidates against the original physical T4 gate."""
import argparse
import json
from pathlib import Path
import sys
import time
import subprocess

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'benchmarks')]
import torch
from real_t4_attention_study import hardware, load, measure, label, save
from attention_root_causes import metrics
from patches.tilelang_sparse_attention import prepare,make_routes,execute
from patches.tilelang_dense_preparation import prepare_dense,run_prevalidated_dense,dense_attention
from patches.tilelang_grouped_qk import prepare_grouped,execute_grouped,grouped_kernel,grouped_reference


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--profile',choices=['baseline','fast_dense','group32'])
    parser.add_argument('--file',default='te_layer10_call-1.pt')
    parser.add_argument('--representatives',action='store_true')
    parser.add_argument('--dit-rank-heads',type=int)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    torch.set_grad_enabled(False);torch.backends.cuda.matmul.allow_tf32=False
    hw=hardware();assert len(hw['devices'])==2 and all(g['capability']==[7,5] for g in hw['devices']),hw
    records=json.loads((args.data/'capture_manifest.json').read_text())['records']
    scope=('full sequences and native role heads; DiT rank slice when requested; operator only' if args.representatives else
           'three diagnostic heads with full sequence; no deployment throughput claim')
    result={'status':'running','hardware':hw,'scope':scope,'cases':[]}
    for record in records:
        if args.representatives and record['file'] not in ['te_layer10_call-1.pt','dit_layer49_call09.pt','video_layer00_call00.pt']:continue
        if args.profile and record['file']!=args.file:continue
        count=args.dit_rank_heads if record['component']=='dit' else None
        (q,k,v,native),ids=load(args.data,record,heads=count,expand_gqa=False);causal=record['component']=='te'
        gqa=q.shape[1]!=k.shape[1]
        half=[x.half() for x in (q,k,v)]
        base=prepare(*half,enable_gqa=gqa);routes=make_routes(base,'dense')
        expected,_=execute(base,routes,False,False,causal=causal)
        optimized,optimized_routes=prepare_dense(*half,want_lse=True,enable_gqa=gqa)
        actual,_=run_prevalidated_dense(optimized,optimized_routes,causal=causal,want_lse=True)
        assert torch.equal(actual,expected),'Dense preparation changed attention output'
        candidate=prepare_grouped(*half,enable_gqa=gqa)
        start=time.perf_counter();grouped,_=execute_grouped(candidate,causal)
        torch.cuda.synchronize();compile_seconds=time.perf_counter()-start
        reference=grouped_reference(candidate,causal)
        implementation_error=metrics(grouped,reference)
        assert max(implementation_error['relative_l2'])<.002,implementation_error
        def baseline_api():
            with label('baseline/cast'):
                inputs=[x.half() for x in (q,k,v)]
            with label('baseline/preparation'):
                current=prepare(*inputs,enable_gqa=gqa)
            with label('baseline/routing'):
                route=make_routes(current,'dense')
            with label('baseline/attention_API'):
                return execute(current,route,False,False,causal=causal)[0]
        def fast_api():
            with label('fast_dense/cast'):
                inputs=[x.half() for x in (q,k,v)]
            with label('fast_dense/dense_only_preparation'):
                current,route=prepare_dense(*inputs,enable_gqa=gqa)
            with label('fast_dense/prevalidated_attention'):
                return run_prevalidated_dense(current,route,causal=causal)
        def grouped_api():
            with label('group32/preparation'):
                current=prepare_grouped(*[x.half() for x in (q,k,v)],enable_gqa=gqa)
            with label('group32/QK_softmax_PV'):
                return execute_grouped(current,causal)[0]
        row={'capture':record,'baseline_error':metrics(expected,native),'fast_bit_equal':True,
             'q_shape':list(q.shape),'k_shape':list(k.shape),'head_ids':ids,
             'group32_error':metrics(grouped,native),'group32_implementation_error':implementation_error,
             'group32_first_execution_and_compile_seconds':compile_seconds}
        if args.profile:
            selected={'baseline':baseline_api,'fast_dense':fast_api,'group32':grouped_api}[args.profile]
            for _ in range(5):selected()
            torch.cuda.synchronize();torch.cuda.profiler.start()
            for _ in range(3):selected()
            torch.cuda.synchronize();torch.cuda.profiler.stop()
        else:
            row['timings']={name:measure(fn,repeats=15) for name,fn in
                [('baseline_attention_API',baseline_api),('fast_dense_attention_API',fast_api),('group32_attention_API',grouped_api),
                 ('group32_raw_kernel',lambda:grouped_kernel(*q.shape,causal)(candidate.qi,candidate.ki,candidate.v,candidate.qs,candidate.ks))]}
        result['cases'].append(row);save(args.output/'targeted_improvements.json',result)
        print('IMPROVEMENT',record['file'],'group32_error',row['group32_error'],'implementation',implementation_error,flush=True)
    result['status']='complete';save(args.output/'targeted_improvements.json',result)


if __name__=='__main__':main()
