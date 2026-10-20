"""Faithful sparse preparation ablation on actual native H56 T4 captures."""
import argparse
import json
from pathlib import Path
import random
import sys
import time
from dataclasses import replace
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'benchmarks')]
import torch
from real_t4_attention_study import hardware,load,measure,label,save
from attention_root_causes import metrics
from patches.tilelang_sparse_attention import prepare,execute,reference
from patches.tilelang_sol_drop_fused import prepare_drop_fused,bound_sol_routes,_execute_bound_drop
from patches.tilelang_sage_preparation import prepare_sage_fused
from patches.tilelang_dense_preparation import run_prevalidated_dense


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--process',type=int,default=0);ap.add_argument('--profile',action='store_true');args=ap.parse_args()
    args.output.mkdir(parents=True,exist_ok=True);torch.set_grad_enabled(False);torch.backends.cuda.matmul.allow_tf32=False
    hw=hardware();assert len(hw['devices'])==2 and all(d['capability']==[7,5] for d in hw['devices'])
    result={'status':'running','hardware':hw,'process':args.process,'cases':[],
        'scope':'GPU0 actual H56 native capture -> FP16 BHND; not native output layout, generation quality or TP2 throughput',
        'ablation':'Same Sol tau0 centering/QI/KI/scales/routes/kernel/FP16-PV; fused quantizers; no unused V8/VC/LSE or host validation bool'}
    records=json.loads((args.data/'capture_manifest.json').read_text())['records']
    for record in records:
        (q,k,v,native),heads=load(args.data,record);assert q.shape==torch.Size([1,56,6814,128])
        original=[q,k,v];half=[x.half() for x in original]
        old=prepare(*half);fast=prepare_drop_fused(*half)
        ro=bound_sol_routes(old,874);rf=bound_sol_routes(fast,874)
        n=q.shape[2]
        match={'QI':torch.equal(old.qi[:,:,:n],fast.qi[:,:,:n]),'KI':torch.equal(old.ki[:,:,:n],fast.ki[:,:,:n]),
               'QS':torch.equal(old.qs[:,:,:n],fast.qs[:,:,:n]),'KS':torch.equal(old.ks[:,:,:n],fast.ks[:,:,:n]),
               'KC':torch.equal(old.kc,fast.kc),'counts':torch.equal(old.counts,fast.counts),'routes':torch.equal(ro,rf)}
        assert all(match.values()),match
        modes=['legacy_sol_drop','legacy_sol_unchecked','fused_sol_drop','fused_sage32'];random.Random(712+args.process).shuffle(modes)
        for mode in modes:
            def prep():
                hh=[x.half() for x in original]
                if mode=='fused_sage32':return prepare_sage_fused(*hh)
                return (prepare_drop_fused if mode=='fused_sol_drop' else prepare)(*hh)
            def route(p):return bound_sol_routes(p,874)
            def raw(p,r):
                if mode=='fused_sage32':return run_prevalidated_dense(*p)
                return execute(p,r,pooled=False)[0] if mode=='legacy_sol_drop' else _execute_bound_drop(p,r)
            def api():
                with label(mode+'/input_cast_and_preparation'):pp=prep()
                with label(mode+'/routing'):rr=None if mode=='fused_sage32' else route(pp)
                with label(mode+'/validation_QK_softmax_PV_output'):return raw(pp,rr)
            start=time.perf_counter();p=prep();r=None if mode=='fused_sage32' else route(p);actual=raw(p,r)
            torch.cuda.synchronize()
            row={'capture':record,'shape':list(q.shape),'candidate':mode,'legacy_operand_match':match,
                 'compile_first_call_seconds':time.perf_counter()-start,'heads':heads}
            if mode!='fused_sage32':
                row['density']=r.float().mean().item()
                if args.process==0 and not args.profile:
                    ids=torch.tensor([0,28,55],device=q.device)
                    selected=replace(old,**{key:value[:,ids].contiguous() for key,value in old.__dict__.items()
                        if isinstance(value,torch.Tensor) and value.ndim>=2 and value.shape[1]==q.shape[1]})
                    expected=reference(selected,ro[:,ids].contiguous(),pooled=False)[0]
                    row['implementation_error']=metrics(actual[:,ids],expected)
                    assert max(row['implementation_error']['relative_l2'])<.003,row['implementation_error']
                    expected_old=execute(old,ro,pooled=False)[0]
                    row['legacy_output_error']=metrics(actual,expected_old)
                    assert max(row['legacy_output_error']['relative_l2'])<1e-6,row['legacy_output_error']
            if args.profile:
                measure(api,repeats=3);torch.cuda.profiler.start()
                for _ in range(3):api()
                torch.cuda.synchronize();torch.cuda.profiler.stop()
            else:
                row['timings']={'preparation':measure(prep),'API':measure(api)}
                if mode=='fused_sage32':row['timings']['raw']=measure(lambda:raw(p,None))
                else:
                    # Isolate identical raw device kernel from public validation/LSE work.
                    row['timings']['raw']=measure(lambda:_execute_bound_drop(p,r))
                    row['timings']['routing']=measure(lambda:route(p))
            row['status']='complete';result['cases'].append(row);save(args.output/'results.json',result)
            print('SOL_STUDY',record['layer'],mode,'complete',flush=True)
    result['status']='complete';save(args.output/'results.json',result)


if __name__=='__main__':main()
