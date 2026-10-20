"""Fused Group32/64 validation and fair API comparisons on physical T4."""
import argparse
import json
from pathlib import Path
import random
import re
import sys
import time
from dataclasses import replace
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'benchmarks')]
import torch
from real_t4_attention_study import hardware,load,measure,label,save
from attention_root_causes import metrics
from patches.tilelang_grouped_qk import prepare_grouped,execute_grouped
from patches.tilelang_grouped_fused import prepare_grouped_fused,execute_grouped_fused,dequantized_reference
from patches.tilelang_sage_preparation import prepare_sage_fused
from patches.tilelang_dense_preparation import run_prevalidated_dense


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--process',type=int,default=0);ap.add_argument('--profile',action='store_true')
    ap.add_argument('--modes');ap.add_argument('--reference-heads',type=int,default=3)
    ap.add_argument('--save-code',action='store_true');args=ap.parse_args()
    args.output.mkdir(parents=True,exist_ok=True);torch.set_grad_enabled(False);torch.backends.cuda.matmul.allow_tf32=False
    hw=hardware();assert len(hw['devices'])==2 and all(x['capability']==[7,5] for x in hw['devices'])
    result={'status':'running','hardware':hw,'process':args.process,'cases':[],
        'API_boundary':'original captured FP32/FP16 BHND CUDA -> FP16 attention output; necessary casts and preparation included',
        'scope':'GPU0 selected native diagnostic heads, not full H56 deployment or quality acceptance',
        'contract':'GroupG per-token contiguous channels; FP32 Sol leaf-centroid mean; nearest-even INT8; int32 dot per group, FP32 scaled sum; FP16 P/V and FP32 online softmax/PV.'}
    records=json.loads((args.data/'capture_manifest.json').read_text())['records']
    for record in records:
        if args.profile and record['component']!='dit':continue
        (q,k,v,native),heads=load(args.data,record);half=[x.half() for x in [q,k,v]];causal=record['component']=='te'
        candidates=['fused_sage32','legacy_group32','fused_group32_legacy_mma']+['fused_group'+str(g) for g in [32,64,128,16] if q.shape[-1]%g==0]
        if args.profile:candidates=['fused_sage32','fused_group32','fused_group64']
        if args.modes:candidates=[name for name in candidates if name in args.modes.split(',')]
        random.Random(104+args.process).shuffle(candidates)
        for name in candidates:
            group_of=lambda value:int(re.match(r'fused_group(\d+)',value).group(1))
            def prep():
                hh=[x.half() for x in [q,k,v]]
                if name=='fused_sage32':return prepare_sage_fused(*hh)
                if name=='legacy_group32':return prepare_grouped(*hh)
                return prepare_grouped_fused(*hh,group=group_of(name))
            def execute(p):
                if name=='fused_sage32':return run_prevalidated_dense(*p,causal=causal)
                if name=='legacy_group32':return execute_grouped(p,causal)[0]
                if name=='fused_group32_legacy_mma':
                    from patches.tilelang_grouped_qk import grouped_kernel
                    return grouped_kernel(*q.shape,causal)(p.qi,p.ki,p.v,p.qs,p.ks)[0][:,:,:p.tokens]
                return execute_grouped_fused(p,group=group_of(name),causal=causal)
            row={'capture':record,'shape':list(q.shape),'head_ids':heads,'candidate':name}
            try:
                start=time.perf_counter();p=prep();actual=execute(p);torch.cuda.synchronize();row['compile_first_call_seconds']=time.perf_counter()-start
                if args.save_code and args.process==0 and name.startswith('fused_group'):
                    from patches.tilelang_grouped_fused import attention_kernel
                    group=group_of(name);code=args.output/f'{name}_b{q.shape[0]}h{q.shape[1]}n{q.shape[2]}d{q.shape[3]}'
                    compiled=attention_kernel(*q.shape,group,causal)
                    if name=='fused_group32_legacy_mma':
                        from patches.tilelang_grouped_qk import grouped_kernel
                        compiled=grouped_kernel(*q.shape,causal)
                    try:
                        if not code.with_suffix('.ptx').exists():
                            code.with_suffix('.cu').write_text(compiled.get_kernel_source())
                            compiled.export_ptx(str(code.with_suffix('.ptx')))
                        text=code.with_suffix('.ptx').read_text()
                        row['static_mma_opcodes']=sorted(set(re.findall(r'(?:w?mma\.sync[^\s;]*)',text)))
                        row['static_code_scope']='Compiled SM75 PTX; opcode presence is not runtime Tensor Core utilization.'
                    except Exception as error:row['code_export_error']=type(error).__name__
                if name.startswith('fused_group') and args.process==0:
                    group=group_of(name)
                    ids=torch.linspace(0,q.shape[1]-1,min(args.reference_heads,q.shape[1]),device=q.device).round().long()
                    selected=replace(p,**{key:value[:,ids].contiguous() for key,value in p.__dict__.items()
                        if isinstance(value,torch.Tensor) and value.ndim>=2 and value.shape[1]==q.shape[1]})
                    expected=dequantized_reference(selected,group,causal)
                    row['implementation_reference_heads']=ids.cpu().tolist()
                    row['implementation_error']=metrics(actual[:,ids],expected)
                    assert max(row['implementation_error']['relative_l2'])<.003,row['implementation_error']
                    if group==32:
                        old=prepare_grouped(*half);n=q.shape[-2]
                        row['legacy_operand_match']={
                            'qi_equal':torch.equal(old.qi[:,:,:n],p.qi[:,:,:n]),'ki_equal':torch.equal(old.ki[:,:,:n],p.ki[:,:,:n]),
                            'qs_max_diff':(old.qs[:,:,:n]-p.qs[:,:,:n]).abs().max().item(),
                            'ks_max_diff':(old.ks[:,:,:n]-p.ks[:,:,:n]).abs().max().item()}
                        assert all(row['legacy_operand_match'][x] for x in ['qi_equal','ki_equal']),row['legacy_operand_match']
                        assert row['legacy_operand_match']['qs_max_diff']==row['legacy_operand_match']['ks_max_diff']==0,row['legacy_operand_match']
                row['versus_native_diagnostic_only']=metrics(actual,native)
                def api():
                    with label(name+'/casts_and_quantization'):current=prep()
                    with label(name+'/QK_softmax_PV'):return execute(current)
                if args.profile:
                    measure(api,repeats=3);torch.cuda.profiler.start()
                    for _ in range(3):api()
                    torch.cuda.synchronize();torch.cuda.profiler.stop()
                else:row['timings']={key:measure(fn) for key,fn in [('raw',lambda:execute(p)),('preparation',prep),('API',api)]}
                row['status']='complete'
            except Exception as error:
                row.update(status='failed',error_type=type(error).__name__,error=str(error)[-4000:])
                result['cases'].append(row);save(args.output/'results.json',result)
                if name!='fused_group16':raise
                continue
            result['cases'].append(row);save(args.output/'results.json',result)
            print('GROUP',record['component'],name,'complete',flush=True)
    result['status']='complete';save(args.output/'results.json',result)


if __name__=='__main__':main()
