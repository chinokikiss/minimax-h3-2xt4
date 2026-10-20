"""Second-pass exact-shape operator study; optional isolated SM120 smoke run."""
import argparse
import datetime
import gc
import hashlib
import importlib
import json
import os
from pathlib import Path
import random
import sys
import time
import traceback
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'benchmarks')]
import torch
import torch.distributed as dist
import comfy_kitchen as ck
from tp2_kitchen_attention import identity,label,save
from tp2_timing import measure


def local_modules(output):
    from local_kitchen_preview import adapt
    _,previous=adapt(output,'pack')
    path=ROOT/'patches/tilelang_grouped_tp_kitchen_v2.py';source=path.read_text(encoding='utf8')
    source=source.replace('from patches.','from preview_patches.').replace("'sm_75'","'sm_120'").replace('!=(7,5)','!=(12,0)')
    source=source.replace('Actual SM75','Local SM120 preview')
    (output/'adapted/preview_patches'/path.name).write_text(source,encoding='utf8')
    from tilelang.contrib import nvcc
    nvcc.py_str=lambda x:x.decode('utf8',errors='replace') if isinstance(x,bytes) else str(x)
    compile_cuda=nvcc.compile_cuda
    def compatible(code,*pos,**kw):
        if 'tl::pack_float16x4(' in code:
            helper='''template <typename H> __device__ __forceinline__ unsigned long long preview_pack_float16x4(H x,H y,H z,H w){
  static_assert(sizeof(H)==2,"16-bit representation required");
  unsigned long long a=*((unsigned short*)&x), b=*((unsigned short*)&y), c=*((unsigned short*)&z), d=*((unsigned short*)&w);
  return a|(b<<16)|(c<<32)|(d<<48);
}
'''
            name=hashlib.sha256(code.encode()).hexdigest()[:16]
            (output/f'{name}_original.cu').write_text(code,encoding='utf8')
            code=code.replace('tl::pack_float16x4(','preview_pack_float16x4(');at=code.index('extern "C"');code=code[:at]+helper+code[at:]
            (output/f'{name}_compatible.cu').write_text(code,encoding='utf8')
        return compile_cuda(code,*pos,**kw)
    nvcc.compile_cuda=compatible
    return previous,importlib.import_module('preview_patches.tilelang_grouped_tp_kitchen_v2')


def gate_units(previous,new,device):
    rows=[];torch.manual_seed(20261017)
    for dtype,d,n,nk,h,hk,tile in ((torch.float32,64,129,129,4,1,128),(torch.float16,128,97,65,2,2,64)):
        q=torch.randn(1,h,n,d,device=device,dtype=dtype);k=torch.randn(1,hk,nk,d,device=device,dtype=dtype)
        storage=torch.randn(1,nk,3*hk*d,device=device,dtype=dtype);v=storage.view(1,nk,3,hk,d)[:,:,2].permute(0,2,1,3)
        expected=previous.prepare(q,k,v,query_tile=tile)
        for mode in ('eager','castpad'):
            p=new.prepare(q,k,v,query_tile=tile,mean_mode=mode);checks=identity(p,expected)
            assert all(checks.values()),checks
            rows.append(dict(test='ragged_GQA_packedV',dtype=str(dtype),mean_mode=mode,operand_identity=checks))
    q=torch.randn(1,2,65,64,device=device,dtype=torch.float16);k=torch.randn_like(q);v=torch.randn_like(q)
    for value in (float('nan'),float('inf'),-float('inf')):
        changed=v.clone();changed[:,:,0,0]=value;p=previous.prepare(q,k,changed,query_tile=64)
        expected=torch.nan_to_num(previous.execute(p,layout='BND'))
        actual=new.execute(p,layout='BND',finite_output=True)
        assert torch.equal(actual.view(torch.uint8),expected.view(torch.uint8))
        rows.append(dict(test='finite_epilogue',input_value=str(value),bitwise_equal=True))
    return rows


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',required=True,type=Path);ap.add_argument('--matrix',required=True,type=Path)
    ap.add_argument('--output',required=True,type=Path);ap.add_argument('--selection',type=Path);ap.add_argument('--round',type=int,default=0)
    ap.add_argument('--profile',action='store_true');ap.add_argument('--components',nargs='+',default=['te','dit','video'])
    ap.add_argument('--local',action='store_true');ap.add_argument('--logical-rank',type=int,default=0);a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    rank=a.logical_rank if a.local else int(os.environ['LOCAL_RANK']);device=0 if a.local else rank
    torch.cuda.set_device(device);torch.set_grad_enabled(False);torch.backends.cuda.matmul.allow_tf32=False
    if a.local:
        assert torch.cuda.get_device_capability()==(12,0);previous,new=local_modules(a.output)
    else:
        assert torch.cuda.device_count()==2 and all(torch.cuda.get_device_capability(i)==(7,5) and 'T4' in torch.cuda.get_device_name(i) for i in (0,1))
        from patches import tilelang_grouped_tp_kitchen as previous,tilelang_grouped_tp_kitchen_v2 as new
        dist.init_process_group('nccl',device_id=torch.device('cuda',device),timeout=datetime.timedelta(seconds=90))
    barrier=lambda:dist.barrier() if dist.is_initialized() else None
    result=dict(status='running',rank=rank,hardware=torch.cuda.get_device_name(device),capability=list(torch.cuda.get_device_capability(device)),torch=torch.__version__,local_preview=a.local,cases=[],
        scope='Current TP2 dense operators only. Previous T4 winners as baseline; frozen Group32 numerics. No H3 pipeline, sparse policy change or KV exchange.')
    target=a.output/f'rank{rank}.json';save(target,result)
    matrix=json.loads(a.matrix.read_text())['configurations'];selection=json.loads(a.selection.read_text()) if a.selection else None
    if not selection:
        result['unit_gates']=gate_units(previous,new,device);save(target,result);print('V2_UNIT_GATES_PASS',rank,flush=True)
    records=json.loads((a.data/'capture_manifest.json').read_text())['records']
    files=dict(te='te_layer00_call-1.pt',dit='dit_layer25_call09.pt',video='video_layer00_call00.pt')
    for component in a.components:
        record=next(r for r in records if r['file']==files[component]);path=a.data/record['file']
        with path.open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==record['sha256']
        data=torch.load(path,map_location='cpu',weights_only=True)
        if component=='te':values=[data['q'][:,rank*32:(rank+1)*32],data['k'][:,rank*4:(rank+1)*4],data['v'][:,rank*4:(rank+1)*4]];offset=0
        elif component=='dit':values=[data[x][:,rank*28:(rank+1)*28] for x in ('q','k','v')];offset=0
        else:
            lo,hi=((0,1024),(1024,1797))[rank];values=[data['q'][:,:,lo:hi],data['k'],data['v']];offset=lo
        original=[x.to(device).contiguous() for x in values];del values,data;gc.collect()
        if component=='dit':
            b,h,n,d=original[2].shape;storage=torch.empty((b,n,3*h*d),device=device,dtype=original[2].dtype)
            packed=storage.view(b,n,3,h,d)[:,:,2].permute(0,2,1,3);packed.copy_(original[2]);original[2]=packed
        b,h,n,d=original[0].shape;causal=component=='te';base_cfg=next(c for c in matrix[component] if c.get('baseline'))
        base=previous.prepare(*original,query_tile=base_cfg['tile'])
        baseline_kw=dict(causal=causal,query_offset=offset,cache=base_cfg['cache'],layout='BND',output_dtype=original[0].dtype,threads=base_cfg['threads'],warp_policy=base_cfg['warp'])
        expected=previous.execute(base,**baseline_kw)
        if component=='video':expected=torch.nan_to_num(expected)
        configs=list(matrix[component])
        if selection:configs=[c for c in configs if c['name'] in ('current',selection['winners'][component]['name'],'kitchen')]
        random.Random(20261017+a.round).shuffle(configs)
        for cfg in configs:
            name=component+'/'+cfg['name'];native=cfg.get('native',False)
            def prep():
                with label(name+'/preparation'):
                    if cfg['prep']=='previous':return previous.prepare(*original,query_tile=cfg['tile'])
                    return new.prepare(*original,query_tile=cfg['tile'],mean_mode='castpad' if cfg['prep']=='castpad' else 'eager')
            def raw(p):
                with label(name+'/QK_softmax_PV_and_output'):
                    kw=dict(causal=causal,query_offset=offset,cache=cfg['cache'],layout='BND',output_dtype=original[0].dtype,threads=cfg['threads'],warp_policy=cfg['warp'],finite_output=cfg.get('finite',False))
                    out=new.execute(p,**kw,row_scores=cfg.get('row_scores',False),staged_scales=cfg.get('staged_scales',False)) if cfg.get('row_scores') or cfg.get('staged_scales') or cfg.get('finite') else previous.execute(p,**kw)
                    return torch.nan_to_num(out) if component=='video' and not cfg.get('finite') else out
            def api():
                with label(name+'/API'):
                    if native:
                        out=ck.int8_attention(*original);return torch.nan_to_num(out.transpose(1,2).reshape(b,n,h*d).contiguous())
                    return raw(prep())
            row=dict(component=component,name=cfg['name'],config=cfg,q_shape=list(original[0].shape),k_shape=list(original[1].shape),input_strides=[list(x.stride()) for x in original],fixture_sha256=record['sha256'])
            error=None;p=None;actual=None;barrier();start=time.perf_counter()
            try:
                if not native:
                    p=prep();actual=raw(p);own=previous.prepare(*original,query_tile=cfg['tile'])
                    row['operand_identity']=identity(p,own);assert all(row['operand_identity'].values()),row['operand_identity'];del own
                    torch.testing.assert_close(actual,expected,rtol=.001,atol=.0002)
                    row['bitwise_equal_to_current']=torch.equal(actual.view(torch.uint8),expected.view(torch.uint8))
                else:actual=api()
                torch.cuda.synchronize();row['first_call_including_JIT_and_gate_seconds']=time.perf_counter()-start
                assert torch.isfinite(actual).all();row['output_sha256']=hashlib.sha256(actual.cpu().contiguous().numpy().tobytes()).hexdigest()
            except Exception:error=traceback.format_exc()[-6000:]
            errors=[error]
            if dist.is_initialized():errors=[None,None];dist.all_gather_object(errors,error)
            if any(errors):
                row.update(status='failed',errors=errors);result['cases'].append(row);save(target,result);print('V2_FAILED',rank,name,errors,flush=True);continue
            if a.profile:
                for _ in range(5):api()
                torch.cuda.synchronize();barrier();torch.cuda.profiler.start()
                for _ in range(3):api()
                torch.cuda.synchronize()
                with label('profiling/rank_completion'):barrier()
                torch.cuda.profiler.stop()
            else:
                if not native and not selection:
                    row['preparation']=measure(prep,warmup=5,repeats=10)
                    row['attention_and_output']=measure(lambda:raw(p),warmup=5,repeats=10)
                row['API']=measure(api,warmup=5 if a.local else 10,repeats=10 if a.local else 15)
            row['status']='complete';result['cases'].append(row);save(target,result)
            print('V2_COMPLETE',rank,name,row.get('API',{}).get('gpu_median_ms'),flush=True);del p,actual;gc.collect()
        del base,expected,original;gc.collect();torch.cuda.empty_cache()
    result['status']='complete';save(target,result)
    all_results=[result]
    if dist.is_initialized():all_results=[None,None];dist.all_gather_object(all_results,result)
    if (rank==0 or a.local) and not a.profile:
        winners={}
        for component in a.components:
            maps=[{r['name']:r for r in x['cases'] if r['component']==component and r['status']=='complete' and not r['config'].get('native')} for x in all_results]
            common=set.intersection(*(set(m) for m in maps));scores={name:max(m[name]['API']['gpu_median_ms'] for m in maps) for name in common}
            assert 'current' in scores,'Baseline failed';name=min(scores,key=scores.get)
            winners[component]=dict(name=name,slower_rank_API_ms=scores[name],all_slower_rank_API_ms=scores)
        save(a.output/'selection.json',dict(winners=winners,scope='Group32 only; Kitchen numerics excluded; no production switch.'))
    if dist.is_initialized():barrier();dist.destroy_process_group()


if __name__=='__main__':main()
