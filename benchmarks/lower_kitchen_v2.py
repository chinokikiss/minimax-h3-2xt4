"""CPU-only SM75 source screen for the second pass."""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import tilelang
from tilelang.engine import lower
from tilelang import tvm
from patches import tilelang_grouped_tp_kitchen_v2 as candidate


def inspect(func,**kw):
    target=tvm.target.Target(kw['target'])
    with target,tvm.transform.PassContext(config=kw['pass_configs']):return lower(func,target=target)


def main():
    output=Path('D:/attention_eval/operator_optimization_tp2_20261017/kitchen_secondpass/offline');output.mkdir(parents=True,exist_ok=True)
    tilelang.compile=inspect;rows=[]
    for component in ('te','dit','video'):
        for rank in ((0,1) if component=='video' else (0,)):
            h,hk,n,nk,d,dtype=(32,4,353,353,128,'float32') if component=='te' else ((28,28,6814,6814,128,'float32') if component=='dit' else (32,32,1024 if rank==0 else 773,1797,64,'float16'))
            tile,cache,threads,warp=(64,'q',256,'row') if component=='dit' else ((64,'group',128,'square') if component=='video' else (32,'group',128,'square'))
            qs=(h*n*d,n*d,d);ks=(hk*nk*d,nk*d,d);vs=(nk*3*h*d,d,3*h*d) if component=='dit' else ks
            cases=[('register_prep',candidate.fused_quantizer,(1,h,hk,n,nk,d,dtype,*qs,*ks,*vs,tile)),
                   ('castpad',candidate.cast_pad_key,(1,hk,nk,d,dtype,*ks))]
            if component=='video':cases.append(('finite_epilogue',candidate.kernel,(1,h,hk,n,nk,d,'dense',False,1024 if rank else 0,tile,cache,'BND',dtype,True,threads,warp,False,False)))
            if component=='dit':
                for name,row,stage in (('row_scores',True,False),('staged_scales',False,True)):
                    cases.append((name,candidate.kernel,(1,h,hk,n,nk,d,'dense',False,0,tile,cache,'BND',dtype,False,threads,warp,row,stage)))
            for name,fn,values in cases:
                entry=dict(component=component,rank=rank,name=name)
                try:
                    artifact=fn(*values);source=output/f'{component}_{rank}_{name}.cu';source.write_text(artifact.kernel_source,encoding='utf8')
                    entry.update(status='lowered',source=source.name,barrier_source_sites=artifact.kernel_source.count('__syncthreads()'))
                except Exception as e:entry.update(status='failed',error=(type(e).__name__+': '+str(e))[-4000:])
                rows.append(entry);print('LOWER_V2',component,rank,name,entry,flush=True)
                (output/'lowering.json').write_text(json.dumps(dict(cases=rows),indent=2))


if __name__=='__main__':main()
