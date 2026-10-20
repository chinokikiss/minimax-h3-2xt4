"""Offline source generation at verified TP2 shapes; never launches a GPU."""
import argparse
import json
from pathlib import Path
import sys
sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parent)]
import tilelang
from tilelang.engine import lower
from tilelang import tvm
from patches import tilelang_grouped_tp_kitchen as candidate
from tp2_kitchen_matrix import configurations


def inspect(func,**kw):
    target=tvm.target.Target(kw['target'])
    with target,tvm.transform.PassContext(config=kw['pass_configs']):return lower(func,target=target)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',required=True,type=Path);a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    tilelang.compile=inspect;rows=[]
    for component in ('te','dit','video'):
        for rank in ((0,1) if component=='video' else (0,)):
            h,hk,n,nk,d,dtype=(32,4,353,353,128,'float32') if component=='te' else ((28,28,6814,6814,128,'float32') if component=='dit' else (32,32,1024 if rank==0 else 773,1797,64,'float16'))
            configs=configurations(component)
            cases=[]
            for cfg in configs:
                if cfg.get('native') or cfg.get('baseline'):continue
                if cfg['name']=='fused_prepare':
                    qs=(h*n*d,n*d,d);ks=(hk*nk*d,nk*d,d)
                    vs=(nk*3*h*d,d,3*h*d) if component=='dit' else ks
                    cases.append((cfg['name']+'_prep',candidate.fused_quantizer,
                        (1,h,hk,n,nk,d,dtype,*qs,*ks,*vs,cfg['tile'])))
                else:
                    cases.append((cfg['name'],candidate.kernel,
                        (1,h,hk,n,nk,d,'dense',component=='te',1024 if component=='video' and rank else 0,
                         cfg['tile'],cfg['cache'],'BND',dtype,False,cfg['threads'],cfg['warp'])))
            for name,function,values in cases:
                record=dict(component=component,rank=rank,name=name)
                try:
                    artifact=function(*values);path=a.output/f'{component}_{rank}_{name}.cu';path.write_text(artifact.kernel_source)
                    record.update(status='lowered',source=path.name,bytes=path.stat().st_size)
                except Exception as e:record.update(status='failed',error=(type(e).__name__+': '+str(e))[-3000:])
                rows.append(record);print('STATIC_SM75',component,rank,name,record['status'],flush=True)
                if record['status']=='failed':print(record['error'],flush=True)
    (a.output/'lowering.json').write_text(json.dumps(dict(scope='CPU codegen only',cases=rows),indent=2))


if __name__=='__main__':main()
