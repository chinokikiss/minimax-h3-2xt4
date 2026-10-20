"""Static SM75 lowering for resources and instruction-mapping inspection."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import tilelang
from tilelang import tvm
from tilelang.engine import lower
from patches import tilelang_attention_p2 as a,tilelang_transport_p2 as c,tilelang_grouped_tp_kitchen as baseline


def compile_static(fn,**kw):
    target=tvm.target.Target(kw['target'])
    with target,tvm.transform.PassContext(config=kw['pass_configs']):return lower(fn,target=target)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',required=True,type=Path);args=ap.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    tilelang.compile=compile_static;records=[]
    cases=[('fp16_fused',c.epilogue,(6814,5376,'fp16',12),{}),('int8_fused',c.epilogue,(6814,5376,'int8_gather',12),{}),
           ('centroid_scores',a.centroid_scores,(1,28,6848,107,128),{})]
    cases.append(('dense_fp16_baseline',baseline.kernel,(1,28,28,6814,6814,128,'dense',False,0,64,'q','BND','float32',False,256,'row'),{}))
    cases.append(('scalar_fp16_baseline',baseline.kernel,(1,28,28,6814,6814,128,'scalar',False,0,64,'group','BND','float32',False,128,'square'),{}))
    for name,mode,threads,pv8,unit,running,pooled in [('dense_pv8_256','dense',256,True,False,False,False),
            ('dense_pv8_128','dense',128,True,False,False,False),('dense_unit_restore','dense',256,True,True,False,False),
            ('dense_running','dense',256,True,False,True,False),('scalar_pv8','scalar',128,True,False,False,False),
            ('scalar_pooled_fp16','scalar',128,False,False,False,True),('scalar_pooled_pv8','scalar',128,True,False,False,True)]:
        values=(1,28,28,6814,6814,128,mode,False,0,64,'q' if mode=='dense' else 'group','BND','float32',False,threads,'row' if mode=='dense' else 'square',pv8,False,unit,running,pooled)
        cases.append((name,a.kernel,values,{}))
    for name,fn,values,kw in cases:
        row=dict(name=name)
        try:
            artifact=fn(*values,**kw);p=args.output/(name+'.cu');p.write_text(artifact.kernel_source,encoding='utf8')
            row.update(status='lowered',source=p.name)
        except Exception as e:row.update(status='failed',error=type(e).__name__+': '+str(e))
        records.append(row);(args.output/'lowering.json').write_text(json.dumps(records,indent=2));print(name,row['status'],row.get('error','')[-1500:],flush=True)


if __name__=='__main__':main()
