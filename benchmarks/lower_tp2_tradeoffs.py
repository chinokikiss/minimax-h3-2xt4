"""Offline SM75 lowering for the bounded numerical/transport candidate suite."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import tilelang
from tilelang import tvm
from tilelang.engine import lower
from patches import tilelang_attention_tradeoffs as a,tilelang_transport_tradeoffs as c


def inspect(fn,**kw):
    target=tvm.target.Target(kw['target'])
    with target,tvm.transform.PassContext(config=kw['pass_configs']):return lower(fn,target=target)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',required=True,type=Path);args=ap.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    tilelang.compile=inspect;rows=[]
    cases=[('fp16_encode',c.fp16_codec,(6814,5376,12)),('fp16_decode',c.fp16_codec,(6814,5376,12,True)),
           ('int8_pack',c.pack_int8,(6814,5376)),('int8_unpack',c.unpack_int8,(6814,5376)),
           ('routes',a.route_compactor,(1,28,107,107)),('value128',a.value_quantizer,(1,28,6848,128)),
           ('value64',a.value_quantizer,(1,32,1856,64))]
    for comp,h,hk,n,nk,d,tile,cache,threads,warp,causal in [('te',32,4,353,353,128,32,'group',128,'square',True),
            ('dit',28,28,6814,6814,128,64,'q',256,'row',False),('video',32,32,1024,1797,64,64,'group',128,'square',False)]:
        cases.append((comp+'_pv8',a.kernel,(1,h,hk,n,nk,d,'dense',causal,0,tile,cache,'BND','float32' if comp!='video' else 'float16',False,threads,warp,True,False)))
    cases.append(('drop_compact',a.kernel,(1,28,28,6814,6814,128,'drop',False,0,32,'group','BND','float32',False,128,'square',False,True)))
    cases.append(('scalar_pv8',a.kernel,(1,28,28,6814,6814,128,'scalar',False,0,64,'group','BND','float32',False,128,'square',True,False)))
    for name,fn,values in cases:
        row=dict(name=name)
        try:
            artifact=fn(*values);path=args.output/(name+'.cu');path.write_text(artifact.kernel_source,encoding='utf8')
            row.update(status='lowered',source=path.name)
        except Exception as e:row.update(status='failed',error=type(e).__name__+': '+str(e))
        rows.append(row);(args.output/'lowering.json').write_text(json.dumps(dict(cases=rows),indent=2));print(name,row['status'],row.get('error','')[-3500:],flush=True)
    assert all(x['status']=='lowered' for x in rows)


if __name__=='__main__':main()
