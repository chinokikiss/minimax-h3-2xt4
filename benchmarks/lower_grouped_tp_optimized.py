"""Offline SM75 codegen at actual TP2 shapes; never executes a GPU kernel."""
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import tilelang
from tilelang.engine import lower
from tilelang import tvm
from patches.tilelang_grouped_tp_optimized import kernel


def inspect(func,**kwargs):
    target=tvm.target.Target(kwargs['target'])
    with target,tvm.transform.PassContext(config=kwargs['pass_configs']):return lower(func,target=target)


def main():
    tilelang.compile=inspect
    out=Path('D:/attention_eval/operator_optimization_tp2_20261016/offline_optimization');out.mkdir(parents=True,exist_ok=True)
    cases=[('video0',32,32,1024,1797,64,'dense',False,0,64,'group','BND','float16'),
           ('video0_finite64',32,32,1024,1797,64,'dense',False,0,64,'group','BND','float16'),
           ('video1',32,32,773,1797,64,'dense',False,1024,128,'q','BND','float16'),
           ('video1_safe64',32,32,773,1797,64,'dense',False,1024,64,'group','BND','float16'),
           ('te',32,4,353,353,128,'dense',True,0,64,'q','BND','float32'),
           ('dit_dense',28,28,6814,6814,128,'dense',False,0,32,'q','BND','float32'),
           ('dit_drop',28,28,6814,6814,128,'drop',False,0,32,'group','BND','float32'),
           ('dit_scalar',28,28,6814,6814,128,'scalar',False,0,64,'group','BND','float32')]
    result=[]
    for name,*args in cases:
        args.append('finite' in name)
        a=kernel(1,*args);file=out/(name+'.cu');file.write_text(a.kernel_source)
        attrs=[{str(k):str(v) for k,v in f.attrs.items()} for f in a.device_mod.functions.values() if f.attrs is not None]
        result.append({'name':name,'arguments':args,'source':file.name,'bytes':file.stat().st_size,'status':'lowered','device_attributes':attrs})
        print('STATIC_SM75',name,file.stat().st_size,flush=True)
    (out/'lowering.json').write_text(json.dumps({'scope':'Static CUDA generation only; no GPU timings/execution','cases':result},indent=2))


if __name__=='__main__':main()
