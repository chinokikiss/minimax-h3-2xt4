"""Lower the four real TP2 SwiGLU/padding layouts for offline SM75 compilation."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import tilelang
from tilelang import tvm
from tilelang.engine import lower
from patches.tilelang_swiglu_pad import kernel


def inspect(function,**kw):
    target=tvm.target.Target(kw['target'])
    with target,tvm.transform.PassContext(config=kw['pass_configs']):return lower(function,target=target)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',required=True,type=Path);a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    tilelang.compile=inspect;records=[]
    for component,rows,width,stride in (('te',353,12800,12800),('dit',6814,7168,14336)):
        for rank in (0,1):
            artifact=kernel(rows,width,stride,stride,2,rank,'sm_75')
            path=a.output/f'{component}_rank{rank}.cu';path.write_text(artifact.kernel_source,encoding='utf8')
            records.append(dict(component=component,rank=rank,M=rows,local_width=width,global_width=width*2,input_row_stride=stride,
                source=path.name,barrier_source_sites=artifact.kernel_source.count('__syncthreads()'),status='lowered'))
            print('SWIGLU_SM75_LOWER',component,rank,flush=True)
    (a.output/'lowering.json').write_text(json.dumps(dict(cases=records,scope='CPU codegen only'),indent=2))


if __name__=='__main__':main()
