"""Inspect SM75 lowering without claiming local SM75 hardware execution."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import tilelang
from tilelang.engine import lower
from tilelang import tvm
from patches.tilelang_sparse_attention import get_kernel


def inspect(func,**kwargs):
    target=tvm.target.Target(kwargs['target'])
    with target, tvm.transform.PassContext(config=kwargs['pass_configs']):
        return lower(func,target=target)


if __name__=='__main__':
    # Replace compilation only within this diagnostic process.
    tilelang.compile=inspect
    output=Path('tilelang_t4_results/local_lowering');output.mkdir(parents=True,exist_ok=True)
    for d in [64,128]:
        for pooled,pv8 in [(False,False),(True,False),(True,True)]:
            artifact=get_kernel(1,2,65,d,pooled,pv8)
            path=output/f'd{d}_pool{int(pooled)}_pv8{int(pv8)}.cu'
            path.write_text(artifact.kernel_source)
            print(path,len(artifact.kernel_source),flush=True)
