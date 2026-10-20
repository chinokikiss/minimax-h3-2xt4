"""Static SM75 lowering only: no local GPU allocation or timing."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import tilelang
from tilelang.engine import lower
from tilelang import tvm
from patches.tilelang_grouped_tp import kernel


def inspect(func,**kwargs):
    target=tvm.target.Target(kwargs['target'])
    with target,tvm.transform.PassContext(config=kwargs['pass_configs']):return lower(func,target=target)


if __name__=='__main__':
    tilelang.compile=inspect
    out=Path('D:/attention_eval/operator_optimization_tp2_20261016/static_sm75');out.mkdir(exist_ok=True)
    for mode in ('dense','drop','scalar'):
        for cached in (False,True):
            # Small ragged length retains all control flow without expensive JIT.
            h,hk=(4,1) if mode=='dense' else (2,2)
            artifact=kernel(1,h,hk,65,129 if mode=='dense' else 65,128,mode,mode=='dense',0,cached)
            path=out/f'{mode}_cached{int(cached)}.cu';path.write_text(artifact.kernel_source)
            print('STATIC_LOWERED',path.name,len(artifact.kernel_source),flush=True)
