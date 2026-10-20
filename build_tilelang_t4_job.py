"""Package only the TileLang experiment sources into a private Kaggle notebook."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def build(smoke=False,te_gate=False):
    folder = ROOT / ('kaggle_tilelang_t4_smoke' if smoke else ('kaggle_tilelang_t4_te_gate' if te_gate else 'kaggle_tilelang_t4_attention'))
    folder.mkdir(exist_ok=True)
    slug = 'tilelang-t4-sm75-smoke' if smoke else ('tilelang-t4-causal-gqa-gate' if te_gate else 'tilelang-t4-int8-sparse-attention')
    setup = '''import json, os, pathlib, subprocess, sys, time, traceback
OUT=pathlib.Path('/kaggle/working/tilelang_t4_results'); OUT.mkdir(exist_ok=True)
subprocess.run([sys.executable,'-m','pip','install','tilelang==0.1.15'],check=True,timeout=600)
import torch, tilelang
hw={'torch':torch.__version__,'torch_cuda':torch.version.cuda,'tilelang':tilelang.__version__,
    'gpus':[{'name':torch.cuda.get_device_name(i),'capability':list(torch.cuda.get_device_capability(i))} for i in range(torch.cuda.device_count())]}
hw['nvcc']=subprocess.run(['nvcc','--version'],capture_output=True,text=True).stdout
(OUT/'hardware.json').write_text(json.dumps(hw,indent=2)); print('HARDWARE',json.dumps(hw),flush=True)
assert torch.cuda.get_device_capability(0)==(7,5),hw
assert 'T4' in torch.cuda.get_device_name(0),hw
'''
    if smoke:
        code = '''import tilelang.language as T
@tilelang.jit(out_idx=[2], target={'kind':'cuda','arch':'sm_75'}, pass_configs={'tl.enable_async_copy':False})
def smoke_gemm(dtype):
    @T.prim_func
    def main(A:T.Tensor((32,64),dtype),B:T.Tensor((64,64),dtype),C:T.Tensor((32,64),'int32' if dtype=='int8' else 'float32')):
        with T.Kernel(1,threads=128):
            a=T.alloc_shared((32,64),dtype)
            b=T.alloc_shared((64,64),dtype)
            c=T.alloc_fragment((32,64),'int32' if dtype=='int8' else 'float32')
            T.copy(A,a); T.copy(B,b); T.clear(c)
            T.gemm(a,b,c,transpose_B=True)
            T.copy(c,C)
    return main
result={'passed':False,'cases':[]}
try:
    for dtype in ['int8','float16']:
        a=(torch.randint(-127,128,(32,64),device='cuda',dtype=torch.int8) if dtype=='int8' else torch.randn(32,64,device='cuda',dtype=torch.float16))
        b=(torch.randint(-127,128,(64,64),device='cuda',dtype=torch.int8) if dtype=='int8' else torch.randn(64,64,device='cuda',dtype=torch.float16))
        kernel=smoke_gemm(dtype); out=kernel(a,b); torch.cuda.synchronize()
        ref=a.float()@b.float().T
        error=(out.float()-ref).abs().max().item()
        (OUT/(dtype+'.cu')).write_text(kernel.get_kernel_source())
        kernel.export_ptx(str(OUT/(dtype+'.ptx')))
        torch.testing.assert_close(out.float(),ref,rtol=0 if dtype=='int8' else 1e-4,atol=0 if dtype=='int8' else 1e-4)
        result['cases'].append({'dtype':dtype,'max_abs':error}); print('SMOKE',dtype,error,flush=True)
    result['passed']=True
except Exception:
    result['traceback']=traceback.format_exc(); print(result['traceback'],flush=True)
finally:
    (OUT/'smoke.json').write_text(json.dumps(result,indent=2))
assert result['passed'],result
'''
    else:
        entry='benchmarks/tilelang_te_gate.py' if te_gate else 'benchmarks/tilelang_t4_attention.py'
        files = ['patches/tilelang_sparse_attention.py', entry]
        manifest = {}
        code = "SRC=pathlib.Path('/kaggle/working/tilelang_source'); SRC.mkdir(exist_ok=True)\n"
        for relative in files:
            content = (ROOT / relative).read_text(encoding='utf-8')
            manifest[relative] = hashlib.sha256(content.encode()).hexdigest()
            code += f"p=SRC/{relative!r}; p.parent.mkdir(parents=True,exist_ok=True); p.write_text({content!r})\n"
        code += f"(OUT/'source_manifest.json').write_text(json.dumps({manifest!r},indent=2))\n"
        code += f"subprocess.run([sys.executable,str(SRC/{entry!r}),'--output',str(OUT)],check=True,timeout=2400)\n"
    notebook = {'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}},
                'cells':[{'cell_type':'code','metadata':{},'execution_count':None,'outputs':[],'source':(setup+code).splitlines(keepends=True)}]}
    (folder/'experiment.ipynb').write_text(json.dumps(notebook),encoding='utf-8')
    metadata = dict(id='chinokiki9/'+slug,title='TileLang T4 '+('SM75 Smoke' if smoke else ('Causal GQA Gate' if te_gate else 'INT8 Sparse Attention')),
                    code_file='experiment.ipynb',language='python',kernel_type='notebook',is_private=True,
                    enable_gpu=True,enable_tpu=False,enable_internet=True,dataset_sources=[],competition_sources=[],kernel_sources=[],model_sources=[])
    (folder/'kernel-metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    print(folder)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--smoke',action='store_true');parser.add_argument('--te-gate',action='store_true')
    options=parser.parse_args();build(options.smoke,options.te_gate)
