"""CPU gate and bounded two-device subprocess controller for this experiment."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
import unittest


def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(value,indent=2)+'\n');temporary.replace(path)


def cpu(output):
    import torch
    from .attention import runtime_contract
    from .derive import verify_vendor
    assert not torch.cuda.is_available(),'CPU qualification must not allocate a GPU'
    torch.set_num_threads(2)
    state=dict(status='running',vendor=verify_vendor(),
        versions={n:importlib.metadata.version(n) for n in ('torch','comfy-kitchen','tilelang')})
    try:
        from . import test_contracts,test_matrix
        suite=unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromModule(m) for m in (test_contracts,test_matrix))
        result=unittest.TextTestRunner(verbosity=2).run(suite)
        state['tests']=dict(run=result.testsRun,errors=len(result.errors),failures=len(result.failures),passed=result.wasSuccessful())
        assert result.wasSuccessful(),'New CPU contracts failed'
        kitchen=runtime_contract(require_extension=False)
        state['cuda_extension_loadable_on_cpu']=hasattr(kitchen._cuda_backend._C,'sage_sdpa_quantize')
        import inspect
        from patches import tilelang_grouped_tp_kitchen_v2,tilelang_grouped_tp_optimized,tilelang_attention_p2
        state['target_signatures']={name:str(inspect.signature(fn)) for name,fn in (
            ('group32_prepare',tilelang_grouped_tp_kitchen_v2.prepare),
            ('group32_execute',tilelang_grouped_tp_kitchen_v2.execute),
            ('sparse_execute',tilelang_grouped_tp_optimized.execute),
            ('split_execute',tilelang_attention_p2.execute))}
        nvcc=shutil.which('nvcc')
        if nvcc:
            version=subprocess.check_output([nvcc,'--version'],text=True,timeout=10)
            state['compiler']=dict(path=nvcc,version=version)
            if 'V12.8.93' not in version:
                raise RuntimeError('CPU image compiler differs from the approved CUDA12.8 toolchain; no substitution')
            from .build import build
            library=build(output/'native_cpu_build',nvcc=nvcc)
            state['cpu_native_compile']=dict(status='passed',library_sha256=hashlib.sha256(library.read_bytes()).hexdigest())
        else:
            from .cpu_compiler import prepare
            nvcc,archives=prepare('/kaggle/temp/h3_cuda128_cpu')
            state['cpu_build_tool_archives']=archives
            from .build import build
            library=build(output/'native_cpu_build',nvcc=str(nvcc))
            state['cpu_native_compile']=dict(status='passed',library_sha256=hashlib.sha256(library.read_bytes()).hexdigest(),
                qualification='Exact CUDA12.8.93 compiler and headers extracted for CPU build only; native packages unchanged')
        state['status']='cpu_ready'
    except Exception:
        state.update(status='failed',error=traceback.format_exc());raise
    finally:save(output/'result.json',state)


def supervise_pair(commands,output,label,budget):
    started=time.monotonic();processes=[];rows=[]
    try:
        for rank,command in enumerate(commands):
            stream=(output/f'{label}_rank{rank}.log').open('w')
            proc=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
            processes.append((proc,stream))
        for rank,(proc,stream) in enumerate(processes):
            try:code=proc.wait(timeout=max(.01,budget-(time.monotonic()-started)))
            except subprocess.TimeoutExpired:code=None
            rows.append(dict(rank=rank,returncode=code,deadline_exceeded=code is None))
    finally:
        import signal
        for proc,stream in processes:
            if proc.poll() is None:
                try:os.killpg(proc.pid,signal.SIGKILL)
                except ProcessLookupError:pass
                proc.wait(timeout=10)
            stream.close()
    return rows


def gpu(output,qualification):
    import torch
    assert torch.cuda.device_count()==2,'Two physical T4 GPUs required'
    for rank in (0,1):
        assert torch.cuda.get_device_capability(rank)==(7,5) and 'T4' in torch.cuda.get_device_name(rank)
    approved=json.loads(Path(qualification).read_text())
    assert approved['status']=='cpu_ready'
    assert approved['cpu_native_compile']['status']=='passed'
    from .derive import verify_vendor
    assert approved['vendor']==verify_vendor()
    nvcc=shutil.which('nvcc');assert nvcc,'Existing nvcc12.8 required'
    version=subprocess.check_output([nvcc,'--version'],text=True,timeout=10)
    assert 'V12.8.93' in version,'Do not substitute a different CUDA compiler'
    # Parent owns no tensors; both physical workers have independent contexts.
    state=dict(status='running',torch=torch.__version__,cuda=torch.version.cuda,compiler=version,
        packages={n:importlib.metadata.version(n) for n in ('comfy-kitchen','tilelang','numpy')},
        input_kind='Synthetic operator benchmark; not a generation or trajectory quality experiment')
    target=output/'batch_results.json';save(target,state)
    try:
        from .build import build
        library=None;build_root=Path('/kaggle/temp/h3_native_matrix_build')
        try:library=build(build_root,nvcc=nvcc)
        except Exception:
            state['native_build_error']=traceback.format_exc()
        finally:
            for name in ('build_receipt.json','compile.log'):
                if (build_root/name).exists():shutil.copyfile(build_root/name,output/name)
        if library is None:
            state['native_gate_status']=['compile_failed','compile_failed'];state['native_qualified']=False
            state['gate_processes']=[]
        else:
            state['gate_processes']=supervise_pair([[sys.executable,'-u','-m','experiments.kitchen_attention.qualify',
                '--library',str(library),'--output',str(output/f'native_gate_rank{rank}.json'),
                '--device',str(rank),'--deadline','120'] for rank in (0,1)],output,'native_gate',150)
        qualified=[]
        for rank in (0,1):
            file=output/f'native_gate_rank{rank}.json'
            receipt=json.loads(file.read_text()) if file.exists() else dict(status='missing')
            qualified.append(receipt.get('status')=='passed')
        native_ok=all(qualified)
        state.update(native_qualified=native_ok,native_gate_status=qualified);save(target,state)
        # Isolate unqualified native failures before running any baseline timing.
        commands=[[sys.executable,'-u','-m','experiments.kitchen_attention.batch','--mode','worker',
            '--output',str(output),'--rank',str(rank),'--library',str(library),
            *(['--native-ok'] if native_ok else [])] for rank in (0,1)]
        state['matrix_processes']=supervise_pair(commands,output,'matrix',820)
        state['ranks']=[json.loads((output/f'matrix_rank{rank}.json').read_text())
            if (output/f'matrix_rank{rank}.json').exists() else dict(status='missing',rank=rank) for rank in (0,1)]
        state['status']='completed' if all(r.get('status','').startswith('completed') for r in state['ranks']) else 'failed'
        state['native_status']='qualified' if native_ok else 'rejected_no_native_timing'
    except Exception:
        state.update(status='failed',error=traceback.format_exc());raise
    finally:save(target,state)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--mode',choices=('cpu','gpu','worker'),required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--qualification',type=Path)
    parser.add_argument('--library',type=Path);parser.add_argument('--rank',type=int,choices=(0,1));parser.add_argument('--native-ok',action='store_true')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    if args.mode=='cpu':cpu(args.output)
    elif args.mode=='gpu':gpu(args.output,args.qualification)
    else:
        from .matrix import run_matrix
        run_matrix(args.library,args.output/f'matrix_rank{args.rank}.json',args.rank,args.native_ok)
