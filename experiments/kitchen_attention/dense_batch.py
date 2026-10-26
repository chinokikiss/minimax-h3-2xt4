"""One bounded CPU preflight and two-device dense/TE operator experiment."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import shutil
import subprocess
import sys
import traceback
import unittest

from .batch import cpu as base_cpu,save,supervise_pair


def cpu(output):
    base_cpu(output)
    state=json.loads((output/'result.json').read_text())
    try:
        names=['benchmarks.test_kitchen_te_attention','benchmarks.test_h3_configuration_defaults',
               'benchmarks.test_h3_default_fused_row','benchmarks.test_h3_l1_current_e2e']
        result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromNames(names))
        state['dense_promotion_tests']=dict(run=result.testsRun,passed=result.wasSuccessful(),errors=len(result.errors),failures=len(result.failures))
        assert result.wasSuccessful(),'TE promotion CPU contracts failed'
        from benchmarks.test_h3_operator_integration import control_gates
        control_gates();state['operator_control_gates']='passed'
    except Exception:
        state.update(status='failed',error=traceback.format_exc());raise
    finally:save(output/'result.json',state)


def gpu(output,qualification):
    import torch
    assert torch.cuda.device_count()==2
    for rank in (0,1):assert torch.cuda.get_device_capability(rank)==(7,5) and 'T4' in torch.cuda.get_device_name(rank)
    approved=json.loads(Path(qualification).read_text())
    assert approved['status']=='cpu_ready' and approved['cpu_native_compile']['status']=='passed'
    assert approved['dense_promotion_tests']['passed'] and approved['operator_control_gates']=='passed'
    nvcc=shutil.which('nvcc');version=subprocess.check_output([nvcc,'--version'],text=True,timeout=10)
    assert 'V12.8.93' in version
    state=dict(status='running',torch=torch.__version__,cuda=torch.version.cuda,compiler=version,
        packages={n:importlib.metadata.version(n) for n in ('comfy-kitchen','tilelang','numpy')},
        input_kind='Synthetic dense/TE operators, not H3 generation',ranks=[])
    target=output/'batch_results.json';save(target,state)
    build_root=Path('/kaggle/temp/h3_dense_sm75_build')
    try:
        from .build import build
        library=build(build_root,nvcc=nvcc)
        state['matrix_processes']=supervise_pair([[sys.executable,'-u','-m','experiments.kitchen_attention.dense_batch',
            '--mode','worker','--output',str(output/f'matrix_rank{rank}.json'),
            '--rank',str(rank),'--library',str(library)] for rank in (0,1)],output,'dense',560)
        state['ranks']=[json.loads((output/f'matrix_rank{rank}.json').read_text())
            if (output/f'matrix_rank{rank}.json').exists() else dict(status='missing',rank=rank) for rank in (0,1)]
        state['status']='completed' if all(r['status']=='completed' for r in state['ranks']) else 'failed'
        state['native_status']='qualified' if state['status']=='completed' else 'see_rank_gates'
    except Exception:
        state.update(status='failed',error=traceback.format_exc());raise
    finally:
        for name in ('build_receipt.json','compile.log'):
            if (build_root/name).exists():shutil.copyfile(build_root/name,output/name)
        save(target,state)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=('cpu','gpu','worker'),required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--qualification',type=Path)
    p.add_argument('--library',type=Path);p.add_argument('--rank',type=int,choices=(0,1))
    args=p.parse_args()
    if args.mode=='worker':
        from .dense_bench import run
        args.output.parent.mkdir(parents=True,exist_ok=True);run(args.library,args.output,args.rank)
    else:
        args.output.mkdir(parents=True,exist_ok=True)
        if args.mode=='cpu':cpu(args.output)
        else:gpu(args.output,args.qualification)
