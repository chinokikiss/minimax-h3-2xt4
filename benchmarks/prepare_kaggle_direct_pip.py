"""Zero-mounted-data bootstrap: frozen source and exact, hashed PyPI wheels.

The native Python/NumPy/Torch/CUDA family is retained. Packages already at the
required version are reused; differences install into an isolated overlay.
No dependency resolver, source build, upstream update or platform replacement.
"""
import argparse
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import traceback
try:
    from .prepare_kaggle_native_torch import inventory
except ImportError:
    from prepare_kaggle_native_torch import inventory


PROTECTED={'torch','torchvision','torchaudio','numpy','pip','setuptools'}


def requirements(lock,base,minor):
    pins=lock['core_pins']
    if PROTECTED&pins.keys() or any(n.startswith('cuda-') or (n.startswith('nvidia-') and n!='nvidia-ml-py') for n in pins):
        raise ValueError('Native platform packages must not enter the core overlay')
    if pins.get('comfy-kitchen')!='0.2.37' or pins.get('tilelang')!='0.1.13':
        raise ValueError('Preserved H3 core pins changed')
    if minor not in lock['requirements']:raise RuntimeError('Unqualified Python ABI: '+minor)
    changed={n:v for n,v in pins.items() if base.get(n)!=v}
    return changed,''.join(lock['requirements'][minor][n]+'\n' for n in sorted(changed))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--distribution',type=Path,required=True)
    parser.add_argument('--source',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--mode',choices=('cpu','e2e','operators','dit-profile','review-operators','vae-parallel'),required=True)
    parser.add_argument('--review-suite',action='store_true')
    parser.add_argument('--qualification',type=Path);args=parser.parse_args()
    out=args.output;out.mkdir(parents=True,exist_ok=True)
    receipt=dict(status='preparing',mode=args.mode,delivery='zero Kaggle dataset/kernel mounts; exact direct pip')
    phase='start';stopped=threading.Event();started=time.monotonic()
    def save():(out/'setup_receipt.json').write_text(json.dumps(receipt,indent=2))
    def mark(name):
        nonlocal phase
        phase=name;receipt['phase']=name;save();print('LIGHT_SETUP_STAGE',name,flush=True)
    def heartbeat():
        while not stopped.wait(60):print('LIGHT_SETUP_HEARTBEAT',phase,round(time.monotonic()-started),flush=True)
    threading.Thread(target=heartbeat,daemon=True).start()
    def run(argv,label,env=None,timeout=180):
        mark(label)
        with (out/(label+'.log')).open('w') as stream:
            result=subprocess.run(argv,stdout=stream,stderr=subprocess.STDOUT,env=env,timeout=timeout)
        if result.returncode:
            print((out/(label+'.log')).read_text(errors='replace')[-5000:],flush=True)
            raise RuntimeError(label+' failed, exit '+str(result.returncode))
    try:
        manifest=json.loads((args.distribution/'delivery_manifest.json').read_text())
        lock=json.loads((args.distribution/'direct_core_requirements.json').read_text())
        receipt['capsule_sha256']=manifest['capsule_sha256']
        receipt['source_hashes']=manifest['source_hashes']
        if args.mode=='cpu':
            from frozen_kaggle_setup import require_cpu_setup
            require_cpu_setup()
        else:
            q=json.loads(args.qualification.read_text())
            if (q.get('status')!='cpu_ready' or q['capsule_sha256']!=manifest['capsule_sha256']
                    or q['core_pins']!=lock['core_pins']):raise RuntimeError('Missing matching CPU qualification')
            if args.mode=='dit-profile' and not q.get('worker_cpu_probe',{}).get('passed'):
                raise RuntimeError('Missing native Python concurrent CPU profiler qualification')
            if args.mode=='review-operators' and not q.get('review_suite',{}).get('passed'):
                raise RuntimeError('Missing review optimization CPU qualification')
        mark('frozen_source')
        spec=importlib.util.spec_from_file_location('frozen_delivery',args.distribution/'frozen/frozen_kaggle_setup.py')
        frozen=importlib.util.module_from_spec(spec);spec.loader.exec_module(frozen)
        frozen.prepare_source(args.distribution/'frozen',args.source)
        base=inventory();minor=f'{sys.version_info.major}.{sys.version_info.minor}'
        changed,req=requirements(lock,base,minor)
        overlay=Path('/kaggle/temp/h3_direct_overlay');overlay.mkdir(parents=True,exist_ok=True)
        target=out/'direct-core.txt';target.write_text(req)
        receipt.update(core_pins=lock['core_pins'],native_base_packages=base,install_packages=changed,
            reuse_native_exact_packages=sorted(set(lock['core_pins'])-set(changed)),python=sys.version)
        if changed:
            run([sys.executable,'-m','pip','install','--disable-pip-version-check','--no-input','--no-deps',
                '--only-binary=:all:','--require-hashes','--index-url','https://pypi.org/simple',
                '--target',str(overlay),'-r',str(target)],'direct_pip',timeout=480)
        if inventory()!=base:raise RuntimeError('Native package environment changed')
        profile=dict(profile='kaggle-native-torch-direct-pinned-core',packages=dict(lock['core_pins']),native_base_packages=base)
        for name in ('torch','torchvision','torchaudio','numpy'):
            if name in base:profile['packages'][name]=base[name]
        (out/'runtime_profile.json').write_text(json.dumps(profile,indent=2))
        env=dict(os.environ,PYTHONPATH=os.pathsep.join([str(overlay),str(args.source),str(args.source/'ComfyUI'),str(args.source/'benchmarks')]),
            COMFY_ROOT=str(args.source/'ComfyUI'),H3_RUNTIME_LOCK=str(out/'runtime_profile.json'),
            OMP_NUM_THREADS='2',NCCL_P2P_DISABLE='1',NCCL_DEBUG='WARN',NCCL_MIN_NCHANNELS='4',NCCL_MAX_NCHANNELS='4',PYTHONUNBUFFERED='1')
        # Keep token/auth values in process memory. Persist only paths needed
        # by the next supervised execution phase, never the inherited env.
        safe=('PYTHONPATH','COMFY_ROOT','H3_RUNTIME_LOCK','OMP_NUM_THREADS','NCCL_P2P_DISABLE','NCCL_DEBUG','NCCL_MIN_NCHANNELS','NCCL_MAX_NCHANNELS','PYTHONUNBUFFERED')
        (out/'execution_env.json').write_text(json.dumps({k:env[k] for k in safe},indent=2))
        tree=ast.parse((args.source/'benchmarks/prepare_kaggle_native_torch.py').read_text())
        probe=next(n.value.value for n in ast.walk(tree) if isinstance(n,ast.Assign)
            and any(isinstance(t,ast.Name) and t.id=='probe' for t in n.targets))
        run([sys.executable,'-c',probe],'api_probe',env)
        line=next(s for s in (out/'api_probe.log').read_text().splitlines() if s.startswith('H3_NATIVE_API_RECEIPT '))
        receipt['api']=json.loads(line.split(' ',1)[1])
        if args.mode=='cpu':
            tests="""import sys,unittest,importlib.util
sys.argv=['light-tests','--cpu']
import comfy.options
comfy.options.enable_args_parsing()
import comfy.cli_args
sys.argv=['light-tests']
names=['benchmarks.test_h3_row_partial_combined','benchmarks.test_h3_reference_loading',
 'benchmarks.test_h3_loading_stages.ComponentAndIdle','benchmarks.test_h3_l1_current_e2e',
 'benchmarks.test_t4_notebook_watchdog','benchmarks.test_direct_pip_delivery',
 'benchmarks.test_bounded_column_int8']
if importlib.util.find_spec('benchmarks.test_dit_temporaries') is not None:names+=['benchmarks.test_dit_temporaries']
if importlib.util.find_spec('benchmarks.test_dit_single_profile') is not None:names+=['benchmarks.test_dit_single_profile','benchmarks.test_video_decode_memory']
if importlib.util.find_spec('benchmarks.test_tp2_bundle') is not None:names+=['benchmarks.test_tp2_bundle','benchmarks.test_worker_trace']
if importlib.util.find_spec('benchmarks.test_tp2_disk_bake') is not None:names+=['benchmarks.test_tp2_disk_bake','benchmarks.test_h3_l1_turbo_e2e']
if importlib.util.find_spec('benchmarks.test_temporal_vae') is not None:names+=['benchmarks.test_temporal_vae','benchmarks.test_temporal_delivery']
result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromNames(names))
assert result.wasSuccessful()
"""
            run([sys.executable,'-c',tests],'cpu_tests',env,180)
            if args.review_suite:
                run([sys.executable,str(args.source/'benchmarks/run_review_cpu.py'),
                    '--output',str(out/'review_cpu_tests.json')],'review_cpu_tests',env,180)
                receipt['review_suite']=json.loads((out/'review_cpu_tests.json').read_text())
            probe_tests=(out/'cpu_tests.log').read_text()
            probe_tested=all(name in probe_tests for name in (
                'test_resident_profile_clock_remains_valid_with_concurrent_disk_thread',
                'test_incremental_cpu_evidence_survives_without_finish','test_corrupt_profile_times_are_rejected'))
            receipt['worker_cpu_probe']=dict(passed=probe_tested,
                qualification='Concurrent disk-thread wall-clock validity, live-profile disabled, incremental output, invalid-clock rejection and one-attempt claims tested in CPU suite',
                python=sys.version)
            if (args.source/'benchmarks/test_tp2_disk_bake.py').exists():
                receipt['disk_bake_cpu']=dict(passed=all(name in probe_tests for name in (
                    'test_disk_bake_is_native_exact_and_has_no_remerge_after_base_payload_removal',
                    'test_cache_recipe_and_corruption_are_rejected',
                    'test_failure_never_publishes_a_complete_descriptor',
                    'test_real_header_when_bound_in_frozen_capsule')),
                    qualification='Native CPU fixture equivalence, atomic persistence, no inference remerge, immutable Turbo/base header geometry; not physical T4 bake acceptance')
            run([sys.executable,str(args.source/'benchmarks/check_linux_pread.py'),'--output',str(out/'linux_pread.json')],
                'linux_pread',env,60)
            if (args.source/'benchmarks/launch_dit_profile.py').exists():
                run([sys.executable,str(args.source/'benchmarks/launch_dit_profile.py'),
                    '--output',str(out/'profiler_availability_cpu'),'--tools-only'],'profiler_tool_versions',env,45)
            receipt.update(status='cpu_ready',qualification='Direct-pip/source/API/CPU tests only; physical T4 execution remains separate')
        else:receipt['status']='ready_for_device_gate'
    except BaseException as error:
        receipt.update(status='blocked',error=repr(error));(out/'exception.txt').write_text(traceback.format_exc());raise
    finally:
        stopped.set();receipt['seconds']=time.monotonic()-started;save()
        print('LIGHT_SETUP_FINAL',receipt['status'],flush=True)


if __name__=='__main__':main()
