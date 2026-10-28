"""Explicit VAE-only CUDA 13 overlay, after the unchanged native/core setup."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback
from prepare_kaggle_native_torch import inventory


def requirements(lock):
    assert lock['format']=='h3-vae-cu130-v1' and lock['torch']=='2.11.0+cu130'
    allowed={'torch','torchvision','torchaudio','triton'}
    lines=[]
    for name,row in sorted(lock['packages'].items()):
        if name not in allowed and not name.startswith(('nvidia-','cuda-')):
            raise ValueError('Unexpected platform replacement: '+name)
        if len(row['sha256'])!=64 or not row['url'].startswith(('https://download-r2.pytorch.org/whl/cu130/','https://files.pythonhosted.org/packages/')):
            raise ValueError('Unapproved wheel identity')
        lines.append(f"{name} @ {row['url']} --hash=sha256:{row['sha256']}\n")
    return ''.join(lines)


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--cpu',action='store_true')
    p.add_argument('--qualification',type=Path);args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    state=dict(status='starting',scope='VAE only; no DiT stack replacement')
    def save():(args.output/'setup_receipt.json').write_text(json.dumps(state,indent=2))
    try:
        source=args.source.resolve();lock_path=source/'vae_cuda13_lock.json'
        lock=json.loads(lock_path.read_text());digest=hashlib.sha256(lock_path.read_bytes()).hexdigest()
        manifest=json.loads((source.parent/'h3_light_distribution/delivery_manifest.json').read_text())
        state.update(lock_sha256=digest,capsule_sha256=manifest['capsule_sha256'])
        if not args.cpu:
            q=json.loads(args.qualification.read_text())
            assert q['status']=='cuda13_cpu_ready' and q['lock_sha256']==digest and q['capsule_sha256']==manifest['capsule_sha256']
        base=inventory()
        state['native_platform']={n:base.get(n) for n in ('torch','torchvision','torchaudio','numpy')}
        native_env=json.loads(Path('/kaggle/working/light_setup/execution_env.json').read_text())
        overlay=Path('/kaggle/temp/h3_vae_cu130_overlay');overlay.mkdir(exist_ok=False)
        target=args.output/'platform.txt';target.write_text(requirements(lock))
        state['status']='installing_hashed_platform';save();print('VAE_CU13_INSTALL_START',flush=True)
        with (args.output/'pip.log').open('w') as log:
            subprocess.run([sys.executable,'-m','pip','install','--disable-pip-version-check','--no-input',
                '--no-deps','--no-cache-dir','--only-binary=:all:','--require-hashes','--target',str(overlay),'-r',str(target)],
                stdout=log,stderr=subprocess.STDOUT,check=True,timeout=900)
        if inventory()!=base:raise RuntimeError('Kaggle base environment was modified')
        original=json.loads(Path(native_env['H3_RUNTIME_LOCK']).read_text())
        profile=dict(original,profile='vae-only-explicit-cu130-hashed-overlay')
        profile['packages']=dict(original['packages'])
        profile['packages'].update({n:lock['packages'][n]['version'] for n in ('torch','torchvision','torchaudio')})
        runtime=args.output/'runtime_profile.json';runtime.write_text(json.dumps(profile,indent=2))
        env=dict(os.environ);env.update(native_env)
        env['PYTHONPATH']=str(overlay)+os.pathsep+env['PYTHONPATH']
        libraries=sorted({str(p.parent) for p in (overlay/'nvidia').rglob('lib*.so*')})
        env['LD_LIBRARY_PATH']=os.pathsep.join(libraries+[str(overlay/'torch/lib'),env.get('LD_LIBRARY_PATH','')])
        env['H3_RUNTIME_LOCK']=str(runtime)
        # Keep the validated 12.8 compiler for any unchanged TileLang JIT. This
        # experiment upgrades Torch's runtime, not the system driver or nvcc.
        safe=dict(native_env,PYTHONPATH=env['PYTHONPATH'],LD_LIBRARY_PATH=env['LD_LIBRARY_PATH'],H3_RUNTIME_LOCK=str(runtime))
        (args.output/'execution_env.json').write_text(json.dumps(safe,indent=2))
        with (args.output/'probe.log').open('w') as log:
            subprocess.run([sys.executable,'-u',str(source/'benchmarks/run_vae_dispatch_ablation.py'),
                '--source',str(source),'--output',str(args.output/'probe'),'--cpu-preflight'],
                env=env,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=180)
        state.update(status='cuda13_cpu_ready' if args.cpu else 'ready_for_device_gate',
                     probe=json.loads((args.output/'probe/result.json').read_text()),native_environment_unchanged=True,
                     packages={n:r['version'] for n,r in lock['packages'].items()})
    except BaseException as error:
        state.update(status='blocked',error=repr(error));(args.output/'exception.txt').write_text(traceback.format_exc());raise
    finally:save()


if __name__=='__main__':main()
