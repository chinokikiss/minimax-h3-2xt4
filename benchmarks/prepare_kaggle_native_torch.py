"""CPU preparation: retain Kaggle Torch/CUDA, pin H3 core libraries in an overlay.

This is the explicit user-authorized Kaggle profile, not the Windows lock.
Never install or replace Torch, its matching vision/audio packages, or CUDA.
"""
import argparse
import hashlib
import importlib.metadata as md
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import traceback
import zipfile
from email.parser import BytesParser


def canonical(name):return name.lower().replace('_','-').replace('.','-')


def inventory():
    # Kaggle can retain duplicate dist-info entries. Match md.version/import
    # precedence rather than letting the last stale entry overwrite the active one.
    names={canonical(d.metadata['Name']) for d in md.distributions() if d.metadata['Name']}
    return {name:md.version(name) for name in sorted(names)}


def core_packages(packages):
    # Python/NumPy/Torch are the native Kaggle ABI family. UI distributions are
    # irrelevant to these headless operator/component jobs.
    excluded={'torch','torchvision','torchaudio','numpy','pip','setuptools'}
    return {n:v for n,v in packages.items() if n not in excluded and not n.startswith('comfyui-')}


def run(command,log,env=None,timeout=600):
    with log.open('w') as stream:
        result=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,env=env,timeout=timeout)
    if result.returncode:raise RuntimeError(log.name+' failed; exit '+str(result.returncode))


def install_overlay(wheelhouse,overlay,expected,log,expected_hashes=None):
    lines=[];found={};hashes={}
    for path in sorted(wheelhouse.glob('*.whl')):
        with path.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
        if expected_hashes is not None and expected_hashes.get(path.name)!=digest:
            raise RuntimeError('CPU-prepared wheel checksum mismatch: '+path.name)
        with zipfile.ZipFile(path) as archive:
            names=[n for n in archive.namelist() if n.endswith('.dist-info/METADATA')]
            if len(names)!=1:raise RuntimeError('Invalid wheel metadata')
            info=BytesParser().parsebytes(archive.read(names[0]))
        name=canonical(info['Name']);version=info['Version']
        if name in found or expected.get(name)!=version:raise RuntimeError('Unexpected core wheel: '+path.name)
        found[name]=version;hashes[path.name]=digest
        lines.append(str(path)+' --hash=sha256:'+digest+'\n')
    if found!=expected:raise RuntimeError('Incomplete core wheelhouse')
    requirements=wheelhouse.parent/'hashed-core-wheels.txt';requirements.write_text(''.join(lines))
    run([sys.executable,'-m','pip','install','--no-deps','--no-index','--require-hashes',
         '--target',str(overlay),'-r',str(requirements)],log)
    return hashes


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--bundle',required=True,type=Path)
    parser.add_argument('--source',required=True,type=Path);parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    receipt=dict(status='preparing',profile='kaggle-native-torch-pinned-h3-core',
        scope='CPU environment/API/I-O preflight only; no GPU or full H3 inference')
    try:
        spec=importlib.util.spec_from_file_location('frozen',args.bundle/'frozen_kaggle_setup.py')
        frozen=importlib.util.module_from_spec(spec);spec.loader.exec_module(frozen)
        frozen.require_cpu_setup();args.source.mkdir(parents=True,exist_ok=True)
        manifest=frozen.prepare_source(args.bundle,args.source)
        base=inventory();core=core_packages(manifest['packages'])
        receipt.update(base_packages=base,core_pins=core,python=sys.version,
            source_archive_sha256=manifest['source_archive_sha256'])
        if 'torch' not in base:raise RuntimeError('Kaggle image has no native Torch')
        wheelhouse=args.output/'core-wheelhouse';wheelhouse.mkdir(exist_ok=True)
        requirements=args.output/'core-pins.txt'
        requirements.write_text(''.join(n+'=='+v+'\n' for n,v in sorted(core.items())))
        print('NATIVE_TORCH_CPU_STAGE download_pinned_core',flush=True)
        run([sys.executable,'-m','pip','download','--disable-pip-version-check','--only-binary=:all:',
             '--no-deps','-r',str(requirements),'--dest',str(wheelhouse)],args.output/'core_download.log')
        overlay=Path('/kaggle/temp/h3_native_overlay')
        receipt['wheels_sha256']=install_overlay(wheelhouse,overlay,core,args.output/'core_install.log')
        if inventory()!=base:raise RuntimeError('Base Kaggle environment was modified')
        env=dict(os.environ,PYTHONPATH=os.pathsep.join([str(overlay),str(args.source),str(args.source/'ComfyUI')]),
                 COMFY_ROOT=str(args.source/'ComfyUI'),H3_RUNTIME_LOCK=str(args.output/'runtime_profile.json'))
        profile=dict(schema=1,profile=receipt['profile'],packages=dict(core),native_base_packages=base,
            native_torch=base['torch'],python_minor=f'{sys.version_info.major}.{sys.version_info.minor}',
            source_archive_sha256=manifest['source_archive_sha256'],overlay_packages=core)
        for name in ('torch','torchvision','torchaudio','numpy'):
            if name in base:profile['packages'][name]=base[name]
        (args.output/'runtime_profile.json').write_text(json.dumps(profile,indent=2))
        probe=r'''
import sys,json,inspect,importlib.metadata as md
sys.argv=['h3-preflight','--cpu']
import comfy.options
comfy.options.enable_args_parsing()
import torch,comfy_kitchen as ck,tilelang
from patches.legacy_runtime import validate_packages,validate_source
import os
validate_packages();validate_source(os.environ['COMFY_ROOT'])
from comfy.ldm.minimax import vae
signature=inspect.signature(vae.Attention.forward)
assert {'pre_norm','residual','residual_scale'}.issubset(signature.parameters)
api=dict(signature=str(signature),source=vae.__file__,
 gpu_available_on_cpu_host=ck.int8_attention_is_available(),
 qualification='CPU signature/import check only; GPU availability is checked by the actual T4 operator gate')
assert callable(ck.int8_attention) and callable(ck.int8_linear)
assert callable(ck.rms_rope_split_half_)
from comfy_kitchen.tensor.int8 import TensorWiseINT8Layout
from patches import tp_dit,tp_qwen3vl,comfy_tp_runtime,staged_vae,parallel_vae
from comfy.text_encoders.llama import Qwen3VL_32BConfig
result=dict(torch=torch.__version__,cuda=torch.version.cuda,python=sys.version,
 kitchen=md.version('comfy-kitchen'),tilelang=md.version('tilelang'),legacy_video_api=api,
 attention_signature=str(inspect.signature(ck.int8_attention)),linear_signature=str(inspect.signature(ck.int8_linear)),
 cuda_arch_flags=torch._C._cuda_getArchFlags() if hasattr(torch._C,'_cuda_getArchFlags') else None,
 core_imports='passed',tf32_default=torch.backends.cuda.matmul.allow_tf32)
print('H3_NATIVE_API_RECEIPT '+json.dumps(result))
'''
        print('NATIVE_TORCH_CPU_STAGE import_and_api',flush=True)
        run([sys.executable,'-c',probe],args.output/'api_probe.log',env,180)
        line=next(line for line in (args.output/'api_probe.log').read_text().splitlines() if line.startswith('H3_NATIVE_API_RECEIPT '))
        receipt['api']=json.loads(line.split(' ',1)[1])
        print('NATIVE_TORCH_CPU_STAGE linux_pread',flush=True)
        run([sys.executable,str(args.source/'benchmarks/check_linux_pread.py'),'--output',str(args.output/'linux_pread.json')],
             args.output/'linux_pread.log',env,120)
        tools={}
        for name in ('nvcc','nsys','ncu'):
            candidates=[p for p in (shutil.which(name),) if p]
            for folder in ('/usr/local','/opt/nvidia'):
                path=Path(folder)
                if path.exists():candidates.extend(str(p) for p in path.glob('*/bin/'+name) if p.is_file())
            tools[name]=[]
            for path in dict.fromkeys(candidates):
                try:info=subprocess.check_output([path,'--version'],stderr=subprocess.STDOUT,text=True,timeout=10)
                except Exception as error:info=str(error)
                tools[name].append(dict(path=path,version=info))
        receipt.update(core_imports='passed',linux_pread='passed',tools=tools)
        (wheelhouse/'wheelhouse_receipt.json').write_text(json.dumps(dict(core_pins=core,
            wheels_sha256=receipt['wheels_sha256'],native_torch=base['torch'],python_minor=profile['python_minor']),indent=2))
        # CPU/GPU Kaggle images need not share a Python ABI. Prepare the pinned
        # Linux wheels for all supported target minors without allocating GPUs.
        receipt['target_wheelsets']={}
        for minor in ('3.11','3.12','3.13'):
            target=Path('/kaggle/temp/h3_target_wheels')/minor
            target.mkdir(parents=True,exist_ok=True)
            print('NATIVE_TORCH_CPU_STAGE target_wheels',minor,flush=True)
            if minor==profile['python_minor']:
                for path in wheelhouse.glob('*.whl'):shutil.copyfile(path,target/path.name)
            else:
                platforms=['manylinux_2_28_x86_64','manylinux_2_27_x86_64','manylinux_2_24_x86_64',
                           'manylinux_2_17_x86_64','manylinux2014_x86_64','manylinux2010_x86_64','manylinux1_x86_64']
                command=[sys.executable,'-m','pip','download','--disable-pip-version-check','--only-binary=:all:',
                    '--no-deps','--python-version',minor,'--implementation','cp','--abi','cp'+minor.replace('.','')]
                for name in platforms:command+=['--platform',name]
                command+=['-r',str(requirements),'--dest',str(target)]
                run(command,args.output/('core_download_py'+minor+'.log'))
            hashes={}
            for path in target.glob('*.whl'):
                with path.open('rb') as stream:hashes[path.name]=hashlib.file_digest(stream,'sha256').hexdigest()
            archive=args.output/('core_wheels_py'+minor+'.tar.bin')
            with tarfile.open(archive,'w') as tar:
                for path in sorted(target.glob('*.whl')):tar.add(path,arcname=path.name)
            with archive.open('rb') as stream:archive_sha=hashlib.file_digest(stream,'sha256').hexdigest()
            receipt['target_wheelsets'][minor]=dict(archive=archive.name,sha256=archive_sha,wheels_sha256=hashes)
        # Keep output listings small: GPU jobs use the bound archives above.
        shutil.rmtree(wheelhouse)
        receipt.update(status='cpu_ready',
            qualification='CPU imports/signatures/Linux reader verified. Other Python ABI wheels are hash-bound publisher artifacts; actual GPU Torch/CUDA extension compatibility is checked by T4 operators.')
    except BaseException as error:
        receipt.update(status='blocked',error=repr(error),traceback=traceback.format_exc());raise
    finally:
        (args.output/'preflight_receipt.json').write_text(json.dumps(receipt,indent=2))
        print('NATIVE_TORCH_CPU_FINAL',receipt['status'],flush=True)


if __name__=='__main__':main()
