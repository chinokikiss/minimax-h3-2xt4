"""Use a frozen local source/dependency bundle; never clone upstream or update it.

This setup/preflight requires no GPU. Unavailable wheels, resolver conflicts,
and previously unseen Linux dependencies block before package installation.
"""
import argparse
import hashlib
import importlib.metadata as md
import json
import re
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile


def canonical(name):return re.sub(r'[-_.]+','-',name).lower()


def manifest_digest(manifest):
    return hashlib.sha256(json.dumps(manifest,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def require_cpu_setup():
    try:
        probe=subprocess.run(['nvidia-smi','--query-gpu=name','--format=csv,noheader'],capture_output=True,text=True,timeout=10)
    except FileNotFoundError:return
    if probe.returncode==0 and probe.stdout.strip():
        raise RuntimeError('Dependency resolution/download must run on a CPU-only host; do not spend Kaggle GPU time on environment setup checks')


def check_resolution(report,manifest):
    expected=manifest['packages'];errors=[]
    for item in report.get('install',[]):
        name=canonical(item['metadata']['name']);version=item['metadata']['version']
        if name not in expected:errors.append('Unqualified platform dependency: '+name+'=='+version)
        elif expected[name]!=version:errors.append('Version substitution: '+name+'=='+version+'; locked '+expected[name])
        if not item.get('download_info',{}).get('archive_info',{}).get('hashes',{}).get('sha256'):
            errors.append('Missing wheel SHA256: '+name)
    if errors:raise RuntimeError('; '.join(errors))


def prepare_source(bundle,root,expected_digest=None):
    manifest=json.loads((bundle/'local_conda.json').read_text(encoding='utf-8'))
    if expected_digest and manifest_digest(manifest)!=expected_digest:
        raise RuntimeError('Attached runtime manifest differs from the experiment source binding')
    for name,digest in manifest['lock_files_sha256'].items():
        if hashlib.sha256((bundle/name).read_bytes()).hexdigest()!=digest:
            raise RuntimeError('Frozen setup/requirements checksum mismatch: '+name)
    archive=bundle/'comfy_source.tar.gz'
    if not archive.is_file():archive=bundle/'comfy_source.tar.gz.bin'
    if hashlib.sha256(archive.read_bytes()).hexdigest()!=manifest['source_archive_sha256']:
        raise RuntimeError('Frozen source archive checksum mismatch')
    expected=manifest['source_files_sha256']
    with tarfile.open(archive,'r:gz') as tar:
        names={m.name.removeprefix('ComfyUI/') for m in tar.getmembers()}
        if names!=set(expected) or any(not m.isfile() or not m.name.startswith('ComfyUI/') or '..' in Path(m.name).parts for m in tar.getmembers()):
            raise RuntimeError('Frozen source archive inventory mismatch')
        for member in tar.getmembers():
            data=tar.extractfile(member).read()
            if hashlib.sha256(data).hexdigest()!=expected[member.name.removeprefix('ComfyUI/')]:
                raise RuntimeError('Frozen source file checksum mismatch: '+member.name)
        tar.extractall(root,filter='data')
    runtime=root/'runtime';runtime.mkdir(exist_ok=True)
    for name in ('local_conda.json','requirements-local-conda.txt','constraints-local-conda.txt'):
        shutil.copyfile(bundle/name,runtime/name)
    return manifest


def offline_install(bundle,manifest):
    import email.parser
    import zipfile
    wheelhouse=bundle/'linux-wheelhouse'
    receipt=json.loads((wheelhouse/'wheelhouse_receipt.json').read_text(encoding='utf-8'))
    if receipt['manifest_sha256']!=manifest_digest(manifest):raise RuntimeError('Wheelhouse belongs to a different runtime lock')
    lines=[];found={}
    for name,digest in receipt['wheels_sha256'].items():
        if Path(name).name!=name or not name.endswith('.whl'):raise RuntimeError('Invalid wheelhouse entry')
        wheel=wheelhouse/name
        if hashlib.sha256(wheel.read_bytes()).hexdigest()!=digest:raise RuntimeError('Wheel checksum mismatch: '+name)
        with zipfile.ZipFile(wheel) as archive:
            metadata=[p for p in archive.namelist() if p.endswith('.dist-info/METADATA')]
            if len(metadata)!=1:raise RuntimeError('Invalid wheel metadata: '+name)
            info=email.parser.BytesParser().parsebytes(archive.read(metadata[0]))
        package=canonical(info['Name']);version=info['Version']
        if package in found or manifest['packages'].get(package)!=version:
            raise RuntimeError('Wheelhouse version/inventory differs from frozen packages: '+name)
        found[package]=version
        lines.append(str(wheel.resolve())+' --hash=sha256:'+digest+'\n')
    if found!=manifest['packages']:raise RuntimeError('Wheelhouse does not cover all frozen packages')
    # The mounted dataset is read-only; put the pip input in the working folder.
    import tempfile
    with tempfile.TemporaryDirectory() as folder:
        requirements=Path(folder)/'wheels.txt';requirements.write_text(''.join(lines),encoding='utf-8')
        subprocess.run([sys.executable,'-m','pip','install','--no-index','--no-deps','--require-hashes','-r',str(requirements)],check=True,timeout=300)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--bundle',type=Path)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--install',action='store_true')
    parser.add_argument('--offline-install',action='store_true',help='GPU jobs: install only preverified wheels from the attached dataset')
    parser.add_argument('--export-wheelhouse',type=Path,help='CPU-only Linux preparation destination; does not qualify T4 kernels')
    parser.add_argument('--expected-manifest-sha256')
    args=parser.parse_args()
    if args.offline_install and (args.install or args.export_wheelhouse):parser.error('Offline and online preparation modes are mutually exclusive')
    if args.bundle is None:
        matches=list(Path('/kaggle/input').rglob('local_conda.json'))
        if len(matches)!=1:raise RuntimeError('Attach exactly one frozen user runtime dataset; no upstream/download fallback is allowed')
        args.bundle=matches[0].parent
    args.root.mkdir(parents=True,exist_ok=True)
    manifest=prepare_source(args.bundle,args.root,args.expected_manifest_sha256)
    expected_python='.'.join(manifest['python_version'].split('.')[:2])
    if f'{sys.version_info.major}.{sys.version_info.minor}'!=expected_python:
        raise RuntimeError('Use the locked Python minor version '+expected_python)
    receipt=dict(status='source_verified',profile=manifest['profile'],source_archive_sha256=manifest['source_archive_sha256'])
    receipt_path=args.root/'runtime/setup_receipt.json'
    try:
        if args.install or args.export_wheelhouse:
            require_cpu_setup()
            # pip may resolve metadata, but cannot install until every candidate
            # is checked against the user snapshot. No broad requirements file.
            report=args.root/'runtime/resolution.json'
            command=[sys.executable,'-m','pip','install','--dry-run','--ignore-installed','--only-binary=:all:',
                '--report',str(report),'--index-url','https://pypi.org/simple',
                '--extra-index-url','https://download.pytorch.org/whl/cu132',
                '-r',str(args.root/'runtime/requirements-local-conda.txt'),
                '-c',str(args.root/'runtime/constraints-local-conda.txt')]
            subprocess.run(command,check=True,timeout=900)
            resolution=json.loads(report.read_text());check_resolution(resolution,manifest)
            wheels=args.root/'runtime/resolved-wheels.txt'
            wheels.write_text(''.join(item['download_info']['url']+'#sha256='+item['download_info']['archive_info']['hashes']['sha256']+'\n'
                for item in resolution.get('install',[])),encoding='utf-8')
            if args.export_wheelhouse:
                if sys.platform!='linux':raise RuntimeError('Prepare the Linux wheelhouse on a CPU-only Linux host')
                args.export_wheelhouse.mkdir(parents=True,exist_ok=True)
                subprocess.run([sys.executable,'-m','pip','download','--no-deps','--require-hashes','-r',str(wheels),
                    '--dest',str(args.export_wheelhouse)],check=True,timeout=1200)
                wheel_receipt=dict(manifest_sha256=manifest_digest(manifest),python_minor=expected_python,
                    wheels_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in args.export_wheelhouse.glob('*.whl')})
                (args.export_wheelhouse/'wheelhouse_receipt.json').write_text(json.dumps(wheel_receipt,indent=2),encoding='utf-8')
                receipt.update(status='wheelhouse_prepared',qualification='Exact-version Linux wheel resolution only; T4 operator qualification pending.')
                if not args.install:return
            if args.install:subprocess.run([sys.executable,'-m','pip','install','--no-deps','--require-hashes','-r',str(wheels)],check=True,timeout=1200)
        if args.offline_install:offline_install(args.bundle,manifest)
        actual={}
        for name,required in manifest['packages'].items():
            try:actual[name]=md.version(name)
            except md.PackageNotFoundError:actual[name]='missing'
        mismatch={n:dict(expected=v,actual=actual[n]) for n,v in manifest['packages'].items() if actual[n]!=v}
        if mismatch:raise RuntimeError('Runtime does not match the frozen conda environment: '+json.dumps(mismatch))
        receipt.update(status='packages_match',packages=actual,
            qualification='Environment preflight only; Linux kernels/T4 numerical and performance qualification still required.')
        print('FROZEN_RUNTIME_READY',manifest['profile'],flush=True)
    except BaseException as error:
        receipt.update(status='blocked',reason=str(error));raise
    finally:receipt_path.write_text(json.dumps(receipt,indent=2),encoding='utf-8')


if __name__=='__main__':main()
