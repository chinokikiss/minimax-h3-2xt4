"""CPU-only Linux platform candidate, without changing the Windows snapshot.

Resolve metadata first, prove all added packages are reachable dependencies of
the frozen packages, then install only exact SHA256-bound wheels into an
isolated CPython 3.11.15. This is preparation, never SM75 qualification.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
from urllib.request import urlopen

PYTHON_URL='https://github.com/astral-sh/python-build-standalone/releases/download/20260320/cpython-3.11.15%2B20260320-x86_64-unknown-linux-gnu-install_only_stripped.tar.gz'
PYTHON_SHA256='67e5923c364e3190af2068188040b8173f2259549be4bef07f503be8bdb3bf68'


def run(command, log, timeout):
    with log.open('w') as stream:
        result=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,timeout=timeout)
    if result.returncode:
        raise RuntimeError(log.name+' exited '+str(result.returncode))


def validate_platform_resolution(report, manifest):
    try:
        from packaging.markers import default_environment
        from packaging.requirements import Requirement
        from packaging.utils import canonicalize_name
    except ModuleNotFoundError:
        from pip._vendor.packaging.markers import default_environment
        from pip._vendor.packaging.requirements import Requirement
        from pip._vendor.packaging.utils import canonicalize_name
    rows={canonicalize_name(row['metadata']['name']):row for row in report['install']}
    if len(rows)!=len(report['install']):raise RuntimeError('Duplicate package names in resolution')
    expected=manifest['packages']
    missing=set(expected)-set(rows)
    if missing:raise RuntimeError('Resolution omitted frozen packages: '+str(sorted(missing)))
    for name,version in expected.items():
        if rows[name]['metadata']['version']!=version:
            raise RuntimeError('Version substitution forbidden: '+name)
    environment=default_environment()
    if environment['platform_system']!='Linux' or environment['python_full_version']!='3.11.15':
        raise RuntimeError('Resolve only on Linux CPython 3.11.15')
    # Prove the platform extension contains dependencies, not arbitrary extras.
    pending=[(name,'') for name in expected];visited=set()
    while pending:
        name,extra=pending.pop()
        if (name,extra) in visited:continue
        visited.add((name,extra))
        for text in rows[name]['metadata'].get('requires_dist',[]):
            requirement=Requirement(text)
            if requirement.marker and not requirement.marker.evaluate(dict(environment,extra=extra)):
                continue
            dependency=canonicalize_name(requirement.name)
            if dependency not in rows:
                raise RuntimeError('Incomplete dependency closure: '+text)
            if requirement.specifier and not requirement.specifier.contains(rows[dependency]['metadata']['version']):
                raise RuntimeError('Unsatisfied dependency: '+text)
            pending.append((dependency,''))
            pending.extend((dependency,e) for e in requirement.extras)
    if set(rows)-{n for n,e in visited}:
        raise RuntimeError('Resolved additions are not reachable from frozen packages')
    for name,row in rows.items():
        info=row['download_info']
        if not info['url'].split('#')[0].endswith('.whl') or not info.get('archive_info',{}).get('hashes',{}).get('sha256'):
            raise RuntimeError('Only hashed publisher wheels allowed: '+name)
    return {n:r['metadata']['version'] for n,r in rows.items()}


def bootstrap(output):
    archive=output/'python311.tar.gz.bin'
    with urlopen(PYTHON_URL,timeout=60) as response, archive.open('wb') as stream:
        shutil.copyfileobj(response,stream)
    if hashlib.sha256(archive.read_bytes()).hexdigest()!=PYTHON_SHA256:
        raise RuntimeError('Pinned Python binary digest mismatch')
    target=Path('/kaggle/temp/h3_frozen_linux');target.mkdir(parents=True,exist_ok=True)
    with tarfile.open(archive,'r:gz') as tar:tar.extractall(target,filter='data')
    python=target/'python/bin/python3.11'
    version=subprocess.check_output([str(python),'-c','import platform;print(platform.python_version())'],text=True).strip()
    if version!='3.11.15':raise RuntimeError('Python bootstrap version mismatch: '+version)
    return python


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--bundle',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--resolved',action='store_true')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    spec=importlib.util.spec_from_file_location('frozen_setup',args.bundle/'frozen_kaggle_setup.py')
    frozen=importlib.util.module_from_spec(spec);spec.loader.exec_module(frozen)
    frozen.require_cpu_setup()
    source=Path('/kaggle/temp/h3_frozen_source');source.mkdir(parents=True,exist_ok=True)
    manifest=frozen.prepare_source(args.bundle,source)
    receipt=dict(scope='CPU-only Linux platform candidate; not T4 qualification',
        status='preparing',manifest_sha256=frozen.manifest_digest(manifest),
        original_packages=manifest['packages'],python_url=PYTHON_URL,python_sha256=PYTHON_SHA256)
    destination=args.output/'candidate_receipt.json'
    try:
        if not args.resolved:
            python=bootstrap(args.output)
            # Re-enter with the exact interpreter. No packages installed yet.
            run([str(python),str(Path(__file__).resolve()),'--bundle',str(args.bundle),
                 '--output',str(args.output),'--resolved'],args.output/'prepare.log',1800)
            return
        resolution=args.output/'resolution.json'
        command=[sys.executable,'-m','pip','install','--dry-run','--ignore-installed','--only-binary=:all:',
            '--disable-pip-version-check','--report',str(resolution),
            '--index-url','https://pypi.org/simple','--extra-index-url','https://download.pytorch.org/whl/cu132',
            '-r',str(args.bundle/'requirements-local-conda.txt'),'-c',str(args.bundle/'constraints-local-conda.txt')]
        run(command,args.output/'resolution.log',900)
        report=json.loads(resolution.read_text());packages=validate_platform_resolution(report,manifest)
        added={n:v for n,v in packages.items() if n not in manifest['packages']}
        receipt.update(platform_additions=added,packages=packages)
        # This is an explicit new Linux supplement, never a rewrite of local_conda.json.
        lock=dict(schema=1,base_manifest_sha256=frozen.manifest_digest(manifest),
                  platform='Linux x86_64 CPython 3.11.15',platform_additions=added,packages=packages)
        lock_path=args.output/'linux_platform_candidate.json'
        lock_path.write_text(json.dumps(lock,indent=2),encoding='utf-8')
        requirements=args.output/'resolved-wheels.txt'
        requirements.write_text(''.join(row['download_info']['url']+' --hash=sha256:'+
            row['download_info']['archive_info']['hashes']['sha256']+'\n' for row in report['install']))
        wheelhouse=args.output/'linux-wheelhouse';wheelhouse.mkdir(exist_ok=True)
        run([sys.executable,'-m','pip','download','--no-deps','--require-hashes','-r',str(requirements),
             '--dest',str(wheelhouse)],args.output/'wheel_download.log',1200)
        combined=dict(manifest,packages=packages)
        wheel_receipt=dict(manifest_sha256=frozen.manifest_digest(combined),
            base_manifest_sha256=frozen.manifest_digest(manifest),
            platform_lock_sha256=hashlib.sha256(lock_path.read_bytes()).hexdigest(),
            wheels_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in wheelhouse.glob('*.whl')})
        (wheelhouse/'wheelhouse_receipt.json').write_text(json.dumps(wheel_receipt,indent=2))
        frozen.offline_install(args.output,combined)
        run([sys.executable,'-m','pip','check'],args.output/'pip_check.log',60)
        probe="""
import importlib,importlib.metadata as md,json,platform
result={'python':platform.python_version(),'imports':{}}
for name in ('torch','torchvision','torchaudio','comfy_kitchen','tilelang'):
 try:
  module=importlib.import_module(name)
  result['imports'][name]={'status':'ok','version':getattr(module,'__version__',None)}
 except Exception as error:result['imports'][name]={'status':'failed','reason':type(error).__name__+': '+str(error)}
print(json.dumps(result,indent=2))
if any(v['status']!='ok' for v in result['imports'].values()):raise SystemExit(2)
"""
        run([sys.executable,'-c',probe],args.output/'import_probe.log',180)
        receipt.update(status='cpu_prepared',qualification='Exact packages and CPU imports only; T4/operator/compiler qualification pending')
    except BaseException as error:
        receipt.update(status='blocked',reason=type(error).__name__+': '+str(error))
        print(json.dumps(receipt,indent=2),flush=True)
        raise
    finally:
        # The child writes the authoritative preparation receipt.
        if args.resolved or not destination.exists():
            destination.write_text(json.dumps(receipt,indent=2),encoding='utf-8')


if __name__=='__main__':main()
