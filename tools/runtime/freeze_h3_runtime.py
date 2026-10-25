"""Snapshot the user's existing conda runtime; never install or upgrade packages."""
import argparse
import hashlib
import importlib.metadata as md
import json
from pathlib import Path
import platform
import subprocess
import sys
import tarfile
from packaging.markers import default_environment
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from patches.legacy_runtime import validate_source,COMFY_REVISION,SOURCE_HASHES

ROOT=Path(__file__).resolve().parents[2]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=ROOT/'runtime')
    parser.add_argument('--bundle',type=Path,default=Path('D:/attention_eval/h3_conda_runtime_20261019'))
    args=parser.parse_args()
    if Path(sys.prefix).name.lower()!='comfyui' or not (Path(sys.prefix)/'conda-meta').is_dir():
        raise RuntimeError('Freeze from the existing conda comfyui environment, not the experimental venv')
    source=Path(validate_source(ROOT/'ComfyUI'))
    roots={Requirement(line).name for line in (source/'requirements.txt').read_text(encoding='utf-8').splitlines()
        if line.strip() and not line.startswith('#')}
    roots.update(['torchaudio','tilelang','huggingface-hub','nvidia-ml-py','packaging','setuptools','pip'])
    snapshot={canonicalize_name(d.metadata['Name']):md.version(d.metadata['Name']) for d in md.distributions() if d.metadata['Name']}
    pending=list(roots);packages={};missing=set()
    environments=[default_environment()]
    environments.append(environments[0]|dict(sys_platform='linux',platform_system='Linux',os_name='posix',platform_machine='x86_64'))
    while pending:
        name=canonicalize_name(pending.pop())
        if name in packages or name in missing:continue
        if name not in snapshot:missing.add(name);continue
        packages[name]=snapshot[name]
        for raw in md.requires(name) or []:
            requirement=Requirement(raw)
            if not requirement.marker or any(requirement.marker.evaluate(e|{'extra':''}) for e in environments):
                pending.append(requirement.name)
    manifest=dict(schema=1,profile='user-conda-comfyui',python=sys.executable,prefix=sys.prefix,
        python_version=platform.python_version(),comfy_revision=COMFY_REVISION,comfy_source=str(source),
        comfy_source_normalized_sha256=SOURCE_HASHES,packages=dict(sorted(packages.items())),
        unresolved_platform_dependencies=sorted(missing),
        qualification='Local conda baseline. Linux/T4 equivalence must be checked; no implicit package substitutions.',
        policy='No upstream updates, unpinned installs, package upgrades, or automatic version substitutions.')
    args.output.mkdir(parents=True,exist_ok=True);args.bundle.mkdir(parents=True,exist_ok=True)
    archive=args.bundle/'comfy_source.tar.gz'
    tracked=subprocess.check_output(['git','-C',str(source),'ls-files','-z']).decode().split('\0')
    identities={}
    for name in sorted(n for n in tracked if n):
        path=source/name
        if not path.is_file() or path.is_symlink():raise RuntimeError('Unsupported source entry: '+name)
        identities[name]=hashlib.sha256(path.read_bytes()).hexdigest()
    existing_path=args.output/'local_conda.json'
    existing=json.loads(existing_path.read_text(encoding='utf-8')) if existing_path.is_file() else None
    if existing and (existing['packages']!=manifest['packages'] or existing['source_files_sha256']!=identities):
        raise RuntimeError('Source/packages drifted from the frozen baseline. Preserve it; create a separately reviewed baseline instead of overwriting this lock.')
    if archive.exists() and existing:
        if hashlib.sha256(archive.read_bytes()).hexdigest()!=existing['source_archive_sha256']:
            raise RuntimeError('Existing frozen archive changed; preserve and investigate it')
    else:
        with tarfile.open(archive,'w:gz') as tar:
            for name in identities:tar.add(source/name,arcname='ComfyUI/'+name,recursive=False)
    manifest['source_files_sha256']=identities
    manifest['source_archive_sha256']=hashlib.sha256(archive.read_bytes()).hexdigest()
    lock_files={
        'requirements-local-conda.txt':''.join(f'{n}=={v}\n' for n,v in sorted(packages.items())),
        'constraints-local-conda.txt':''.join(f'{n}=={v}\n' for n,v in sorted(snapshot.items())),
        'frozen_kaggle_setup.py':(ROOT/'benchmarks/frozen_kaggle_setup.py').read_text(encoding='utf-8')}
    manifest['lock_files_sha256']={name:hashlib.sha256(content.encode()).hexdigest() for name,content in lock_files.items()}
    encoded=json.dumps(manifest,indent=2)
    for directory in (args.output,args.bundle):
        (directory/'local_conda.json').write_text(encoded,encoding='utf-8')
        for name,content in lock_files.items():
            if directory==args.output and name=='frozen_kaggle_setup.py':continue
            (directory/name).write_text(content,encoding='utf-8',newline='\n')
    # Publish only this flat directory. Evidence/preflight folders contain other
    # manifests and must not be uploaded as runtime data. .bin avoids dataset
    # services automatically unpacking the source archive during processing.
    publish=args.bundle/'kaggle_bundle';publish.mkdir(exist_ok=True)
    import shutil
    shutil.copyfile(archive,publish/'comfy_source.tar.gz.bin')
    (publish/'local_conda.json').write_text(encoded,encoding='utf-8')
    for name,content in lock_files.items():(publish/name).write_text(content,encoding='utf-8',newline='\n')
    print('FROZEN_CONDA_RUNTIME',sys.executable,len(packages),'packages',len(identities),'source files',args.bundle)
    if missing:print('UNRESOLVED_PLATFORM_DEPENDENCIES',sorted(missing))


if __name__=='__main__':main()
