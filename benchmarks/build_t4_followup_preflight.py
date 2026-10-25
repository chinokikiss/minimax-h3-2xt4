"""Build a new private CPU preflight, preserving every historical submission."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--owner',required=True)
    parser.add_argument('--frozen-bundle',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--candidate',action='store_true',help='CPU-only isolated exact-version Linux preparation')
    args=parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Use a fresh output directory; never overwrite a submitted source')
    dataset=args.output/'dataset';job=args.output/'cpu_preflight'
    dataset.mkdir(parents=True);job.mkdir()
    for name in ('comfy_source.tar.gz.bin','local_conda.json',
                 'requirements-local-conda.txt','constraints-local-conda.txt','frozen_kaggle_setup.py'):
        shutil.copyfile(args.frozen_bundle/name,dataset/name)
    frozen=json.loads((dataset/'local_conda.json').read_text(encoding='utf-8'))
    for name,digest in frozen['lock_files_sha256'].items():
        if hashlib.sha256((dataset/name).read_bytes()).hexdigest()!=digest:
            raise RuntimeError('Frozen file checksum mismatch: '+name)
    if hashlib.sha256((dataset/'comfy_source.tar.gz.bin').read_bytes()).hexdigest()!=frozen['source_archive_sha256']:
        raise RuntimeError('Frozen source checksum mismatch')
    slug=args.owner+'/h3-frozen-runtime-20261021'
    (dataset/'dataset-metadata.json').write_text(json.dumps(dict(
        id=slug,title='H3 Frozen User Runtime 20261021',licenses=[dict(name='other')],
        description='Private unchanged frozen user ComfyUI source and exact dependency snapshot. No models or credentials.'),indent=2),encoding='utf-8')
    # Script is uploaded as the notebook source, not installed in the runtime.
    code=(ROOT/'benchmarks/run_frozen_linux_preflight.py').read_text(encoding='utf-8')
    invoke="""
from pathlib import Path
matches=list(Path('/kaggle/input').rglob('local_conda.json'))
if len(matches)!=1: raise RuntimeError('Attach exactly one frozen bundle')
sys.argv=['preflight','--bundle',str(matches[0].parent),'--output','/kaggle/working/preflight_results']
raise SystemExit(main())
"""
    code=code.replace("if __name__=='__main__':\n    raise SystemExit(main())",invoke)
    notebook=dict(nbformat=4,nbformat_minor=5,metadata=dict(kernelspec=dict(
        display_name='Python 3',language='python',name='python3')),
        cells=[dict(id='frozen-linux-audit',cell_type='code',execution_count=None,metadata={},outputs=[],source=code.splitlines(keepends=True))])
    if args.candidate:
        program=(ROOT/'benchmarks/prepare_frozen_linux_candidate.py').read_text(encoding='utf-8')
        code="""import sys,subprocess
from pathlib import Path
matches=list(Path('/kaggle/input').rglob('local_conda.json'))
if len(matches)!=1:raise RuntimeError('Attach exactly one unchanged frozen source bundle')
program=Path('/kaggle/temp/prepare_frozen_linux_candidate.py')
program.parent.mkdir(parents=True,exist_ok=True)
"""+ 'program.write_text('+repr(program)+",encoding='utf-8')\n"+"""
subprocess.run([sys.executable,str(program),'--bundle',str(matches[0].parent),
    '--output','/kaggle/working/linux_candidate'],check=True,timeout=2100)
"""
        notebook['cells'][0].update(id='frozen-linux-candidate',source=code.splitlines(keepends=True))
    (job/'preflight.ipynb').write_text(json.dumps(notebook),encoding='utf-8')
    (job/'kernel-metadata.json').write_text(json.dumps(dict(
        id=args.owner+('/h3-frozen-linux-cpu-candidate' if args.candidate else '/h3-frozen-linux-cpu-preflight'),
        title='H3 Frozen Linux CPU Candidate' if args.candidate else 'H3 Frozen Linux CPU Preflight',
        code_file='preflight.ipynb',language='python',kernel_type='notebook',
        is_private=True,enable_gpu=False,enable_tpu=False,enable_internet=True,
        dataset_sources=[slug],competition_sources=[],kernel_sources=[],model_sources=[]),indent=2),encoding='utf-8')
    binding=dict(scope='CPU-only; never submit this job with a GPU accelerator',
                 frozen_manifest_sha256=hashlib.sha256((dataset/'local_conda.json').read_bytes()).hexdigest(),
                 files_sha256={str(p.relative_to(args.output)):hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in args.output.rglob('*') if p.is_file()})
    (args.output/'submission_binding.json').write_text(json.dumps(binding,indent=2),encoding='utf-8')
    print(args.output)


if __name__=='__main__':main()
