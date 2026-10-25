"""Freeze fresh Kaggle jobs for the authorized native-Torch/core-pinned profile."""
import argparse
import ast
import base64
import gzip
import hashlib
import io
import json
from pathlib import Path
import tarfile

ROOT=Path(__file__).resolve().parents[1]
BENCHMARKS=['prepare_kaggle_native_torch.py','check_linux_pread.py','run_h3_physical_fp16_flow.py',
 'run_h3_native_fp32_flow.py','run_h3_tensor_flow.py','fixture_inputs.py','logical_dual_gpu.py',
 'test_h3_scaled_fp16_row.py','tp2_tradeoffs.py','tp2_timing.py','test_h3_operator_integration.py',
 'test_h3_operator_enhancements.py','qualify_h3_layer_bake.py','qualify_h3_staged_vae.py',
 'check_native_audio_decode.py','check_legacy_video_query.py','checkpoint_ranges.py']


def capsule():
    files=[p for p in (ROOT/'patches').glob('*.py')]
    files += [ROOT/'benchmarks'/name for name in BENCHMARKS]
    files += [ROOT/'custom_nodes/minimax_h3_tp2/__init__.py']
    hashes={};buffer=io.BytesIO()
    with tarfile.open(fileobj=buffer,mode='w') as archive:
        for path in sorted(files):
            data=path.read_bytes();name=path.relative_to(ROOT).as_posix()
            if path.suffix=='.py':ast.parse(data,filename=name)
            hashes[name]=hashlib.sha256(data).hexdigest()
            item=tarfile.TarInfo(name);item.size=len(data);archive.addfile(item,io.BytesIO(data))
    return gzip.compress(buffer.getvalue(),mtime=0),hashes


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--owner',default='pamcks');args=parser.parse_args()
    if args.output.exists():raise FileExistsError('Never overwrite submitted sources')
    args.output.mkdir(parents=True)
    data,hashes=capsule();digest=hashlib.sha256(data).hexdigest()
    code='''import base64,hashlib,io,json,pathlib,subprocess,sys,tarfile
OUT=pathlib.Path('/kaggle/working/native_profile');OUT.mkdir(exist_ok=True)
SRC=pathlib.Path('/kaggle/temp/h3_native_source');SRC.mkdir(parents=True,exist_ok=True)
'''
    code+='payload=base64.b64decode('+repr(base64.b64encode(data).decode())+')\n'
    code+=f"assert hashlib.sha256(payload).hexdigest()=={digest!r}\n"
    code+="(OUT/'source_capsule.tar.gz.bin').write_bytes(payload)\n"
    code+="with tarfile.open(fileobj=io.BytesIO(payload),mode='r:gz') as archive:archive.extractall(SRC,filter='data')\n"
    code+=f"(OUT/'source_hashes.json').write_text(json.dumps({hashes!r},indent=2))\n"
    code+='''bundles=list(pathlib.Path('/kaggle/input').rglob('local_conda.json'))
if len(bundles)!=1:raise RuntimeError('Attach exactly one unchanged frozen source bundle')
subprocess.run([sys.executable,str(SRC/'benchmarks/prepare_kaggle_native_torch.py'),
 '--bundle',str(bundles[0].parent),'--source',str(SRC),'--output',str(OUT)],check=True,timeout=1200)
'''
    ast.parse(code)
    notebook=dict(nbformat=4,nbformat_minor=5,metadata=dict(kernelspec=dict(
        name='python3',display_name='Python 3',language='python')),cells=[dict(id='native-torch-cpu',
        cell_type='code',execution_count=None,metadata={},outputs=[],source=code.splitlines(keepends=True))])
    (args.output/'experiment.ipynb').write_text(json.dumps(notebook),encoding='utf-8')
    (args.output/'kernel-metadata.json').write_text(json.dumps(dict(
        id=args.owner+'/h3-native-torch-cpu-preflight',title='H3 Native Torch CPU Preflight',
        code_file='experiment.ipynb',language='python',kernel_type='notebook',is_private=True,
        enable_gpu=False,enable_tpu=False,enable_internet=True,
        dataset_sources=[args.owner+'/h3-frozen-runtime-20261021'],competition_sources=[],kernel_sources=[],model_sources=[]),indent=2))
    (args.output/'source_binding.json').write_text(json.dumps(dict(capsule_sha256=digest,files_sha256=hashes),indent=2))
    print(args.output)


if __name__=='__main__':main()
