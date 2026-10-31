"""Explicit CPU compilation; importing the adapter never compiles or installs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

from .derive import ROOT, derive_header, verify_vendor


def build(output, *, nvcc=None, arch='75', device_only=False, verify_hashes=True):
    if arch != '75':
        raise ValueError('This experiment qualifies SM75 only')
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    compiler = nvcc or shutil.which('nvcc')
    if not compiler:
        raise RuntimeError('An existing nvcc is required; no compiler is installed automatically')
    version = subprocess.check_output([str(compiler), '--version'], text=True, timeout=15)
    derived_hash = derive_header(output / 'h3_attention.cuh',verify_hashes=verify_hashes)
    library = output / ('kitchen_sm75.cubin' if device_only else
                         ('kitchen_sm75.dll' if os.name == 'nt' else 'kitchen_sm75.so'))
    argv = [str(compiler), '-std=c++17', '-arch=sm_75', '-O3', '--expt-relaxed-constexpr',
            '-diag-suppress=177,550',
            '-Xptxas=-v,--warn-on-spills', '-I', str(output), '-I', str(ROOT/'native/vendor'),'-I',str(ROOT/'native'),
            str(ROOT/'native/kitchen_sm75.cu'), '-o', str(library)]
    if device_only:
        argv += ['--cubin']
    else:
        argv += ['--shared']
    if os.name == 'nt':
        vswhere = Path('C:/Program Files (x86)/Microsoft Visual Studio/Installer/vswhere.exe')
        found = subprocess.check_output([str(vswhere), '-latest', '-products', '*', '-find',
            'VC/Tools/MSVC/**/bin/Hostx64/x64/cl.exe'], text=True, timeout=15).strip().splitlines()
        if not found:
            raise RuntimeError('Existing MSVC toolchain missing')
        argv += ['--compiler-bindir', str(Path(found[0]).parent), '-Xcompiler=/Zc:preprocessor,/utf-8']
    elif not device_only:
        argv += ['-Xcompiler=-fPIC']
    if os.name!='nt':
        toolkit=Path(compiler).resolve().parent.parent
        argv+=['-I',str(toolkit/'include')]
        for directory in ('lib64','lib'):
            if (toolkit/directory).is_dir():argv+=['-L',str(toolkit/directory)]
    process = subprocess.run(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             timeout=300, cwd=output)
    (output/'compile.raw.log').write_bytes(process.stdout)
    diagnostic=process.stdout.decode('utf8',errors='replace')
    (output/'compile.log').write_text(diagnostic,encoding='utf8')
    resources=[]
    for section in diagnostic.split('ptxas info    : Compiling entry function ')[1:]:
        registers=re.search(r'Used (\d+) registers',section)
        spill=re.search(r'(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads',section)
        if registers and spill:
            row=dict(kernel=section.splitlines()[0],registers=int(registers[1]),
                stack_bytes=int(spill[1]),spill_store_bytes=int(spill[2]),spill_load_bytes=int(spill[3]),
                spill_free=all(int(value)==0 for value in spill.groups()))
            shape=re.search(r'h3_attentionILi(\d+)ELi(\d+)ELi(\d+)ELb([01])ELb([01])ELb([01])E(6__half|13__nv_bfloat16)(?:Lb([01])E([mjy]))?',section.splitlines()[0])
            if shape:
                values=shape.groups()
                row.update(head_dim=int(values[0]),query_tile=int(values[1]),key_tile=int(values[2]),
                    routed=bool(int(values[3])),causal=bool(int(values[4])),cache_q=bool(int(values[5])),
                    output_dtype='float16' if values[6]=='6__half' else 'bfloat16',
                    pv_n8=values[7]=='1',offset_bits=32 if values[8]=='j' else 64)
            resources.append(row)
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted(ROOT.rglob('*')) if p.is_file() and p.suffix in ('.py','.cu','.cuh','.json')}
    receipt = dict(scope='CPU compilation only; no CUDA module load or GPU execution',
        compiler=str(compiler), compiler_version=version, platform=sys.platform,
        target='sm_75', argv=argv, exit_code=process.returncode,
        derived_header_sha256=derived_hash, vendor=(verify_vendor() if verify_hashes else
            json.loads((ROOT/'native/vendor/manifest.json').read_text())), sources=hashes,
        artifact=str(library), artifact_sha256=hashlib.sha256(library.read_bytes()).hexdigest()
            if process.returncode == 0 else None,
        artifact_bytes=library.stat().st_size if process.returncode == 0 else None,
        resources=resources,
        note='CPU compilation is not physical T4 execution/performance evidence; platform/compiler identity is recorded above')
    (output/'build_receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print(diagnostic[-2500:])
    if process.returncode:
        raise RuntimeError(f'nvcc failed; see {output / "compile.log"}')
    print(library)
    return library


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--nvcc')
    parser.add_argument('--device-only', action='store_true')
    args = parser.parse_args()
    build(args.output, nvcc=args.nvcc, device_only=args.device_only)
