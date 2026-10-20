"""Fail before launching ranks when the pinned operator ABI is unavailable."""
from importlib.metadata import PackageNotFoundError,version


def validate_dependencies(policy,*,t4):
    packages={'comfy-kitchen':'0.2.37'}
    tilelang=(policy.te_attention=='tilelang_group32' or policy.dit_attention.startswith('tilelang_')
              or policy.row_linear_backend.startswith('tilelang_'))
    if t4 and tilelang:packages['tilelang']='0.1.15'
    for package,expected in packages.items():
        try:actual=version(package)
        except PackageNotFoundError:actual='missing'
        if actual!=expected:
            raise RuntimeError(f'H3 TP2 requires {package}=={expected}; found {actual}. '
                'This is the historical operator ABI qualification gate. Preserve the frozen environment; '
                'qualify its differing versions at operator level before TP2 execution instead of changing packages.')


def configure_compiler(policy):
    """Pin the compiler inside SM75 workers before importing TileLang."""
    import os
    from pathlib import Path
    import shutil
    import subprocess
    if not (policy.te_attention=='tilelang_group32' or policy.dit_attention.startswith('tilelang_')
            or policy.row_linear_backend.startswith('tilelang_')):return
    compiler=os.environ.get('CUDACXX') or shutil.which('nvcc')
    if not compiler:raise RuntimeError('SM75 TileLang operators require CUDA 12.8 nvcc; set CUDACXX')
    path=Path(compiler).resolve()
    info=subprocess.check_output([str(path),'--version'],text=True,timeout=10)
    if 'release 12.8' not in info:raise RuntimeError('Use the validated CUDA 12.8 nvcc for H3 TP2 TileLang; set CUDACXX')
    os.environ.update(CUDACXX=str(path),CUDA_HOME=str(path.parent.parent),CUDA_PATH=str(path.parent.parent))
