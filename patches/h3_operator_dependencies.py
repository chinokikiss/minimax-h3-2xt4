"""Fail before launching ranks when the pinned operator ABI is unavailable."""
from importlib.metadata import PackageNotFoundError,version


def _uses_tilelang(policy):
    return (policy.dit_attention.startswith('tilelang_')
            or any(getattr(policy,name,False) for name in ('dit_fused_norm','dit_fused_gate','dit_fused_row_gate'))
            or getattr(policy,'dit_rms_rope','native')=='tilelang_fp32'
            or policy.row_linear_backend.startswith('tilelang_')
            or getattr(policy,'column_linear_backend','preallocated')!='preallocated')


def validate_dependencies(policy,*,t4):
    packages={'comfy-kitchen':'0.2.37'}
    tilelang=_uses_tilelang(policy)
    if t4 and tilelang:packages['tilelang']='0.1.13'
    for package,expected in packages.items():
        try:actual=version(package)
        except PackageNotFoundError:actual='missing'
        if actual!=expected:
            raise RuntimeError(f'H3 TP2 requires {package}=={expected}; found {actual}. '
                'This is the preserved core ABI, exercised by the 2026-10-22 physical T4 operator batch. '
                'Preserve the environment instead of changing packages to bypass this gate.')


def configure_compiler(policy):
    """Pin the compiler inside SM75 workers before importing TileLang."""
    import os
    from pathlib import Path
    import shutil
    import subprocess
    if not _uses_tilelang(policy):return
    compiler=os.environ.get('CUDACXX') or shutil.which('nvcc')
    if not compiler:raise RuntimeError('SM75 TileLang operators require CUDA 12.8 nvcc; set CUDACXX')
    path=Path(compiler).resolve()
    info=subprocess.check_output([str(path),'--version'],text=True,timeout=10)
    if 'release 12.8' not in info:raise RuntimeError('Use the validated CUDA 12.8 nvcc for H3 TP2 TileLang; set CUDACXX')
    os.environ.update(CUDACXX=str(path),CUDA_HOME=str(path.parent.parent),CUDA_PATH=str(path.parent.parent))
