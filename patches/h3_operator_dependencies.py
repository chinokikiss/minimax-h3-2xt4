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


# Only explicitly scoped benchmark jobs may exercise an unqualified CUDA 13.0 toolchain.
# Production and all other H3 jobs retain the validated CUDA 12.8 gate.
PTD_2NFE_CUDA13_TRIAL = 'H3_PTD_2NFE_ALLOW_UNVALIDATED_CUDA13'
TURBO_15S_CUDA13_TRIAL = 'H3_TURBO_15S_ALLOW_UNVALIDATED_CUDA13'
PTD_E2E_CUDA13_TRIAL = 'H3_PTD_E2E_ALLOW_UNVALIDATED_CUDA13'
ALL_OPS_CUDA13_T4_TRIAL = 'H3_ALL_OPS_CUDA130_T4_QUALIFY'


def sm75_native_compiler_qualification(version_text, *, torch_cuda=None):
    """Admit a source-compiled SM75 library without relaxing its ABI silently.

    The new CUDA 13 branch is isolated to the physical-T4 all-operator trial.
    Existing qualified CUDA 12.8.93 binaries retain their original admission.
    """
    import os
    if 'V12.8.93' in version_text and 'release 12.8' in version_text:
        return 'validated_cuda_12_8_93'
    if ('release 13.0' in version_text and torch_cuda == '13.0'
            and os.environ.get(ALL_OPS_CUDA13_T4_TRIAL) == '1'):
        return 'experimental_cuda_13_0_all_ops_t4_only'
    raise RuntimeError('SM75 native library requires CUDA 12.8.93; CUDA 13.0 '
                       'requires an explicit all-operators physical T4 trial '
                       'and matching Torch cu130')


def compiler_qualification(version_text, *, torch_cuda=None):
    """Return the exact admitted compiler branch, or fail closed."""
    import os
    if 'release 12.8' in version_text:
        return 'validated_cuda_12_8'
    if 'release 13.0' in version_text and torch_cuda == '13.0':
        if os.environ.get(ALL_OPS_CUDA13_T4_TRIAL) == '1':
            return 'experimental_cuda_13_0_all_ops_t4_only'
        if os.environ.get(PTD_2NFE_CUDA13_TRIAL) == '1':
            return 'experimental_cuda_13_0_ptd_2nfe_only'
        if os.environ.get(TURBO_15S_CUDA13_TRIAL) == '1':
            return 'experimental_cuda_13_0_turbo_15s_only'
        if os.environ.get(PTD_E2E_CUDA13_TRIAL) == '1':
            return 'experimental_cuda_13_0_ptd_full_e2e_only'
    raise RuntimeError(
        'H3 TP2 requires validated CUDA 12.8 nvcc. '
        'CUDA 13.0 is unqualified and allowed only by an explicit PTD 2NFE, '
        'PTD full E2E, Turbo 15s, or all-operators T4 qualification opt-in '
        'with matching native Torch cu130; '
        f'nvcc={version_text.strip().splitlines()[-1] if version_text.strip() else "empty"}, '
        f'torch_cuda={torch_cuda}')


def configure_compiler(policy):
    """Prefer qualified CUDA 12.8 even when Kaggle's default nvcc is 13.0."""
    import os
    from pathlib import Path
    import shutil
    import subprocess
    if not _uses_tilelang(policy):
        return None

    # A PyTorch cu130 runtime does not imply that its PATH nvcc is the
    # qualified 12.8 compiler. Probe the explicit pin and known side-by-side
    # installation before admitting the benchmark-only CUDA 13.0 fallback.
    candidates = [os.environ.get('CUDACXX'),
                  '/usr/local/cuda-12.8/bin/nvcc',
                  '/opt/cuda-12.8/bin/nvcc',
                  shutil.which('nvcc')]
    checked = []
    seen = set()
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate).resolve()
        if path in seen or not path.is_file():
            continue
        seen.add(path)
        try:
            info = subprocess.check_output(
                [str(path), '--version'], text=True, timeout=10)
        except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
            checked.append(f'{path}: {type(exc).__name__}')
            continue
        checked.append(f'{path}: {info.strip().splitlines()[-1]}')
        if 'release 12.8' in info:
            qualification = compiler_qualification(info)
            break
        if 'release 13.0' in info:
            # Keep as a last resort; 12.8 may be installed at the next path.
            fallback = (path, info)
    else:
        if 'fallback' not in locals():
            raise RuntimeError('No usable validated CUDA 12.8 nvcc found; '
                               'compiler candidates: ' + '; '.join(checked))
        path, info = fallback
        import torch
        qualification = compiler_qualification(info, torch_cuda=torch.version.cuda)

    os.environ.update(CUDACXX=str(path), CUDA_HOME=str(path.parent.parent),
                      CUDA_PATH=str(path.parent.parent))
    return dict(path=str(path), version=info.strip(),
                qualification=qualification, torch_cuda=__import__('torch').version.cuda,
                candidates=checked)
