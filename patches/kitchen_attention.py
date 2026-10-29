"""H3 Kitchen 0.2.37 adapter with direct FP32 output for FP32 inference.

The pinned producer, masks, tiling and INT8 arithmetic are retained. FP16/BF16
calls delegate to the installed API. Source/package files are never patched.
"""
import contextlib
import contextvars
import ctypes as C
from functools import lru_cache
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import threading
import time

import torch

ROOT = Path(__file__).with_name('kitchen_attention_native')
LIBRARY_ENV = 'H3_KITCHEN_FP32_LIBRARY'
_scope = contextvars.ContextVar('h3_corrected_kitchen_attention', default=False)
_proxy_lock = threading.RLock()

DIRECT_STORE = '''  if constexpr (std::is_same<DTypeOut, float>::value) {
#pragma unroll
    for (uint32_t fq = 0; fq < num_tiles_q; ++fq) {
#pragma unroll
      for (uint32_t fv = 0; fv < num_tiles_v; ++fv) {
#pragma unroll
        for (uint32_t pair = 0; pair < 4; ++pair) {
          const uint32_t row = bx * CTA_Q +
              get_warp_idx_q<num_warps_q, num_warps_k>() * WARP_Q +
              fq * 16 + lane_id / 4 + (pair % 2) * 8;
          const uint32_t col = fv * 16 + (lane_id % 4) * 2 + (pair / 2) * 8;
          if (row < qo_len && col < stride_seq_o) {
            float *target = O + batch_id * stride_bz_o + head_id * stride_h_o +
                            row * stride_seq_o + col;
            const float2 value = reinterpret_cast<float2 *>(RO[fq][fv])[pair];
            if ((stride_seq_o % 2) == 0 && col + 1 < stride_seq_o) {
              *reinterpret_cast<float2 *>(target) = value;
            } else {
              target[0] = value.x;
              if (col + 1 < stride_seq_o) target[1] = value.y;
            }
          }
        }
      }
    }
  } else {
'''


def source_hashes():
    manifest = json.loads((ROOT / 'vendor/manifest.json').read_text(encoding='utf-8'))
    for name, digest in manifest['files'].items():
        if hashlib.sha256((ROOT / 'vendor' / name).read_bytes()).hexdigest() != digest:
            raise RuntimeError('Pinned Kitchen source mismatch: ' + name)
    paths = [Path(__file__), ROOT / 'attention.cu', ROOT / 'vendor/manifest.json']
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def derive_header(path):
    source_hashes()
    text = (ROOT / 'vendor/qk_int_sv_i8_cuda.cuh').read_text(encoding='utf-8')
    replacements = (
        ('std::is_same<DTypeOut, nv_bfloat16>::value,\n'
         '                "DTypeOut must be half or nv_bfloat16");',
         'std::is_same<DTypeOut, nv_bfloat16>::value ||\n'
         '                    std::is_same<DTypeOut, float>::value,\n'
         '                "DTypeOut must be half, nv_bfloat16 or float");'),
        ('  // save the result to shared memory\n', DIRECT_STORE + '  // save the result to shared memory\n'),
        ('  if constexpr (return_lse) {\n', '  } // Original 16-bit store.\n\n  if constexpr (return_lse) {\n'))
    for before, after in replacements:
        if text.count(before) != 1: raise RuntimeError('Pinned Kitchen epilogue structure changed')
        text = text.replace(before, after, 1)
    Path(path).write_text(text, encoding='utf-8', newline='\n')
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build(output, *, arch, nvcc=None):
    if arch not in ('75', '120'): raise ValueError('Qualified SM75 or local SM120 required')
    output = Path(output).resolve(); output.mkdir(parents=True, exist_ok=True)
    compiler = str(nvcc or os.environ.get('CUDACXX') or shutil.which('nvcc') or '')
    if not compiler: raise RuntimeError('Direct FP32 Kitchen attention requires an existing CUDA compiler')
    compiler_version = subprocess.check_output([compiler, '--version'], text=True, timeout=15)
    if arch == '75' and 'V12.8.93' not in compiler_version:
        raise RuntimeError('SM75 Kitchen FP32 output requires the qualified CUDA12.8.93 compiler')
    digest = derive_header(output / 'h3_kitchen_attention.cuh')
    library = output / ('kitchen_fp32.dll' if os.name == 'nt' else 'kitchen_fp32.so')
    argv = [compiler, '-std=c++17', '-O3', '-arch=sm_' + arch, '--expt-relaxed-constexpr',
        '-diag-suppress=177,550', '-Xptxas=-v,--warn-on-spills', '--shared',
        '-I', str(output), '-I', str(ROOT / 'vendor'), str(ROOT / 'attention.cu'), '-o', str(library)]
    if arch == '120': argv += ['-DH3_LOCAL_SM120=1']
    if os.name == 'nt':
        finder = Path('C:/Program Files (x86)/Microsoft Visual Studio/Installer/vswhere.exe')
        compilers = subprocess.check_output([str(finder), '-latest', '-products', '*', '-find',
            'VC/Tools/MSVC/**/bin/Hostx64/x64/cl.exe'], text=True, timeout=15).strip().splitlines()
        if not compilers: raise RuntimeError('Existing MSVC compiler required')
        argv += ['--compiler-bindir', str(Path(compilers[0]).parent), '-Xcompiler=/Zc:preprocessor,/utf-8']
    else:
        argv += ['-Xcompiler=-fPIC']
        toolkit = Path(compiler).resolve().parent.parent
        argv += ['-I', str(toolkit / 'include')]
        for name in ('lib64', 'lib'):
            if (toolkit / name).is_dir(): argv += ['-L', str(toolkit / name)]
    started = time.monotonic()
    result = subprocess.run(argv, capture_output=True, timeout=300)
    (output / 'compile.log').write_bytes(result.stdout + result.stderr)
    receipt = dict(status='passed' if result.returncode == 0 else 'failed', target='sm_' + arch,
        compiler_version=compiler_version, argv=argv, sources=source_hashes(),
        derived_header_sha256=digest, seconds=time.monotonic() - started,
        artifact_sha256=hashlib.sha256(library.read_bytes()).hexdigest() if result.returncode == 0 else None)
    (output / 'build_receipt.json').write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8')
    if result.returncode: raise RuntimeError('Kitchen FP32 output build failed: ' + str(output / 'compile.log'))
    return library


def validate_library(path, arch):
    path = Path(path).resolve()
    receipt = json.loads(path.with_name('build_receipt.json').read_text(encoding='utf-8'))
    if receipt['status'] != 'passed' or receipt['target'] != 'sm_' + arch:
        raise RuntimeError('Kitchen FP32 library target/build mismatch')
    if receipt['sources'] != source_hashes(): raise RuntimeError('Kitchen FP32 library source mismatch')
    if hashlib.sha256(path.read_bytes()).hexdigest() != receipt['artifact_sha256']:
        raise RuntimeError('Kitchen FP32 library binary mismatch')
    if arch == '75' and 'V12.8.93' not in receipt['compiler_version']:
        raise RuntimeError('Kitchen FP32 library compiler mismatch')
    return path


def prepare(*, arch='75', nvcc=None):
    if version('comfy-kitchen') != '0.2.37': raise RuntimeError('Exact Kitchen0.2.37 required')
    configured = os.environ.get(LIBRARY_ENV)
    if configured: return validate_library(configured, arch)
    fingerprint = hashlib.sha256(json.dumps(source_hashes(), sort_keys=True).encode()).hexdigest()[:24]
    root = Path(os.environ.get('H3_KITCHEN_FP32_CACHE', str(Path(tempfile.gettempdir()) / 'h3-kitchen-fp32'))) / (arch + '-' + fingerprint)
    root.mkdir(parents=True, exist_ok=True)
    library = root / ('kitchen_fp32.dll' if os.name == 'nt' else 'kitchen_fp32.so')
    lock = root / 'build.lock'; started = time.monotonic()
    while True:
        if library.exists() and library.with_name('build_receipt.json').exists() and not lock.exists():
            return validate_library(library, arch)
        try: fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            if time.monotonic() - started > 330: raise TimeoutError('Kitchen FP32 build lock exceeded 330 seconds')
            time.sleep(.2); continue
        os.close(fd)
        try: return validate_library(build(root, arch=arch, nvcc=nvcc), arch)
        finally: lock.unlink(missing_ok=True)


class AttentionArgs(C.Structure):
    _fields_ = [(name, C.c_void_p) for name in ('q','k','v','o','qs','ks','vs','mask','tile_bias')] + [
        (name, C.c_int64) for name in ('mb','mh','mq','mk')] + [
        (name, C.c_int) for name in ('b','h','hk','n','nk','d','ck','cq','mask_mode','mask_dtype','out_dim')] + [
        ('scale', C.c_float), ('stream', C.c_void_p)]


def launch_metadata(packed, capability, multiprocessors=1):
    q, k, mask = packed.q, packed.k, packed.attn_mask
    mode = 0; strides = (0,0,0,0); code = 0; bias = 0; cq = 128
    if mask is not None:
        if mask.ndim == 3:
            mode = 3
            strides = (0 if mask.shape[0] == 1 else mask.stride(0),
                       0 if mask.shape[1] == 1 else mask.stride(1), 0, 1)
            bias = mask.data_ptr() + ((k.shape[2] + 127) // 128) * 128 * 4
            if q.shape[-1] == 128 and capability[0] >= 10: cq = 64
        else:
            mode = 2 if mask.stride(2) == 0 else 1; strides = mask.stride()
            code = {torch.float32:0,torch.float16:1,torch.bfloat16:2,torch.bool:3}[mask.dtype]
    if (mode == 0 and q.shape[-1] == 128 and packed.cta_k == 128 and capability[0] == 12
            and 4096 <= q.shape[2] <= 4608 and 4096 <= k.shape[2] <= 4608 and q.shape[1] <= 32):
        if q.shape[1] < 30:
            cq = 64
        else:
            blocks64 = q.shape[0] * q.shape[1] * ((q.shape[2] + 63)//64)
            blocks128 = q.shape[0] * q.shape[1] * ((q.shape[2] + 127)//128)
            if (blocks64 + multiprocessors - 1)//multiprocessors < 2*((blocks128 + multiprocessors - 1)//multiprocessors): cq = 64
    return mode, strides, code, bias, cq


class Backend:
    def __init__(self, library):
        self.library = C.CDLL(str(library))
        self.library.h3_attention.argtypes = [C.POINTER(AttentionArgs)]
        self.library.h3_attention.restype = C.c_int

    def execute(self, packed):
        q, k = packed.q, packed.k
        props = torch.cuda.get_device_properties(q.device)
        mode, strides, code, bias, cq = launch_metadata(packed, (props.major, props.minor), props.multi_processor_count)
        out = torch.empty((*q.shape[:-1],packed.original_head_dim), device=q.device, dtype=torch.float32)
        tensors = (q,k,packed.v,out,packed.q_scale,packed.k_scale,packed.v_scale)
        args = AttentionArgs(*[t.data_ptr() for t in tensors],
            packed.attn_mask.data_ptr() if packed.attn_mask is not None else 0, bias,
            *strides, q.shape[0],q.shape[1],k.shape[1],q.shape[2],k.shape[2],q.shape[3],
            packed.cta_k,cq,mode,code,packed.original_head_dim,packed.attention_scale,torch.cuda.current_stream(q.device).cuda_stream)
        error = self.library.h3_attention(C.byref(args))
        if error: raise RuntimeError('Kitchen direct FP32 attention CUDA error ' + str(error))
        return out


@lru_cache(maxsize=4)
def _backend(arch): return Backend(prepare(arch=arch))


def int8_attention(q, k, v, *, scale=None, attn_mask=None):
    import comfy_kitchen as ck
    if q.dtype != torch.float32 or not q.is_cuda or any(t.requires_grad for t in (q,k,v)):
        kwargs = {'attn_mask':attn_mask}
        if scale is not None: kwargs['scale'] = scale
        return ck.int8_attention(q,k,v,**kwargs)
    capability = torch.cuda.get_device_capability(q.device)
    if capability not in ((7,5),(12,0)):
        raise RuntimeError('H3 direct FP32 Kitchen output is qualified for SM75; SM120 is local validation only')
    arch = str(capability[0]) + str(capability[1])
    packed = ck.prequantize_int8_attention(q,k,v,scale=scale,attn_mask=attn_mask)
    backend = _backend(arch)
    return backend.execute(packed)


class _ComfyKitchenProxy:
    def __init__(self, original): self.original = original; self.users = 0
    def __getattr__(self, name): return getattr(self.original, name)
    def int8_attention(self, *args, **kwargs):
        function = int8_attention if _scope.get() else self.original.int8_attention
        return function(*args, **kwargs)


@contextlib.contextmanager
def native_scope():
    """Route Native H3 VAE calls without changing global Kitchen functions."""
    import comfy.quant_ops
    with _proxy_lock:
        if not isinstance(comfy.quant_ops.ck, _ComfyKitchenProxy):
            comfy.quant_ops.ck = _ComfyKitchenProxy(comfy.quant_ops.ck)
        proxy = comfy.quant_ops.ck
        proxy.users += 1
    token = _scope.set(True)
    try: yield
    finally:
        _scope.reset(token)
        with _proxy_lock:
            proxy.users -= 1
            if proxy.users == 0 and comfy.quant_ops.ck is proxy:
                comfy.quant_ops.ck = proxy.original
