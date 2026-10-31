"""Production opt-in Sol All-INT8 attention for SM75.

The native kernel is derived from Zironic/Star7's MIT-licensed Sol SM75 path.
It is compiled from the vendored, source-locked files in
patches/sol_attention_native. No fallback to another attention backend is
performed when Sol is explicitly selected.

DiT may protect a dense prefix. Protected blocks are exact in both directions:
every query attends protected keys, and protected queries attend every key.
The native ABI provides separate D64/D128 specializations; no D64 padding transform is used.
"""
from __future__ import annotations

import hashlib
import json
from functools import lru_cache
import math
import os
from pathlib import Path
import shutil
import subprocess

import torch

ROOT = Path(__file__).resolve().parent
NATIVE = ROOT / "sol_attention_native"
TAU = 1.0
SOL_BLOCK = 64


def _source_identity():
    rows = {}
    for path in sorted(NATIVE.rglob("*")):
        if path.is_file():
            rows[path.relative_to(NATIVE).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    return rows


def _compiler():
    compiler = os.environ.get("CUDACXX") or shutil.which("nvcc")
    if not compiler:
        raise RuntimeError("Sol requires the preserved CUDA 12.8 nvcc")
    version = subprocess.check_output([compiler, "--version"], text=True, timeout=15)
    if "release 12.8" not in version:
        raise RuntimeError("Sol requires the preserved CUDA 12.8 compiler")
    return Path(compiler).resolve(), version


def _cache_key(compiler_version):
    payload = json.dumps(
        dict(sources=_source_identity(), compiler=compiler_version, target="sm_75",
             abi=12),
        sort_keys=True,
    ).encode()
    return hashlib.sha256(payload).hexdigest()[:24]


def _compile(library, compiler):
    toolkit = compiler.parent.parent
    command = [
        str(compiler), "-shared", "-O3", "--use_fast_math", "-std=c++17",
        "-arch=sm_75", "--cudart", "static", "-Xcompiler=-fPIC,-O2",
        "-Xlinker=--exclude-libs,ALL", "-I", str(NATIVE),
        "-I", str(toolkit / "include"),
    ]
    for name in ("lib64", "lib"):
        directory = toolkit / name
        if directory.is_dir():
            command += ["-L", str(directory)]
    temporary = library.with_suffix(".tmp.so")
    command += ["-o", str(temporary), str(NATIVE / "sla_sm75_sparse.cu")]
    result = subprocess.run(
        command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=360,
    )
    log = result.stdout.decode("utf-8", errors="replace")
    library.with_suffix(".compile.log").write_text(log, encoding="utf-8")
    if result.returncode:
        temporary.unlink(missing_ok=True)
        raise RuntimeError("Sol SM75 native build failed; inspect " + str(library.with_suffix(".compile.log")))
    temporary.replace(library)


@lru_cache(maxsize=1)
def native_library():
    """Return the source-locked SM75 shared object, compiling once per source identity."""
    if os.name != "posix":
        raise RuntimeError("Production Sol native compilation is qualified only on Linux SM75")
    compiler, version = _compiler()
    cache = Path(os.environ.get("H3_SOL_CACHE", "~/.cache/minimax_h3/sol_sm75")).expanduser()
    directory = cache / _cache_key(version)
    directory.mkdir(parents=True, exist_ok=True)
    library = directory / "h3_sol_all_int8_sm75.so"
    lock = directory / ".build.lock"
    with lock.open("a+b") as stream:
        try:
            import fcntl
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        except ImportError as exc:
            raise RuntimeError("Production Sol build locking requires POSIX fcntl") from exc
        if not library.is_file():
            _compile(library, compiler)
    os.environ["H3_SOL_ALL_INT8_LIBRARY"] = str(library)
    return library


def prepare_native():
    """Compile/load the exact selected Sol library before the first attention call."""
    path = native_library()
    from patches import sol_sm75_backend as native
    native._load_library()
    return str(path)


def _sm75(q):
    if not q.is_cuda or torch.cuda.get_device_capability(q.device) != (7, 5):
        raise RuntimeError("Sol is an explicit SM75/T4 backend; no fallback was attempted")


_ROUTE_STATS = []


def route_stats_snapshot():
    """Materialize deferred Sol routing statistics after timed inference is complete."""
    grouped={}
    for record in _ROUTE_STATS:
        minimum,maximum,total=record['stats'].detach().cpu().tolist()
        denominator=int(record['denominator'])
        key=(record['kind'],record['head_dim'],record['tau'])
        bucket=grouped.setdefault(key,dict(
            kind=record['kind'],head_dim=record['head_dim'],tau=record['tau'],
            calls=0,minimum=minimum,maximum=maximum,total_exact=0,total_pairs=0))
        bucket['calls']+=1
        bucket['minimum']=min(bucket['minimum'],minimum)
        bucket['maximum']=max(bucket['maximum'],maximum)
        bucket['total_exact']+=int(total)
        bucket['total_pairs']+=denominator
    result=[]
    for bucket in grouped.values():
        bucket['density']=bucket['total_exact']/bucket['total_pairs'] if bucket['total_pairs'] else None
        result.append(bucket)
    return sorted(result,key=lambda row:(row['kind'],row['head_dim'],row['tau']))


def _defer_route_stats(routing,tau,head_dim):
    stats=routing.get('route_stats_gpu')
    if stats is None:return
    topology=routing.get('topology') or {}
    _ROUTE_STATS.append(dict(
        stats=stats,
        denominator=int(routing['route_stats_denominator']),
        kind=str(topology.get('kind','generic')),
        head_dim=int(head_dim),
        tau=float(tau),
    ))


class _PreparedSol(dict):
    """Prepared Sol package owning one stream-local routing workspace lease."""

    def __init__(self, *args, workspace_lease=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._workspace_lease = workspace_lease
        self._pooled_workspace = workspace_lease is not None

    def release(self):
        lease = getattr(self, "_workspace_lease", None)
        if lease is not None:
            self._workspace_lease = None
            try:
                from patches import sol_sm75_backend as native
                native.release_workspace(lease)
            except Exception:
                # Object destruction must never turn interpreter shutdown into
                # a user-visible failure.
                pass

    def __del__(self):
        self.release()


def prepare_attention(
    q, k, v, *, protected_tokens=0, protected_start=0, tau=TAU,
    allow_d64=False, topology=None, sync_free_lut=False, collect_route_stats=True,
    q_block_base=0, q_block_count=None,
):
    """Prepare every Sol input that depends on floating Q/K/V."""
    if q.shape != k.shape or q.shape != v.shape or q.ndim != 4:
        raise ValueError("Sol requires equal BHND self-attention Q/K/V")
    if q.dtype != torch.float16 or k.dtype != q.dtype or v.dtype != q.dtype:
        raise TypeError("Sol production path requires FP16 Q/K/V")
    if not all(t.device == q.device and t.stride(-1) == 1 for t in (q, k, v)):
        raise ValueError("Sol requires same-device Q/K/V with contiguous head dimensions")
    protected_tokens=int(protected_tokens);protected_start=int(protected_start)
    if not 0 <= protected_tokens <= q.shape[-2] or not 0 <= protected_start <= q.shape[-2]-protected_tokens:
        raise ValueError("Invalid protected token span")
    if not math.isfinite(float(tau)) or float(tau) < 0:
        raise ValueError("Sol tau must be finite and nonnegative")
    _sm75(q)
    dim=q.shape[-1]
    if dim not in (64,128):
        raise ValueError("Sol supports native head_dim 64 or 128")
    if dim==64 and not allow_d64:
        raise ValueError("Sol D64 requires explicit allow_d64=True")

    from patches import sol_sm75_backend as native
    native_library()
    routing=native.prepare(
        q,k,v,tau=float(tau),sink_tokens=protected_tokens,
        sink_start=protected_start if protected_tokens else None,
        quantize_qk=True,topology=topology,
        sync_free_lut=bool(sync_free_lut),
        collect_route_stats=bool(collect_route_stats),
        pooled=os.environ.get("H3_SOL_WORKSPACE_POOL", "0") == "1",
        bitmap_lut=os.environ.get("H3_SOL_LUT_BITMAP", "0") == "1",
    )
    _defer_route_stats(routing,tau,dim)
    workspace=routing.pop("_workspace_lease",None)
    k_centroid_fp16=routing.pop("k_centroid")
    v_centroid_fp16=routing.pop("v_centroid")
    try:
        k_centroid,k_centroid_scale=native.quantize(
            k_centroid_fp16,SOL_BLOCK,workspace=workspace,
            workspace_prefix="k_centroid")
        values=native.prepare_all_int8_values(
            v,v_centroid_fp16,int(routing["centroid_padded"]),
            workspace=workspace)
        # Drop the floating centroid owners before returning to tp_dit.
        del k_centroid_fp16,v_centroid_fp16
        return _PreparedSol({
            "_h3_backend":"sol",
            "q_int8":routing["q_int8"],"k_int8":routing["k_int8"],
            "q_scale":routing["q_scale"],"k_scale":routing["k_scale"],
            "row_count":routing["row_count"],"lut":routing["lut"],
            "k_centroid":k_centroid,"k_centroid_scale":k_centroid_scale,
            "exact_mask":None if routing.get("bitmap_lut",False) else routing["exact_mask"],
            "centroid_count":int(routing["centroid_count"]),
            "centroid_padded":int(routing["centroid_padded"]),
            "validated_routing":bool(routing.get("validated_routing",False)),
            "bitmap_lut":bool(routing.get("bitmap_lut",False)),
            **values,
        },workspace_lease=workspace)
    except BaseException:
        native.release_workspace(workspace)
        raise



def prepare_attention_partition(
    q, k, v, *, q_token_offset, protected_tokens=0, protected_start=0,
    tau=TAU, topology=None, sync_free_lut=True, collect_route_stats=False,
):
    """Prepare exact Sol-v2 local-Q / full-KV attention."""
    if q.ndim!=4 or k.ndim!=4 or v.ndim!=4 or k.shape!=v.shape:
        raise ValueError("Sol partition requires Q [B,H,Q,D], K/V [B,H,KV,D]")
    if q.shape[:2]!=k.shape[:2] or q.shape[-1]!=k.shape[-1]:
        raise ValueError("Sol partition Q/K/V batch, heads and head_dim must match")
    if q.dtype!=torch.float16 or k.dtype!=q.dtype or v.dtype!=q.dtype:
        raise TypeError("Sol partition requires FP16 Q/K/V")
    if not all(t.device==q.device and t.stride(-1)==1 for t in (q,k,v)):
        raise ValueError("Sol partition requires same-device contiguous head dimensions")
    _sm75(q)
    dim=q.shape[-1]
    if dim not in (64,128):
        raise ValueError("Sol partition supports D64/D128")
    kv_length=k.shape[-2]
    protected_tokens=int(protected_tokens);protected_start=int(protected_start)
    if not 0<=protected_tokens<=kv_length or not 0<=protected_start<=kv_length-protected_tokens:
        raise ValueError("Invalid protected token span")
    if not math.isfinite(float(tau)) or float(tau)<0:
        raise ValueError("Sol tau must be finite and nonnegative")
    from patches import sol_sm75_backend as native
    native_library()
    routing=native.prepare_partition(
        q,k,v,q_token_offset=int(q_token_offset),tau=float(tau),
        sink_tokens=protected_tokens,
        sink_start=protected_start if protected_tokens else None,
        topology=topology,sync_free_lut=bool(sync_free_lut),
        collect_route_stats=bool(collect_route_stats),
        pooled=os.environ.get("H3_SOL_WORKSPACE_POOL", "0") == "1",
        bitmap_lut=os.environ.get("H3_SOL_LUT_BITMAP", "0") == "1")
    _defer_route_stats(routing,tau,dim)
    workspace=routing.pop("_workspace_lease",None)
    k_centroid_fp16=routing.pop("k_centroid")
    v_centroid_fp16=routing.pop("v_centroid")
    try:
        k_centroid,k_centroid_scale=native.quantize(
            k_centroid_fp16,SOL_BLOCK,workspace=workspace,
            workspace_prefix="k_centroid")
        values=native.prepare_all_int8_values(
            v,v_centroid_fp16,int(routing["centroid_padded"]),
            workspace=workspace)
        del k_centroid_fp16,v_centroid_fp16
        return _PreparedSol({
            "_h3_backend":"sol","_h3_partition":True,
            "q_int8":routing["q_int8"],"k_int8":routing["k_int8"],
            "q_scale":routing["q_scale"],"k_scale":routing["k_scale"],
            "row_count":routing["row_count"],"lut":routing["lut"],
            "k_centroid":k_centroid,"k_centroid_scale":k_centroid_scale,
            "exact_mask":None if routing.get("bitmap_lut",False) else routing["exact_mask"],
            "centroid_count":int(routing["centroid_count"]),
            "centroid_padded":int(routing["centroid_padded"]),
            "validated_routing":bool(routing.get("validated_routing",False)),
            "q_block_base":int(routing["q_block_base"]),
            "bitmap_lut":bool(routing.get("bitmap_lut",False)),
            **values,
        },workspace_lease=workspace)
    except BaseException:
        native.release_workspace(workspace)
        raise


def attention_partition(
    q,k,v,*,q_token_offset,protected_tokens=0,protected_start=0,tau=TAU,
    topology=None,sync_free_lut=True,collect_route_stats=False,
):
    prepared=prepare_attention_partition(
        q,k,v,q_token_offset=q_token_offset,protected_tokens=protected_tokens,
        protected_start=protected_start,tau=tau,topology=topology,
        sync_free_lut=sync_free_lut,collect_route_stats=collect_route_stats)
    return execute_prepared(
        prepared,q_block_base=prepared["q_block_base"],
        q_block_count=prepared["q_int8"].shape[-2]//SOL_BLOCK +
            (1 if prepared["q_int8"].shape[-2]%SOL_BLOCK else 0))


def execute_prepared(prepared, *, q_block_base=None, q_block_count=None):
    if not isinstance(prepared,dict) or prepared.get("_h3_backend")!="sol":
        raise TypeError("Expected prepared Sol attention")
    from patches import sol_sm75_backend as native
    if q_block_base is None:
        q_block_base=int(prepared.get("q_block_base",0))
    if isinstance(prepared, _PreparedSol) and prepared._pooled_workspace:
        lease = prepared._workspace_lease
        if lease is None:
            raise RuntimeError("Sol prepared workspace has already been released")
        native.validate_workspace_execution(lease, prepared["q_int8"].device)
    return native.run_prepared_all_int8(
        prepared["q_int8"],prepared["k_int8"],
        prepared["q_scale"],prepared["k_scale"],
        prepared["row_count"],prepared["lut"],
        prepared["k_centroid"],prepared["k_centroid_scale"],
        prepared["exact_mask"],prepared["v_int8"],prepared["v_scale"],
        prepared["v_centroid_int8"],prepared["v_centroid_scale"],
        centroid_count=prepared["centroid_count"],
        centroid_padded=prepared["centroid_padded"],
        padded_length=prepared["padded_length"],
        validated_routing=prepared["validated_routing"],
        q_block_base=q_block_base,q_block_count=q_block_count,
        bitmap_lut=bool(prepared.get("bitmap_lut",False)),
    )


def _native(q, k, v, *, protected_tokens, protected_start, tau, topology=None,
            sync_free_lut=False, collect_route_stats=True,
            q_block_base=0, q_block_count=None):
    return execute_prepared(prepare_attention(
        q,k,v,protected_tokens=protected_tokens,protected_start=protected_start,
        tau=tau,topology=topology,sync_free_lut=sync_free_lut,
        collect_route_stats=collect_route_stats,
        allow_d64=q.shape[-1]==64,
    ),q_block_base=q_block_base,q_block_count=q_block_count)


def attention(
    q, k, v, *, protected_tokens=0, protected_start=0, tau=TAU,
    allow_d64=False, topology=None, sync_free_lut=False, collect_route_stats=True,
    q_block_base=0, q_block_count=None,
):
    """Complete Sol All-INT8 attention with native D64/D128 specializations.

    protected_start/protected_tokens describe one exact span rounded outward to
    Q64/K64 blocks. Optional topology adds exact token-space local/temporal
    neighborhoods without changing centroid threshold routing elsewhere.
    """
    if q.shape != k.shape or q.shape != v.shape or q.ndim != 4:
        raise ValueError("Sol requires equal BHND self-attention Q/K/V")
    if q.dtype != torch.float16 or k.dtype != q.dtype or v.dtype != q.dtype:
        raise TypeError("Sol production path requires FP16 Q/K/V")
    if not all(t.device == q.device and t.stride(-1) == 1 for t in (q, k, v)):
        raise ValueError("Sol requires same-device Q/K/V with contiguous head dimensions")
    protected_tokens=int(protected_tokens);protected_start=int(protected_start)
    if not 0 <= protected_tokens <= q.shape[-2] or not 0 <= protected_start <= q.shape[-2]-protected_tokens:
        raise ValueError("Invalid protected token span")
    if not math.isfinite(float(tau)) or float(tau) < 0:
        raise ValueError("Sol tau must be finite and nonnegative")
    _sm75(q)
    dim = q.shape[-1]
    if dim not in (64, 128):
        raise ValueError("Sol supports native head_dim 64 or 128")
    if dim == 64 and not allow_d64:
        raise ValueError("Sol D64 requires explicit allow_d64=True")
    return _native(
        q, k, v, protected_tokens=protected_tokens,
        protected_start=protected_start, tau=tau, topology=topology,
        sync_free_lut=sync_free_lut, collect_route_stats=collect_route_stats,
        q_block_base=q_block_base,q_block_count=q_block_count,
    )
