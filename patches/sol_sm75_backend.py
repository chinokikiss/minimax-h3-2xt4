"""SM75 loader for the Star7 Q64/K64 threshold-routed Sol kernels."""

from __future__ import annotations

import ctypes
import os
from pathlib import Path

import torch


SOL_BLOCK_Q = 64
SOL_BLOCK_K = 64
HEAD_DIMS = (64, 128)


_LIBRARY = None

# Stream-local reusable routing/centroid workspace.  Only routing and centroid
# intermediates are retained between calls; the large Q/K/V INT8 payloads stay
# call-owned so the pool cannot pin ~100 MB across the surrounding DiT layer.
_WORKSPACE_POOL = {}


class _WorkspaceLease:
    """Single owner for a stream-local entry; releases on error or GC."""

    def __init__(self, entry):
        self.entry = entry

    def release(self):
        entry, self.entry = getattr(self, "entry", None), None
        if entry is not None:
            entry["busy"] = False

    def __del__(self):
        self.release()


def _acquire_workspace(device):
    device = torch.device(device)
    stream = int(torch.cuda.current_stream(device).cuda_stream)
    key = (device.type, device.index, stream)
    entries = _WORKSPACE_POOL.setdefault(key, [])
    for entry in entries:
        if not entry["busy"]:
            entry["busy"] = True
            entry["reuses"] += 1
            return _WorkspaceLease(entry)
    entry = dict(key=key, busy=True, reuses=0, buffers={})
    entries.append(entry)
    return _WorkspaceLease(entry)


def _workspace_tensor(entry, name, shape, dtype, device):
    if entry is None:
        return torch.empty(shape, dtype=dtype, device=device)
    shape = tuple(int(x) for x in shape)
    entry = entry.entry
    if entry is None:
        raise RuntimeError("Sol workspace lease already released")
    tensor = entry["buffers"].get(name)
    if (
        tensor is None or tuple(tensor.shape) != shape or
        tensor.dtype != dtype or tensor.device != torch.device(device)
    ):
        tensor = torch.empty(shape, dtype=dtype, device=device)
        entry["buffers"][name] = tensor
    return tensor


def release_workspace(entry):
    if entry is not None:
        entry.release()


def validate_workspace_execution(lease, device):
    """The prepared scratch must execute on the stream that owns its lease.

    Same-stream enqueue ordering makes asynchronous CUDA reuse safe.  Without an
    explicit event/wait handshake, executing a prepared package on a different
    stream would invalidate that guarantee, even if its Python owner survives.
    """
    if lease is None:
        return
    entry = getattr(lease, "entry", None)
    if entry is None or not entry["busy"]:
        raise RuntimeError("Sol prepared workspace lease was already released")
    device = torch.device(device)
    stream = int(torch.cuda.current_stream(device).cuda_stream)
    if entry["key"] != (device.type, device.index, stream):
        raise RuntimeError(
            "Sol prepared workspace must execute on its preparation CUDA stream"
        )


def workspace_pool_snapshot():
    rows = []
    for entries in _WORKSPACE_POOL.values():
        for entry in entries:
            rows.append(dict(
                key=entry["key"], busy=bool(entry["busy"]),
                reuses=int(entry["reuses"]),
                bytes=sum(t.numel() * t.element_size()
                          for t in entry["buffers"].values()),
                buffers=sorted(entry["buffers"]),
            ))
    return rows


def clear_workspace_pool():
    # Never invalidate a prepared package which still owns a lease.
    for key in list(_WORKSPACE_POOL):
        free = [entry for entry in _WORKSPACE_POOL[key] if entry["busy"]]
        if free:
            _WORKSPACE_POOL[key] = free
        else:
            del _WORKSPACE_POOL[key]


def _load_library():
    """Load the explicitly built experimental SM75 library.

    The experiment is isolated from Star7's packaged binaries and never changes
    production dispatch. Set H3_SOL_ALL_INT8_LIBRARY to the built shared library.
    """
    global _LIBRARY
    if _LIBRARY is not None:
        return _LIBRARY
    configured = os.environ.get("H3_SOL_ALL_INT8_LIBRARY", "").strip()
    if not configured:
        raise RuntimeError("H3_SOL_ALL_INT8_LIBRARY is not configured")
    path = Path(configured).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Sol All-INT8 SM75 library is missing: {path}")
    library = ctypes.CDLL(str(path))
    try:
        abi = library.star7_sla_sm75_abi_version
        quant_v = library.star7_sla_sm75_quant_v_int8
        shared = library.star7_sol_sm75_shared_bytes
        prepare_routes = library.star7_sol_sm75_prepare_routes
        pack_lut = library.star7_sol_sm75_pack_lut
        quantize = library.star7_sol_sm75_quantize
        quantize_v_with_scale = library.star7_sol_sm75_quantize_v_with_scale
        launch = library.star7_sol_sm75_launch
        launch_all_int8 = library.star7_sol_sm75_launch_all_int8
        launch_all_int8_complete = library.star7_sol_sm75_launch_all_int8_complete
    except AttributeError as exc:
        raise RuntimeError("Experimental SM75 binary is missing required Sol symbols") from exc
    abi.argtypes = []
    abi.restype = ctypes.c_int
    if int(abi()) != 12:
        raise RuntimeError(f"Experimental SM75 ABI mismatch: expected 12, got {int(abi())}")
    quant_v.argtypes = [
        *([ctypes.c_uint64] * 3), *([ctypes.c_int] * 5),
        *([ctypes.c_int64] * 3), ctypes.c_int, ctypes.c_uint64,
    ]
    quant_v.restype = ctypes.c_int
    shared.argtypes = [ctypes.c_int]
    shared.restype = ctypes.c_int
    prepare_routes.argtypes = [
        *([ctypes.c_uint64] * 11), *([ctypes.c_int] * 5),
        ctypes.c_float, ctypes.c_float, *([ctypes.c_int] * 9),
        ctypes.c_uint64,
    ]
    prepare_routes.restype = ctypes.c_int
    pack_lut.argtypes = [
        *([ctypes.c_uint64] * 3), *([ctypes.c_int] * 4), ctypes.c_uint64,
    ]
    pack_lut.restype = ctypes.c_int
    quantize.argtypes = [
        *([ctypes.c_uint64] * 3), *([ctypes.c_int] * 5), ctypes.c_uint64,
    ]
    quantize.restype = ctypes.c_int
    quantize_v_with_scale.argtypes = [
        *([ctypes.c_uint64] * 3), *([ctypes.c_int] * 5), ctypes.c_uint64,
    ]
    quantize_v_with_scale.restype = ctypes.c_int
    launch.argtypes = [
        *([ctypes.c_uint64] * 12), *([ctypes.c_int] * 7),
        ctypes.c_float, ctypes.c_uint64,
    ]
    launch.restype = ctypes.c_int
    launch_all_int8.argtypes = [
        *([ctypes.c_uint64] * 9), *([ctypes.c_int] * 6),
        ctypes.c_float, ctypes.c_uint64,
    ]
    launch_all_int8.restype = ctypes.c_int
    launch_all_int8_complete.argtypes = [
        *([ctypes.c_uint64] * 14), *([ctypes.c_int] * 13),
        ctypes.c_float, ctypes.c_uint64,
    ]
    launch_all_int8_complete.restype = ctypes.c_int
    fused = getattr(library, "star7_sol_sm75_prepare_quantized_routes", None)
    if fused is not None:
        fused.argtypes = [
            *([ctypes.c_uint64] * 15), *([ctypes.c_int64] * 9),
            *([ctypes.c_int] * 5), ctypes.c_float, ctypes.c_float,
            *([ctypes.c_int] * 9), ctypes.c_uint64,
        ]
        fused.restype = ctypes.c_int
    partition = getattr(library, "star7_sol_sm75_prepare_quantized_partition_routes", None)
    if partition is not None:
        partition.argtypes = [
            *([ctypes.c_uint64] * 15), *([ctypes.c_int64] * 9),
            *([ctypes.c_int] * 8), ctypes.c_float, ctypes.c_float,
            *([ctypes.c_int] * 9), ctypes.c_uint64,
        ]
        partition.restype = ctypes.c_int
    statistics = getattr(library, "star7_sol_sm75_route_statistics", None)
    if statistics is not None:
        statistics.argtypes = [
            ctypes.c_uint64, ctypes.c_uint64, ctypes.c_int, ctypes.c_uint64,
        ]
        statistics.restype = ctypes.c_int
    ordered = getattr(library, "star7_sol_sm75_pack_ordered_lut", None)
    if ordered is not None:
        ordered.argtypes = [
            ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64,
            *([ctypes.c_int] * 3), ctypes.c_uint64,
        ]
        ordered.restype = ctypes.c_int
    bitmap_pack = getattr(library, "star7_sol_sm75_pack_bitmap_lut", None)
    bitmap_launch = getattr(library, "star7_sol_sm75_launch_all_int8_complete_bitmap", None)
    if bitmap_pack is not None:
        bitmap_pack.argtypes = [
            *([ctypes.c_uint64] * 3), *([ctypes.c_int] * 3), ctypes.c_uint64,
        ]
        bitmap_pack.restype = ctypes.c_int
    if bitmap_launch is not None:
        bitmap_launch.argtypes = launch_all_int8_complete.argtypes
        bitmap_launch.restype = ctypes.c_int
    active_blocks = getattr(library, "star7_sol_sm75_active_blocks", None)
    if active_blocks is not None:
        active_blocks.argtypes = [ctypes.c_int, ctypes.c_int]
        active_blocks.restype = ctypes.c_int
    _LIBRARY = library
    return library


def prepare(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    tau: float,
    sink_tokens: int = 0,
    sink_start: int | None = None,
    quantize_qk: bool = False,
    topology: dict | None = None,
    sync_free_lut: bool = False,
    collect_route_stats: bool = True,
    pooled: bool = False,
    bitmap_lut: bool = False,
):
    """Run SM75 Sol centroids, topology-aware routing, and LUT packing in CUDA."""
    if q.shape != k.shape or q.shape != v.shape or q.ndim != 4:
        raise ValueError("SM75 Sol preprocessing requires equal [B,H,L,D] Q/K/V")
    if q.dtype != torch.float16 or any(x.dtype != q.dtype for x in (k, v)):
        raise TypeError("SM75 Sol preprocessing requires FP16 Q/K/V")
    library = _load_library()
    if bitmap_lut and not sync_free_lut:
        raise ValueError("Sol bitmap LUT requires sync_free_lut=True")
    if bitmap_lut and (
        getattr(library, "star7_sol_sm75_pack_bitmap_lut", None) is None or
        getattr(library, "star7_sol_sm75_launch_all_int8_complete_bitmap", None) is None
    ):
        raise RuntimeError("Installed SM75 Sol library has no bitmap-LUT extension")
    fused = getattr(library, "star7_sol_sm75_prepare_quantized_routes", None) if quantize_qk else None
    if not all(x.is_cuda and x.device == q.device and x.stride(-1) == 1 for x in (q, k, v)):
        raise ValueError("SM75 Sol preprocessing requires same-device CUDA tensors with contiguous head dimensions")
    if fused is None and not all(x.is_contiguous() for x in (q, k, v)):
        raise ValueError("The installed SM75 Sol preprocessing requires contiguous tensors")
    batch, heads, length, head_dim = q.shape
    if head_dim not in HEAD_DIMS:
        raise ValueError(f"SM75 Sol preprocessing requires head_dim in {HEAD_DIMS}")
    blocks = (length + SOL_BLOCK_K - 1) // SOL_BLOCK_K
    padded_blocks = ((blocks + SOL_BLOCK_K - 1) // SOL_BLOCK_K) * SOL_BLOCK_K
    workspace = _acquire_workspace(q.device) if pooled else None
    summaries = (batch, heads, padded_blocks, head_dim)
    q_centroid = _workspace_tensor(
        workspace, "q_centroid", summaries, q.dtype, q.device)
    k_centroid = _workspace_tensor(
        workspace, "k_centroid", summaries, q.dtype, q.device)
    v_centroid = _workspace_tensor(
        workspace, "v_centroid", summaries, q.dtype, q.device)
    k_mean = _workspace_tensor(
        workspace, "k_mean", (batch, heads, head_dim), torch.float32, q.device)
    k_variance = _workspace_tensor(
        workspace, "k_variance", (batch, heads, head_dim), torch.float32, q.device)
    threshold = _workspace_tensor(
        workspace, "threshold", (batch, heads, blocks), torch.float32, q.device)
    # Bitmap mode needs this byte mask only until bitmap packing completes.
    # Do not retain its storage in the pool past preparation.
    exact_mask = _workspace_tensor(
        None if bitmap_lut else workspace, "exact_mask",
        (batch, heads, blocks, blocks), torch.uint8, q.device)
    row_count = _workspace_tensor(
        workspace, "row_count", (batch, heads, blocks), torch.int32, q.device)
    if sink_tokens > 0:
        start = length - int(sink_tokens) if sink_start is None else int(sink_start)
        sink_first = max(0, start // SOL_BLOCK_K)
        sink_last = min(blocks, (start + int(sink_tokens) + SOL_BLOCK_K - 1) // SOL_BLOCK_K)
    else:
        sink_first = blocks
        sink_last = blocks

    topology = dict(topology or {})
    topology_start = int(topology.get("start_token", -1))
    topology_end = int(topology.get("end_token", -1))
    frame_stride = int(topology.get("frame_stride_tokens", 0))
    temporal_radius = int(topology.get("temporal_radius", 0))
    local_radius = int(topology.get("local_radius_tokens", 0))
    protect_boundaries = int(bool(topology.get("protect_frame_boundaries", False)))
    if topology:
        if not 0 <= topology_start < topology_end <= length:
            raise ValueError("Invalid Sol topology token span")
        if frame_stride <= 0 or temporal_radius < 0 or local_radius < 0:
            raise ValueError("Invalid Sol topology stride/radius")
    else:
        topology_start = topology_end = -1
        frame_stride = temporal_radius = local_radius = protect_boundaries = 0

    stream = torch.cuda.current_stream(q.device).cuda_stream
    quantized = {}
    if fused is not None:
        quantized = {
            "q_int8": torch.empty(q.shape, dtype=torch.int8, device=q.device),
            "k_int8": torch.empty(k.shape, dtype=torch.int8, device=k.device),
            "q_scale": torch.empty((batch, heads, blocks * 4), dtype=torch.float32, device=q.device),
            "k_scale": torch.empty((batch, heads, blocks), dtype=torch.float32, device=q.device),
        }
    extra = [value.data_ptr() for value in quantized.values()]
    if fused is not None:
        extra.extend(stride for value in (q, k, v) for stride in value.stride()[:3])
    prepare_routes = fused or library.star7_sol_sm75_prepare_routes
    code = int(prepare_routes(
        q.data_ptr(), k.data_ptr(), v.data_ptr(),
        q_centroid.data_ptr(), k_centroid.data_ptr(), v_centroid.data_ptr(),
        k_mean.data_ptr(), k_variance.data_ptr(), threshold.data_ptr(),
        exact_mask.data_ptr(), row_count.data_ptr(),
        *extra,
        batch, heads, length, padded_blocks, head_dim,
        float(tau), head_dim ** -0.5, sink_first, sink_last,
        topology_start, topology_end, frame_stride, temporal_radius,
        local_radius, protect_boundaries, int(not sync_free_lut), stream,
    ))
    if code:
        raise RuntimeError(f"SM75 native Sol preprocessing failed with code={code}")

    rows = batch * heads * blocks
    ordered = getattr(library, "star7_sol_sm75_pack_ordered_lut", None)
    minimum = maximum = total = density = None
    if sync_free_lut:
        maximum = (blocks + 31) // 32 if bitmap_lut else blocks
        lut = _workspace_tensor(
            workspace, "bitmap_lut" if bitmap_lut else "lut",
            (batch, heads, blocks, maximum), torch.int32, q.device)
    else:
        statistics = getattr(library, "star7_sol_sm75_route_statistics", None)
        if statistics is not None:
            stats = torch.empty(3, dtype=torch.int64, device=q.device)
            code = int(statistics(row_count.data_ptr(), stats.data_ptr(), row_count.numel(), stream))
            if code:
                raise RuntimeError(f"SM75 Sol route statistics failed with code={code}")
        else:
            minimum_gpu, maximum_gpu = torch.aminmax(row_count)
            stats = torch.stack((minimum_gpu, maximum_gpu, row_count.sum()))
        minimum, maximum, total = stats.cpu().tolist()
        if minimum <= 0 or maximum > blocks:
            raise RuntimeError(
                f"SM75 native Sol routing produced invalid counts {minimum}..{maximum}"
            )
        lut = _workspace_tensor(
            workspace, "lut", (batch, heads, blocks, maximum),
            torch.int32, q.device)
        density = total / (rows * blocks)

    if bitmap_lut:
        code = int(library.star7_sol_sm75_pack_bitmap_lut(
            exact_mask.data_ptr(), lut.data_ptr(), row_count.data_ptr(),
            rows, blocks, maximum, stream,
        ))
    elif ordered is not None:
        code = int(ordered(
            exact_mask.data_ptr(), lut.data_ptr(), row_count.data_ptr(),
            rows, blocks, maximum, stream,
        ))
    else:
        code = int(library.star7_sol_sm75_pack_lut(
            exact_mask.data_ptr(), row_count.data_ptr(), lut.data_ptr(),
            rows, blocks, maximum, 0, stream,
        ))
    if code:
        raise RuntimeError(f"SM75 native Sol LUT packing failed with code={code}")

    # Optional diagnostics are deliberately outside the latency path.
    route_stats_gpu = None
    if sync_free_lut and collect_route_stats:
        statistics = getattr(library, "star7_sol_sm75_route_statistics", None)
        if statistics is not None:
            route_stats_gpu = torch.empty(3, dtype=torch.int64, device=q.device)
            code = int(statistics(
                row_count.data_ptr(), route_stats_gpu.data_ptr(),
                row_count.numel(), stream))
            if code:
                raise RuntimeError(f"SM75 Sol route statistics failed with code={code}")

    return {
        "row_count": row_count,
        "lut": lut,
        "density": density,
        "minimum": minimum,
        "maximum": maximum,
        "exact_mask": exact_mask,
        "k_centroid": k_centroid,
        "v_centroid": v_centroid,
        "centroid_count": blocks,
        "centroid_padded": padded_blocks,
        "validated_routing": ordered is not None,
        "compact_workspace": not sync_free_lut,
        "sync_free_lut": bool(sync_free_lut),
        "head_dim": head_dim,
        "topology": topology or None,
        "route_stats_gpu": route_stats_gpu,
        "route_stats_denominator": rows * blocks,
        "_workspace_lease": workspace,
        "bitmap_lut": bool(bitmap_lut),
        **quantized,
    }



def prepare_partition(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    q_token_offset: int,
    tau: float,
    sink_tokens: int = 0,
    sink_start: int | None = None,
    topology: dict | None = None,
    sync_free_lut: bool = True,
    collect_route_stats: bool = False,
    pooled: bool = False,
    bitmap_lut: bool = False,
):
    """Prepare compact local-Q routing against full-sequence K/V."""
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4 or k.shape != v.shape:
        raise ValueError("SM75 partition preprocessing requires Q [B,H,Q,D], K/V [B,H,KV,D]")
    if q.shape[:2] != k.shape[:2] or q.shape[-1] != k.shape[-1]:
        raise ValueError("SM75 partition Q/K/V batch, heads and head_dim must match")
    if q.dtype != torch.float16 or k.dtype != q.dtype or v.dtype != q.dtype:
        raise TypeError("SM75 partition preprocessing requires FP16 Q/K/V")
    if not all(x.is_cuda and x.device == q.device and x.stride(-1) == 1 for x in (q,k,v)):
        raise ValueError("SM75 partition Q/K/V require same-device CUDA tensors")
    batch,heads,q_length,head_dim=q.shape
    kv_length=k.shape[-2]
    if head_dim not in HEAD_DIMS:
        raise ValueError(f"SM75 partition head_dim must be in {HEAD_DIMS}")
    q_token_offset=int(q_token_offset)
    if q_token_offset<0 or q_token_offset%SOL_BLOCK_Q:
        raise ValueError("SM75 partition q_token_offset must be nonnegative and Q64-aligned")
    q_blocks=(q_length+SOL_BLOCK_Q-1)//SOL_BLOCK_Q
    k_blocks=(kv_length+SOL_BLOCK_K-1)//SOL_BLOCK_K
    q_block_base=q_token_offset//SOL_BLOCK_Q
    if q_block_base+q_blocks>k_blocks or q_token_offset+q_length>kv_length:
        raise ValueError("SM75 partition Q span exceeds full KV sequence")
    q_padded=((q_blocks+15)//16)*16
    k_padded=((k_blocks+SOL_BLOCK_K-1)//SOL_BLOCK_K)*SOL_BLOCK_K
    workspace=_acquire_workspace(q.device) if pooled else None
    qc=_workspace_tensor(
        workspace,"q_centroid",(batch,heads,q_padded,head_dim),q.dtype,q.device)
    kc=_workspace_tensor(
        workspace,"k_centroid",(batch,heads,k_padded,head_dim),q.dtype,q.device)
    vc=_workspace_tensor(
        workspace,"v_centroid",(batch,heads,k_padded,head_dim),q.dtype,q.device)
    k_mean=_workspace_tensor(
        workspace,"k_mean",(batch,heads,head_dim),torch.float32,q.device)
    k_variance=_workspace_tensor(
        workspace,"k_variance",(batch,heads,head_dim),torch.float32,q.device)
    threshold=_workspace_tensor(
        workspace,"threshold",(batch,heads,q_blocks),torch.float32,q.device)
    exact_mask=_workspace_tensor(
        None if bitmap_lut else workspace,"exact_mask",
        (batch,heads,q_blocks,k_blocks),torch.uint8,q.device)
    row_count=_workspace_tensor(
        workspace,"row_count",(batch,heads,q_blocks),torch.int32,q.device)
    # Q/K INT8 payloads remain call-owned; retaining them in the pool would pin
    # the dominant live-set across the rest of the VAE layer.
    q8=torch.empty(q.shape,dtype=torch.int8,device=q.device)
    k8=torch.empty(k.shape,dtype=torch.int8,device=k.device)
    qs=torch.empty((batch,heads,q_blocks*4),dtype=torch.float32,device=q.device)
    ks=torch.empty((batch,heads,k_blocks),dtype=torch.float32,device=q.device)

    if sink_tokens>0:
        start=kv_length-int(sink_tokens) if sink_start is None else int(sink_start)
        sink_first=max(0,start//SOL_BLOCK_K)
        sink_last=min(k_blocks,(start+int(sink_tokens)+SOL_BLOCK_K-1)//SOL_BLOCK_K)
    else:
        sink_first=sink_last=k_blocks
    topology=dict(topology or {})
    topology_start=int(topology.get("start_token",-1))
    topology_end=int(topology.get("end_token",-1))
    frame_stride=int(topology.get("frame_stride_tokens",0))
    temporal_radius=int(topology.get("temporal_radius",0))
    local_radius=int(topology.get("local_radius_tokens",0))
    protect_boundaries=int(bool(topology.get("protect_frame_boundaries",False)))
    if topology:
        if not 0<=topology_start<topology_end<=kv_length:
            raise ValueError("Invalid Sol partition topology token span")
        if frame_stride<=0 or temporal_radius<0 or local_radius<0:
            raise ValueError("Invalid Sol partition topology stride/radius")
    else:
        topology_start=topology_end=-1
        frame_stride=temporal_radius=local_radius=protect_boundaries=0

    library=_load_library()
    if bitmap_lut and not sync_free_lut:
        raise ValueError("Sol bitmap LUT requires sync_free_lut=True")
    if bitmap_lut and (
        getattr(library,"star7_sol_sm75_pack_bitmap_lut",None) is None or
        getattr(library,"star7_sol_sm75_launch_all_int8_complete_bitmap",None) is None
    ):
        raise RuntimeError("Installed SM75 Sol library has no bitmap-LUT extension")
    native=getattr(library,"star7_sol_sm75_prepare_quantized_partition_routes",None)
    if native is None:
        raise RuntimeError("Installed SM75 Sol library lacks local-Q partition preprocessing")
    stream=torch.cuda.current_stream(q.device).cuda_stream
    code=int(native(
        q.data_ptr(),k.data_ptr(),v.data_ptr(),qc.data_ptr(),kc.data_ptr(),
        vc.data_ptr(),k_mean.data_ptr(),k_variance.data_ptr(),threshold.data_ptr(),
        exact_mask.data_ptr(),row_count.data_ptr(),q8.data_ptr(),k8.data_ptr(),
        qs.data_ptr(),ks.data_ptr(),
        *(stride for value in (q,k,v) for stride in value.stride()[:3]),
        batch,heads,q_length,kv_length,q_padded,k_padded,head_dim,q_block_base,
        float(tau),head_dim**-0.5,sink_first,sink_last,topology_start,topology_end,
        frame_stride,temporal_radius,local_radius,protect_boundaries,
        int(not sync_free_lut),stream,
    ))
    if code:
        raise RuntimeError(f"SM75 native Sol partition preprocessing failed with code={code}")

    rows=batch*heads*q_blocks
    maximum=k_blocks
    if sync_free_lut:
        lut=_workspace_tensor(
            workspace,"bitmap_lut" if bitmap_lut else "lut",
            (batch,heads,q_blocks,(k_blocks+31)//32 if bitmap_lut else k_blocks),
            torch.int32,q.device)
    else:
        statistics=getattr(library,"star7_sol_sm75_route_statistics",None)
        stats=torch.empty(3,dtype=torch.int64,device=q.device)
        code=int(statistics(row_count.data_ptr(),stats.data_ptr(),row_count.numel(),stream))
        if code: raise RuntimeError(f"SM75 Sol partition route statistics failed with code={code}")
        minimum,maximum,total=stats.cpu().tolist()
        if minimum<=0 or maximum>k_blocks:
            raise RuntimeError(f"SM75 Sol partition routing produced invalid counts {minimum}..{maximum}")
        lut=_workspace_tensor(
            workspace,"lut",(batch,heads,q_blocks,maximum),torch.int32,q.device)
    ordered=getattr(library,"star7_sol_sm75_pack_ordered_lut",None)
    if bitmap_lut:
        code=int(library.star7_sol_sm75_pack_bitmap_lut(
            exact_mask.data_ptr(),lut.data_ptr(),row_count.data_ptr(),
            rows,k_blocks,(k_blocks+31)//32,stream))
    elif ordered is not None:
        code=int(ordered(exact_mask.data_ptr(),lut.data_ptr(),row_count.data_ptr(),
                         rows,k_blocks,maximum,stream))
    else:
        code=int(library.star7_sol_sm75_pack_lut(
            exact_mask.data_ptr(),row_count.data_ptr(),lut.data_ptr(),
            rows,k_blocks,maximum,0,stream))
    if code:
        raise RuntimeError(f"SM75 native Sol partition LUT packing failed with code={code}")

    route_stats_gpu=None
    if sync_free_lut and collect_route_stats:
        statistics=getattr(library,"star7_sol_sm75_route_statistics",None)
        if statistics is not None:
            route_stats_gpu=torch.empty(3,dtype=torch.int64,device=q.device)
            code=int(statistics(row_count.data_ptr(),route_stats_gpu.data_ptr(),
                                row_count.numel(),stream))
            if code: raise RuntimeError(f"SM75 Sol partition route statistics failed with code={code}")
    return dict(
        row_count=row_count,lut=lut,exact_mask=exact_mask,
        k_centroid=kc,v_centroid=vc,centroid_count=k_blocks,
        centroid_padded=k_padded,validated_routing=ordered is not None,
        compact_workspace=not sync_free_lut,sync_free_lut=bool(sync_free_lut),
        head_dim=head_dim,topology=topology or None,route_stats_gpu=route_stats_gpu,
        route_stats_denominator=rows*k_blocks,q_int8=q8,k_int8=k8,
        q_scale=qs,k_scale=ks,q_block_base=q_block_base,q_length=q_length,
        kv_length=kv_length,_workspace_lease=workspace,
        bitmap_lut=bool(bitmap_lut),
    )


def quantize(value: torch.Tensor, block: int, *, workspace=None,
             workspace_prefix: str | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    if value.dtype != torch.float16 or value.ndim != 4 or not value.is_cuda:
        raise TypeError("SM75 native Sol quantization requires CUDA FP16 [B,H,L,D]")
    if value.shape[-1] not in HEAD_DIMS or not value.is_contiguous():
        raise ValueError(f"SM75 native Sol quantization requires contiguous head_dim in {HEAD_DIMS}")
    batch, heads, length, head_dim = value.shape
    groups = (length + block - 1) // block
    if workspace is not None and workspace_prefix:
        output = _workspace_tensor(
            workspace, workspace_prefix + "_int8", value.shape,
            torch.int8, value.device)
        scale = _workspace_tensor(
            workspace, workspace_prefix + "_scale", (batch, heads, groups),
            torch.float32, value.device)
    else:
        output = torch.empty_like(value, dtype=torch.int8)
        scale = torch.empty(
            (batch, heads, groups), dtype=torch.float32, device=value.device,
        )
    library = _load_library()
    stream = torch.cuda.current_stream(value.device).cuda_stream
    code = int(library.star7_sol_sm75_quantize(
        value.data_ptr(), output.data_ptr(), scale.data_ptr(),
        batch, heads, length, int(block), head_dim, stream,
    ))
    if code:
        raise RuntimeError(f"SM75 native Sol quantization failed with code={code}")
    return output, scale


def quantize_v_with_scale(
    value: torch.Tensor,
    scale: torch.Tensor,
    padded_length: int,
) -> torch.Tensor:
    if value.dtype != torch.float16 or value.ndim != 4 or not value.is_contiguous():
        raise TypeError("SM75 Sol centroid V quantization requires contiguous FP16")
    batch, heads, length, head_dim = value.shape
    if head_dim not in HEAD_DIMS or scale.shape != (batch, heads, head_dim):
        raise ValueError("SM75 Sol centroid V scale shape is invalid")
    output = torch.empty(
        (batch, heads, head_dim, padded_length),
        dtype=torch.int8, device=value.device,
    )
    library = _load_library()
    stream = torch.cuda.current_stream(value.device).cuda_stream
    code = int(library.star7_sol_sm75_quantize_v_with_scale(
        value.data_ptr(), scale.data_ptr(), output.data_ptr(),
        batch, heads, length, padded_length, head_dim, stream,
    ))
    if code:
        raise RuntimeError(
            f"SM75 native Sol centroid-V quantization failed with code={code}"
        )
    return output



def prepare_all_int8_values(
    value: torch.Tensor,
    centroid: torch.Tensor,
    centroid_padded: int,
    *,
    workspace=None,
) -> dict[str, torch.Tensor | int]:
    """Quantize V and centroid-V before the caller releases floating QKV."""
    if value.dtype != torch.float16 or value.ndim != 4 or not value.is_cuda:
        raise TypeError("SM75 Sol prepared V requires CUDA FP16 [B,H,L,D]")
    if centroid.dtype != torch.float16 or centroid.ndim != 4 or not centroid.is_cuda:
        raise TypeError("SM75 Sol prepared centroid V requires CUDA FP16 [B,H,L,D]")
    batch, heads, length, head_dim = value.shape
    if head_dim not in HEAD_DIMS or value.stride(-1) != 1:
        raise ValueError("SM75 Sol prepared V requires contiguous head dimensions")
    if centroid.shape != (batch, heads, int(centroid_padded), head_dim):
        raise ValueError("SM75 Sol prepared centroid V shape is invalid")
    padded_length = ((length + SOL_BLOCK_K - 1) // SOL_BLOCK_K) * SOL_BLOCK_K
    library = _load_library()
    stream = torch.cuda.current_stream(value.device).cuda_stream

    v_int8 = torch.empty(
        (batch, heads, head_dim, padded_length), dtype=torch.int8, device=value.device,
    )
    v_scale = torch.empty(
        (batch, heads, head_dim), dtype=torch.float32, device=value.device,
    )
    code = int(library.star7_sla_sm75_quant_v_int8(
        value.data_ptr(), v_int8.data_ptr(), v_scale.data_ptr(),
        batch, heads, length, head_dim, padded_length,
        value.stride(0), value.stride(1), value.stride(2), 1, stream,
    ))
    if code:
        raise RuntimeError(f"SM75 Sol V quantization failed with code={code}")

    v_centroid_int8 = _workspace_tensor(
        workspace, "v_centroid_int8",
        (batch, heads, head_dim, int(centroid_padded)),
        torch.int8, value.device)
    v_centroid_scale = _workspace_tensor(
        workspace, "v_centroid_scale", (batch, heads, head_dim),
        torch.float32, value.device)
    code = int(library.star7_sla_sm75_quant_v_int8(
        centroid.data_ptr(), v_centroid_int8.data_ptr(),
        v_centroid_scale.data_ptr(), batch, heads, int(centroid_padded),
        head_dim, int(centroid_padded), centroid.stride(0),
        centroid.stride(1), centroid.stride(2), 1, stream,
    ))
    if code:
        raise RuntimeError(
            f"SM75 Sol centroid-V quantization failed with code={code}"
        )
    return dict(
        v_int8=v_int8, v_scale=v_scale,
        v_centroid_int8=v_centroid_int8,
        v_centroid_scale=v_centroid_scale,
        padded_length=padded_length,
    )


def run_prepared_all_int8(
    q: torch.Tensor,
    k: torch.Tensor,
    q_scale: torch.Tensor,
    k_scale: torch.Tensor,
    row_count: torch.Tensor,
    lut: torch.Tensor,
    k_centroid: torch.Tensor,
    k_centroid_scale: torch.Tensor,
    exact_mask: torch.Tensor | None,
    v_int8: torch.Tensor,
    v_scale: torch.Tensor,
    v_centroid_int8: torch.Tensor,
    v_centroid_scale: torch.Tensor,
    *,
    centroid_count: int,
    centroid_padded: int,
    padded_length: int,
    validated_routing: bool = False,
    q_block_base: int = 0,
    q_block_count: int | None = None,
    bitmap_lut: bool = False,
) -> torch.Tensor:
    """Launch Sol after all floating Q/K/V owners may already be released."""
    tensors=(q,k,q_scale,k_scale,row_count,lut,k_centroid,k_centroid_scale,
             v_int8,v_scale,v_centroid_int8,v_centroid_scale)
    if not bitmap_lut:
        tensors+= (exact_mask,)
    if any(not isinstance(x,torch.Tensor) or not x.is_cuda or x.device != q.device
           for x in tensors):
        raise ValueError("SM75 prepared Sol tensors must share one CUDA device")
    if q.dtype != torch.int8 or k.dtype != torch.int8:
        raise TypeError("SM75 prepared Sol requires INT8 Q/K")
    if q.ndim!=4 or k.ndim!=4 or q.shape[:2]!=k.shape[:2] or q.shape[-1]!=k.shape[-1]:
        raise ValueError("SM75 prepared Sol Q/K batch, heads and head_dim must match")
    batch,heads,q_length,head_dim=q.shape
    kv_length=k.shape[-2]
    if head_dim not in HEAD_DIMS or not q.is_contiguous() or not k.is_contiguous():
        raise ValueError("SM75 prepared Sol Q/K layout is invalid")
    q_blocks=(q_length+SOL_BLOCK_Q-1)//SOL_BLOCK_Q
    key_blocks=(kv_length+SOL_BLOCK_K-1)//SOL_BLOCK_K
    q_block_base=int(q_block_base)
    q_block_count=(q_blocks-q_block_base if q_length==kv_length and q_block_base>0
                   else q_blocks) if q_block_count is None else int(q_block_count)
    compact_query=q_length!=kv_length
    q_storage_block_base=q_block_base if compact_query else 0
    route_block_base=q_block_base if compact_query else 0
    if q_block_base<0 or q_block_count<=0 or q_block_count>q_blocks:
        raise ValueError("SM75 prepared Sol query-block range is invalid")
    if q_block_base+q_block_count>key_blocks:
        raise ValueError("SM75 prepared Sol global query span exceeds KV blocks")
    if q_scale.shape!=(batch,heads,q_blocks*4) or q_scale.dtype!=torch.float32:
        raise ValueError("SM75 prepared Sol Q-scale shape is invalid")
    if k_scale.shape!=(batch,heads,key_blocks) or k_scale.dtype!=torch.float32:
        raise ValueError("SM75 prepared Sol K-scale shape is invalid")
    if row_count.shape!=(batch,heads,q_blocks) or row_count.dtype!=torch.int32:
        raise ValueError("SM75 prepared Sol row-count shape is invalid")
    if lut.ndim!=4 or lut.shape[:3]!=row_count.shape or lut.dtype!=torch.int32:
        raise ValueError("SM75 prepared Sol LUT shape is invalid")
    lut_stride=int(lut.shape[-1])
    if bitmap_lut:
        if lut_stride != (key_blocks+31)//32:
            raise ValueError("SM75 prepared Sol bitmap stride is invalid")
        if not validated_routing:
            raise ValueError("SM75 prepared Sol bitmap requires validated routing")
    elif not 0<lut_stride<=key_blocks:
        raise ValueError("SM75 prepared Sol LUT stride is invalid")
    if not validated_routing and (
        int(row_count.min().item())<=0 or int(row_count.max().item())>lut_stride
    ):
        raise ValueError("SM75 prepared Sol row-count exceeds LUT stride")
    centroid_count=int(centroid_count);centroid_padded=int(centroid_padded)
    centroid_groups=centroid_padded//SOL_BLOCK_K
    if k_centroid.shape!=(batch,heads,centroid_padded,head_dim) or k_centroid.dtype!=torch.int8:
        raise ValueError("SM75 prepared Sol K-centroid shape is invalid")
    if k_centroid_scale.shape!=(batch,heads,centroid_groups) or k_centroid_scale.dtype!=torch.float32:
        raise ValueError("SM75 prepared Sol K-centroid scale shape is invalid")
    if bitmap_lut:
        if exact_mask is not None:
            raise ValueError("SM75 bitmap LUT must not retain redundant byte mask")
    elif exact_mask.shape!=(batch,heads,q_blocks,centroid_count) or exact_mask.dtype!=torch.uint8:
        raise ValueError("SM75 prepared Sol exact-mask shape is invalid")
    if v_int8.shape!=(batch,heads,head_dim,int(padded_length)) or v_int8.dtype!=torch.int8:
        raise ValueError("SM75 prepared Sol V-int8 shape is invalid")
    if v_scale.shape!=(batch,heads,head_dim) or v_scale.dtype!=torch.float32:
        raise ValueError("SM75 prepared Sol V-scale shape is invalid")
    if v_centroid_int8.shape!=(batch,heads,head_dim,centroid_padded) or v_centroid_int8.dtype!=torch.int8:
        raise ValueError("SM75 prepared Sol centroid-V shape is invalid")
    if v_centroid_scale.shape!=(batch,heads,head_dim) or v_centroid_scale.dtype!=torch.float32:
        raise ValueError("SM75 prepared Sol centroid-V scale shape is invalid")

    output=torch.empty((batch,heads,q_length,head_dim),dtype=torch.float16,device=q.device)
    library=_load_library()
    stream=torch.cuda.current_stream(q.device).cuda_stream
    launch = (getattr(library,"star7_sol_sm75_launch_all_int8_complete_bitmap",None)
              if bitmap_lut else library.star7_sol_sm75_launch_all_int8_complete)
    if launch is None:
        raise RuntimeError("Installed SM75 Sol library has no bitmap-LUT launcher")
    code=int(launch(
        q.data_ptr(),k.data_ptr(),v_int8.data_ptr(),
        q_scale.data_ptr(),k_scale.data_ptr(),v_scale.data_ptr(),
        row_count.data_ptr(),lut.data_ptr(),k_centroid.data_ptr(),
        v_centroid_int8.data_ptr(),k_centroid_scale.data_ptr(),
        v_centroid_scale.data_ptr(),
        0 if bitmap_lut else exact_mask.data_ptr(),output.data_ptr(),
        batch,heads,kv_length,q_length,int(padded_length),lut_stride,
        centroid_count,centroid_padded,head_dim,q_block_base,q_block_count,
        q_storage_block_base,route_block_base,head_dim**-0.5,stream,
    ))
    if code:
        raise RuntimeError(f"SM75 CUDA prepared Sol All-INT8 launch failed with cudaError={code}")
    return output


def availability() -> tuple[bool, str]:
    if not torch.cuda.is_available():
        return False, "NVIDIA CUDA is unavailable"
    capability = torch.cuda.get_device_capability()
    if capability != (7, 5):
        return False, f"Star7 Sol SM75 requires compute capability 7.5, got {capability}"
    try:
        library = _load_library()
        shared = max(int(library.star7_sol_sm75_shared_bytes(dim)) for dim in HEAD_DIMS)
    except Exception as exc:
        return False, str(exc)
    properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    maximum = int(getattr(properties, "shared_memory_per_block_optin", 0) or 0)
    if maximum and shared > maximum:
        return False, f"Sol kernel needs {shared} shared bytes, GPU allows {maximum}"
    return True, f"native CUDA Q64/K64, 4 warps, shared={shared} bytes"


def supports_fused_preprocess() -> bool:
    return getattr(_load_library(), "star7_sol_sm75_prepare_quantized_routes", None) is not None


def run(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor | list[torch.Tensor],
    q_scale: torch.Tensor,
    k_scale: torch.Tensor,
    row_count: torch.Tensor,
    lut: torch.Tensor,
    *,
    all_int8: bool = False,
    approximation=None,
    validated_routing: bool = False,
) -> torch.Tensor:
    consume_v = isinstance(v, list)
    if consume_v:
        if len(v) != 1:
            raise ValueError("SM75 Sol consuming V input requires one tensor")
        v = v.pop()
    tensors = (q, k, v, q_scale, k_scale, row_count, lut)
    if any(not tensor.is_cuda for tensor in tensors):
        raise ValueError("SM75 Sol requires CUDA tensors")
    if any(tensor.device != q.device for tensor in tensors[1:]):
        raise ValueError("SM75 Sol tensors must share one device")
    if q.dtype != torch.int8 or k.dtype != torch.int8 or v.dtype != torch.float16:
        raise TypeError("SM75 Sol requires INT8 Q/K and FP16 V")
    if q_scale.dtype != torch.float32 or k_scale.dtype != torch.float32:
        raise TypeError("SM75 Sol Q/K scales must be FP32")
    if row_count.dtype != torch.int32 or lut.dtype != torch.int32:
        raise TypeError("SM75 Sol row counts and LUT must be INT32")
    if any(not tensor.is_contiguous() for tensor in tensors if tensor is not v):
        raise ValueError("SM75 Sol requires contiguous tensors")
    if not v.is_contiguous() and (not all_int8 or v.stride(-1) != 1):
        raise ValueError("SM75 Sol FP16-PV requires contiguous V; All-INT8 requires contiguous head dimensions")

    batch, heads, length, head_dim = q.shape
    q_blocks = (length + SOL_BLOCK_Q - 1) // SOL_BLOCK_Q
    key_blocks = (length + SOL_BLOCK_K - 1) // SOL_BLOCK_K
    if head_dim not in HEAD_DIMS or k.shape != q.shape or v.shape != q.shape:
        raise ValueError(f"SM75 Sol requires equal [B,H,L,D] Q/K/V with D in {HEAD_DIMS}")
    if row_count.shape != (batch, heads, q_blocks):
        raise ValueError("SM75 Sol row_count shape does not match Q")
    if lut.ndim != 4 or lut.shape[:3] != row_count.shape:
        raise ValueError("SM75 Sol LUT shape does not match row_count")
    lut_stride = int(lut.shape[-1])
    if lut_stride <= 0 or lut_stride > key_blocks:
        raise ValueError("SM75 Sol LUT stride is invalid")
    if not validated_routing and (int(row_count.min().item()) <= 0 or int(row_count.max().item()) > lut_stride):
        raise ValueError("SM75 Sol row_count exceeds LUT stride")
    if q_scale.shape != (batch, heads, q_blocks * 4):
        raise ValueError("SM75 Sol Q scale must contain four 16-row scales per Q64 block")
    if k_scale.shape != (batch, heads, key_blocks):
        raise ValueError("SM75 Sol K scale shape does not match K64 blocks")

    library = _load_library()
    stream = torch.cuda.current_stream(q.device).cuda_stream
    if all_int8:
        if approximation is None or len(approximation) != 6:
            raise ValueError(
                "SM75 Sol All-INT8 requires exact-plus-centroid approximation inputs"
            )
        (
            k_centroid, v_centroid, k_centroid_scale, exact_mask,
            centroid_count, centroid_padded,
        ) = approximation
        centroid_groups = int(centroid_padded) // SOL_BLOCK_K
        if (
            k_centroid.dtype != torch.int8
            or v_centroid.dtype != torch.float16
            or k_centroid_scale.dtype != torch.float32
            or exact_mask.dtype != torch.uint8
        ):
            raise TypeError("SM75 Sol All-INT8 centroid tensor dtypes are invalid")
        if k_centroid.shape != (batch, heads, centroid_padded, head_dim):
            raise ValueError("SM75 Sol All-INT8 K-centroid shape is invalid")
        if v_centroid.shape != (batch, heads, centroid_padded, head_dim):
            raise ValueError("SM75 Sol All-INT8 V-centroid shape is invalid")
        if k_centroid_scale.shape != (batch, heads, centroid_groups):
            raise ValueError("SM75 Sol All-INT8 K-centroid scale shape is invalid")
        if exact_mask.shape != (batch, heads, q_blocks, centroid_count):
            raise ValueError("SM75 Sol All-INT8 exact-mask shape is invalid")
        padded_length = key_blocks * SOL_BLOCK_K
        v_int8 = torch.empty(
            (batch, heads, head_dim, padded_length),
            dtype=torch.int8,
            device=q.device,
        )
        v_scale = torch.empty(
            (batch, heads, head_dim), dtype=torch.float32, device=q.device,
        )
        quant_code = int(library.star7_sla_sm75_quant_v_int8(
            v.data_ptr(), v_int8.data_ptr(), v_scale.data_ptr(),
            batch, heads, length, head_dim, padded_length,
            v.stride(0), v.stride(1), v.stride(2), 1, stream,
        ))
        if quant_code:
            raise RuntimeError(f"SM75 Sol V quantization failed with code={quant_code}")
        v_centroid_int8 = torch.empty(
            (batch, heads, head_dim, centroid_padded),
            dtype=torch.int8, device=q.device,
        )
        v_centroid_scale = torch.empty(
            (batch, heads, head_dim), dtype=torch.float32, device=q.device,
        )
        centroid_quant_code = int(library.star7_sla_sm75_quant_v_int8(
            v_centroid.data_ptr(), v_centroid_int8.data_ptr(),
            v_centroid_scale.data_ptr(), batch, heads, centroid_padded,
            head_dim, centroid_padded, v_centroid.stride(0),
            v_centroid.stride(1), v_centroid.stride(2), 1, stream,
        ))
        if centroid_quant_code:
            raise RuntimeError(
                "SM75 Sol centroid-V quantization failed with "
                f"code={centroid_quant_code}"
            )
        if consume_v:
            del tensors, v
            output = torch.empty(
                (batch, heads, length, head_dim),
                dtype=torch.float16,
                device=q.device,
            )
        else:
            # Native Sol writes output as contiguous BHLD regardless of V strides.
            output = torch.empty((batch, heads, length, head_dim),
                dtype=torch.float16, device=q.device)
        code = int(library.star7_sol_sm75_launch_all_int8_complete(
            q.data_ptr(), k.data_ptr(), v_int8.data_ptr(),
            q_scale.data_ptr(), k_scale.data_ptr(), v_scale.data_ptr(),
            row_count.data_ptr(), lut.data_ptr(), k_centroid.data_ptr(),
            v_centroid_int8.data_ptr(), k_centroid_scale.data_ptr(),
            v_centroid_scale.data_ptr(), exact_mask.data_ptr(),
            output.data_ptr(), batch, heads, length, length,
            padded_length, lut_stride, centroid_count, centroid_padded, head_dim,
            0, q_blocks, 0, 0, head_dim ** -0.5, stream,
        ))
    else:
        if approximation is None or len(approximation) != 6:
            raise ValueError("SM75 Sol FP16-PV requires centroid approximation inputs")
        (
            k_centroid, v_centroid, k_centroid_scale, exact_mask,
            centroid_count, centroid_padded,
        ) = approximation
        approximation_tensors = (
            k_centroid, v_centroid, k_centroid_scale, exact_mask,
        )
        if any(not tensor.is_cuda for tensor in approximation_tensors):
            raise ValueError("SM75 Sol centroids must be CUDA tensors")
        if any(tensor.device != q.device for tensor in approximation_tensors):
            raise ValueError("SM75 Sol centroids must share the Q device")
        if (
            k_centroid.dtype != torch.int8
            or v_centroid.dtype != torch.float16
            or k_centroid_scale.dtype != torch.float32
            or exact_mask.dtype != torch.uint8
        ):
            raise TypeError("SM75 Sol centroid tensor dtypes are invalid")
        if any(not tensor.is_contiguous() for tensor in approximation_tensors):
            raise ValueError("SM75 Sol centroid tensors must be contiguous")
        centroid_groups = int(centroid_padded) // SOL_BLOCK_K
        if k_centroid.shape != (batch, heads, centroid_padded, head_dim):
            raise ValueError("SM75 Sol K-centroid shape is invalid")
        if v_centroid.shape != k_centroid.shape:
            raise ValueError("SM75 Sol V-centroid shape is invalid")
        if k_centroid_scale.shape != (batch, heads, centroid_groups):
            raise ValueError("SM75 Sol K-centroid scale shape is invalid")
        if exact_mask.shape != (batch, heads, q_blocks, centroid_count):
            raise ValueError("SM75 Sol exact-mask shape is invalid")
        # The FP16-PV launcher uses the same contiguous BHLD output ABI.
        output = torch.empty((batch, heads, length, head_dim),
            dtype=torch.float16, device=q.device)
        code = int(library.star7_sol_sm75_launch(
            q.data_ptr(), k.data_ptr(), v.data_ptr(),
            q_scale.data_ptr(), k_scale.data_ptr(), row_count.data_ptr(),
            lut.data_ptr(), k_centroid.data_ptr(), v_centroid.data_ptr(),
            k_centroid_scale.data_ptr(), exact_mask.data_ptr(),
            output.data_ptr(), batch, heads, length, lut_stride,
            centroid_count, centroid_padded, head_dim, head_dim ** -0.5, stream,
        ))
    if code:
        mode = "All-INT8" if all_int8 else "FP16-PV"
        raise RuntimeError(
            f"SM75 CUDA Sol {mode} launch failed with cudaError={code}; "
            "no fallback was attempted"
        )
    return output
