"""Quantization-preserving primitives shared by the repaired two-rank blocks."""
import torch
import torch.nn.functional as F
import comfy_kitchen as ck


def gather(x, comm, dim):
    if not comm.is_initialized():
        raise RuntimeError("Two-rank execution requires an initialized collective backend")
    length = torch.tensor([x.shape[dim]], device=x.device, dtype=torch.int64)
    lengths = [torch.empty_like(length) for _ in range(comm.get_world_size())]
    comm.all_gather(lengths, length)
    sizes = [int(n.item()) for n in lengths]
    maximum = max(sizes)
    shape = list(x.shape)
    shape[dim] = maximum
    padded = torch.zeros(shape, device=x.device, dtype=x.dtype)
    padded.narrow(dim, 0, x.shape[dim]).copy_(x)
    parts = [torch.empty_like(padded) for _ in sizes]
    comm.all_gather(parts, padded)
    full = torch.cat([p.narrow(dim, 0, n) for p, n in zip(parts, sizes)], dim=dim)
    rank = comm.get_rank()
    return full, sum(sizes[:rank]), sizes[rank]


def quantize(x, groupsize=256, input_act=None, input_act_weight=None, input_act_eps=0.0):
    # Match the native fused ConvRot quantizer, including its FP32 activation
    # calculation. Never silently substitute non-rotated quantization.
    backend = ck.backends.cuda
    flat = x.reshape(-1, x.shape[-1]).contiguous()
    width = flat.shape[-1] // (2 if input_act == 'swiglu' else 1)
    fused = (groupsize == 256 and width % 256 == 0 and 256 <= width <= backend._CONVROT_FUSED_MAX_K
             and backend._convrot_fused_shared_memory_fits(flat, width, groupsize))
    if fused:
        return backend.quantize_int8_rowwise_convrot64(flat, groupsize,
            input_act=input_act, input_act_weight=input_act_weight, input_act_eps=input_act_eps)
    # The actual TE FFN is 25600 wide. Match native's rotated matmul route
    # instead of exceeding the fused quantizer's shared-memory limit.
    flat = backend._apply_input_act(flat, input_act, input_act_weight, input_act_eps)
    if backend._should_use_convrot_fused_kernel(flat, width, groupsize):
        return backend.quantize_int8_rowwise_convrot(flat, groupsize)
    rotation = backend._build_hadamard(groupsize, device=flat.device, dtype=flat.dtype)
    return backend.quantize_and_rotate_rowwise(flat, rotation, groupsize)


def int_mm(q, weight):
    rows = q.shape[0]
    padded_rows = max(32, ((rows + 31) // 32) * 32)
    q = F.pad(q, (0, 0, 0, padded_rows - rows)).contiguous()
    return torch._int_mm(q, weight.t().contiguous())[:rows]


def dequantize(accumulator, activation_scale, weight_scale, dtype, bias=None):
    if accumulator.is_cuda:
        # Preserve native bias/dequantization rounding (including CUDA FMA).
        # Separate eager multiply/add kernels can change INT8 thresholds in
        # subsequent blocks, even if a small initialized fixture agrees.
        backend = ck.backends.cuda
        out = torch.empty(accumulator.shape, device=accumulator.device, dtype=dtype)
        bias_arg = backend._gemm_vector_arg(bias, accumulator.device, dtype) if bias is not None else backend._empty_cuda_tensor(accumulator.device, dtype)
        wrap = backend._wrap_for_dlpack
        backend._C.dequantize_int8_linear(wrap(accumulator.contiguous()), wrap(activation_scale.contiguous()),
            wrap(weight_scale.float().reshape(-1).contiguous()), wrap(bias_arg), wrap(out),
            backend.DTYPE_TO_CODE[dtype], torch.cuda.current_stream(accumulator.device).cuda_stream)
        return out
    out = accumulator.float() * activation_scale.float() * weight_scale.float().reshape(1, -1)
    if bias is not None:
        out += bias.float()
    return out.to(dtype)


def row_linear(x, weight, weight_scale, comm, world_size, groupsize=256, input_act=None,
               bias=None, residual=None, residual_scale=None):
    # Gather the feature vector BEFORE quantization. Native W8A8 uses one
    # maximum per complete row; local maxima introduce a different operator.
    full, start, length = gather(x, comm, -1) if world_size > 1 else (x, 0, x.shape[-1])
    if input_act == "swiglu":
        # Each rank holds [gate_rank, up_rank]. Restore [all_gate, all_up].
        pieces = full.chunk(world_size, -1)
        full = torch.cat([p.chunk(2, -1)[0] for p in pieces] + [p.chunk(2, -1)[1] for p in pieces], -1)
        start //= 2
        length //= 2
    q, scale = quantize(full, groupsize, input_act=input_act)
    partial = int_mm(q[:, start:start + length].contiguous(), weight)
    if world_size > 1:
        comm.all_reduce(partial, op=comm.ReduceOp.SUM)
    # Match the native cuBLAS/Turing path: cast dequantized output before residual.
    out = dequantize(partial, scale, weight_scale, x.dtype, bias).reshape(*x.shape[:-1], weight.shape[0])
    if residual is not None:
        out = torch.addcmul(residual, out.to(x.dtype), residual_scale)
    return out.to(x.dtype)


def packed_rows(weight, scale, rank, world_size, components):
    if weight.shape[0] % (components * world_size):
        raise ValueError("Packed components must divide evenly across ranks")
    chunks = weight.chunk(components, dim=0)
    scale_chunks = scale.chunk(components, dim=0)
    return (torch.cat([w.chunk(world_size, dim=0)[rank] for w in chunks], dim=0),
            torch.cat([s.chunk(world_size, dim=0)[rank] for s in scale_chunks], dim=0))
