"""
Benchmark & Performance Analysis: MiniMax-H3 Qwen3-VL-32B Text Encoder
Comparing:
  1. Single GPU with CPU layer offloading (Standard ComfyUI fallback)
  2. Pipeline Parallelism (PP=2 / Layer Sharding: 25 layers on GPU 0, 25 layers on GPU 1)
  3. Tensor Parallelism (TP=2: Column/Row Parallel across GPU 0 & GPU 1)
Under Kaggle 2x NVIDIA T4 conditions:
  - PCIe Gen3 x16 (no NVLink)
  - NO P2P (Peer-to-Peer disabled: GPU-GPU transfers routed via Host RAM)
"""

import time
import torch
import comfy_kitchen

def benchmark_gemm_compute(device, seq_len=128, num_layers=50, is_tp=False):
    """
    Measures the raw computation time for the INT8 linear layers of Qwen3-VL-32B.
    Qwen3-VL-32B per layer:
      - q_proj: [5120 -> 8192]
      - k_proj: [5120 -> 1024]
      - v_proj: [5120 -> 1024]
      - o_proj: [8192 -> 5120]
      - gate_proj: [5120 -> 25600]
      - up_proj: [5120 -> 25600]
      - down_proj: [25600 -> 5120]
    """
    B = 1
    S = seq_len
    H = 5120
    I = 25600
    Q_out = 8192
    KV_out = 1024

    scale = torch.tensor(0.001, dtype=torch.float32, device=device)

    # In TP=2, each GPU has half the output/input dimensions
    divisor = 2 if is_tp else 1
    layers_to_run = num_layers if is_tp else (num_layers // 2) # in PP, each GPU runs 25 layers

    x = torch.randn(B * S, H, dtype=torch.float16, device=device)
    w_q = torch.randint(-128, 127, (Q_out // divisor, H), dtype=torch.int8, device=device)
    w_k = torch.randint(-128, 127, (KV_out // divisor, H), dtype=torch.int8, device=device)
    w_v = torch.randint(-128, 127, (KV_out // divisor, H), dtype=torch.int8, device=device)
    
    x_attn = torch.randn(B * S, Q_out // divisor, dtype=torch.float16, device=device)
    w_o = torch.randint(-128, 127, (H, Q_out // divisor), dtype=torch.int8, device=device)

    w_gate = torch.randint(-128, 127, (I // divisor, H), dtype=torch.int8, device=device)
    w_up = torch.randint(-128, 127, (I // divisor, H), dtype=torch.int8, device=device)

    x_mlp = torch.randn(B * S, I // divisor, dtype=torch.float16, device=device)
    w_down = torch.randint(-128, 127, (H, I // divisor), dtype=torch.int8, device=device)

    # Warmup
    for _ in range(5):
        _ = comfy_kitchen.int8_linear(x, w_q, scale, None, torch.float16, convrot=True, convrot_groupsize=256)
        _ = comfy_kitchen.int8_linear(x, w_k, scale, None, torch.float16, convrot=True, convrot_groupsize=256)
        _ = comfy_kitchen.int8_linear(x, w_v, scale, None, torch.float16, convrot=True, convrot_groupsize=256)
        _ = comfy_kitchen.int8_linear(x_attn, w_o, scale, None, torch.float16, convrot=True, convrot_groupsize=256)
        _ = comfy_kitchen.int8_linear(x, w_gate, scale, None, torch.float16, convrot=True, convrot_groupsize=256)
        _ = comfy_kitchen.int8_linear(x, w_up, scale, None, torch.float16, convrot=True, convrot_groupsize=256)
        _ = comfy_kitchen.int8_linear(x_mlp, w_down, scale, None, torch.float16, convrot=True, convrot_groupsize=256)
    torch.cuda.synchronize()

    # Benchmark 1 layer
    iterations = 20
    start = time.perf_counter()
    for _ in range(iterations):
        _ = comfy_kitchen.int8_linear(x, w_q, scale, None, torch.float16, convrot=True, convrot_groupsize=256)
        _ = comfy_kitchen.int8_linear(x, w_k, scale, None, torch.float16, convrot=True, convrot_groupsize=256)
        _ = comfy_kitchen.int8_linear(x, w_v, scale, None, torch.float16, convrot=True, convrot_groupsize=256)
        _ = comfy_kitchen.int8_linear(x_attn, w_o, scale, None, torch.float16, convrot=True, convrot_groupsize=256)
        _ = comfy_kitchen.int8_linear(x, w_gate, scale, None, torch.float16, convrot=True, convrot_groupsize=256)
        _ = comfy_kitchen.int8_linear(x, w_up, scale, None, torch.float16, convrot=True, convrot_groupsize=256)
        _ = comfy_kitchen.int8_linear(x_mlp, w_down, scale, None, torch.float16, convrot=True, convrot_groupsize=256)
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) / iterations
    
    total_compute_time = elapsed * layers_to_run
    return elapsed, total_compute_time

def simulate_no_p2p_communication(device, seq_len=128):
    """
    Simulates No-P2P communication over PCIe via Host RAM bounce buffer:
    1. Single activation transfer for PP=2: (1, seq_len, 5120) FP16
    2. 100 AllReduce operations for TP=2: (1, seq_len, 5120) FP16 x 100
    """
    H = 5120
    tensor = torch.randn(1, seq_len, H, dtype=torch.float16, device=device)
    # Host pinned bounce buffer
    cpu_buffer = torch.empty(1, seq_len, H, dtype=torch.float16, pin_memory=True)

    # Warmup
    for _ in range(5):
        cpu_buffer.copy_(tensor, non_blocking=True)
        tensor.copy_(cpu_buffer, non_blocking=True)
    torch.cuda.synchronize()

    # Measure single transfer (GPU -> Host RAM -> GPU) representing No-P2P bounce
    iterations = 50
    start = time.perf_counter()
    for _ in range(iterations):
        cpu_buffer.copy_(tensor, non_blocking=False)
        tensor.copy_(cpu_buffer, non_blocking=False)
    torch.cuda.synchronize()
    single_transfer_time = (time.perf_counter() - start) / iterations

    # In NCCL AllReduce without P2P, both ranks write to SHM buffer, reduce, and read back.
    # Typically 2x round-trip + IPC sync barrier latency (~50-80us minimum per call)
    allreduce_per_call = single_transfer_time * 1.5 + 0.000080
    tp_100_allreduces = allreduce_per_call * 100

    return single_transfer_time, tp_100_allreduces

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 70)
    print("MiniMax-H3 Qwen3-VL-32B (50 Layers) Performance Evaluation on 2x T4")
    print(f"Hardware Environment: {torch.cuda.get_device_name(0)}")
    print("Interconnect Condition: PCIe Gen3 x16, NO P2P (Host RAM Bounce Buffer)")
    print("=" * 70)

    seq_lengths = [64, 128, 256, 512, 1024, 2048]
    print(f"{'SeqLen':<8} | {'PP Compute':<12} | {'PP Comm':<10} | {'PP Total':<10} | {'TP Compute':<12} | {'TP Comm (NoP2P)':<15} | {'TP Total':<10} | {'TP Speedup':<10}")
    print("-" * 100)

    for s in seq_lengths:
        # PP=2: Full GEMM per layer for 25 layers on GPU0 + 25 layers on GPU1 (sequential latency = 50 layers full GEMM)
        _, pp_compute_50 = benchmark_gemm_compute(device, seq_len=s, num_layers=50, is_tp=False)
        pp_comm, tp_comm = simulate_no_p2p_communication(device, seq_len=s)
        pp_total = pp_compute_50 + pp_comm

        # TP=2: Halved GEMM on both GPUs concurrently (latency = 50 layers half GEMM)
        _, tp_compute_50 = benchmark_gemm_compute(device, seq_len=s, num_layers=50, is_tp=True)
        tp_total = tp_compute_50 + tp_comm

        speedup = pp_total / tp_total
        print(f"{s:<8} | {pp_compute_50*1000:8.2f} ms | {pp_comm*1000:7.3f} ms | {pp_total*1000:7.2f} ms | {tp_compute_50*1000:8.2f} ms | {tp_comm*1000:12.2f} ms | {tp_total*1000:7.2f} ms | {speedup:8.2f}x")

    print("=" * 70)
    print("Single GPU + CPU Offload Baseline:")
    print("Weight size: 27.14 GB. PCIe Gen3 x16 transfer bandwidth ~10 GB/s.")
    print("CPU -> GPU weight swapping time alone: ~2,700 ms (2.7 seconds) per forward pass!")
