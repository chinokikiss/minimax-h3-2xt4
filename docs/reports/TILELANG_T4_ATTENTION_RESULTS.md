# TileLang INT8 sparse attention: physical T4 experiments

Experiment date: 2026-10-13. Submitted and collected using the Kaggle CLI.

Hardware: Tesla T4 / SM75. Runtime: PyTorch 2.11.0+cu128, TileLang 0.1.15, CUDA 12.8. The Kaggle session exposes two T4s; these operator experiments execute on GPU 0 only.

**58/58 cases passed the prespecified implementation gates.** All records finished: True. Overall pass: True.

Worst passing implementation error: relative L2 0.000180485, max absolute 0.00183105, LSE max absolute 1.52588e-05.

## What was implemented

- TileLang INT8 Tensor Core QK with INT32 accumulation, per-token Q/K scales and shared K centering.
- FP32 online softmax/normalization; FP16 PV or block/channel-scaled INT8 PV with block-local probability quantization.
- Sol pooled residual blocks using FP16 K/V means and explicit valid counts; selected-block-only top-k and token-order local/global controls.
- D=64/128, BHND FP16 tensors, padded/ragged lengths and strided input preparation. Matching non-causal self-attention only; no GQA, dropout, arbitrary masks or ComfyUI integration in this qualification.
- Preparation, summaries and routing are currently PyTorch GPU operations. TileLang implements the tiled attention operator; preparation fusion and performance tuning are deferred.

## Implementation correctness versus approximation accuracy

The implementation gate compares each TileLang result with an independent streaming PyTorch reference using identical quantized operands, routes, pooling rules and probability-cast rules. Its limits were set before the first hardware run: relative L2 <=0.002, max absolute <=0.015 and LSE max absolute <=0.003. No NxN score matrix is used by this reference.

Dense-FP16 fidelity is reported separately against FP32 unquantized attention rounded to FP16. A passing implementation can still be a poor approximation to dense attention. The inputs are synthetic; these results do not certify MiniMax H3 generation quality.

| Approach | Passed cases | Worst implementation relative L2 |
|---|---:|---:|
| dense_int8 | 10/10 | 8.15321e-05 |
| sol_fp16_pv | 10/10 | 8.15321e-05 |
| sol_int8_pv | 10/10 | 0.000180485 |
| topk_12p5 | 10/10 | 8.15321e-05 |
| local_global | 10/10 | 8.15321e-05 |
| sol_tau0 | 2/2 | 7.93723e-05 |
| sol_tau0p5 | 2/2 | 7.93723e-05 |
| topk_25 | 2/2 | 7.93723e-05 |
| topk_50 | 2/2 | 7.93723e-05 |

## random2048_h4

| Approach | Selected blocks | Relative L2 vs dense | Kernel median ms | Complete call median ms |
|---|---:|---:|---:|---:|
| dense_int8 | 100.00% | 0.00923601 | 1.036 | 2.310 |
| sol_fp16_pv | 26.51% | 0.70933 | 1.024 | 2.550 |
| sol_int8_pv | 26.51% | 0.709401 | 1.122 | 2.613 |
| topk_12p5 | 12.50% | 2.6091 | 0.233 | 1.597 |
| local_global | 12.11% | 2.68435 | 0.220 | 1.499 |
| sol_tau0 | 56.40% | 0.548088 | 0.995 | 2.427 |
| sol_tau0p5 | 39.06% | 0.648343 | 1.010 | 2.495 |
| topk_25 | 25.00% | 1.70789 | 0.334 | 1.684 |
| topk_50 | 50.00% | 0.984807 | 0.546 | 1.928 |
| PyTorch FP16 SDPA | 100% | 0.00030388 | 0.619 | — |

## local2048_h4

| Approach | Selected blocks | Relative L2 vs dense | Kernel median ms | Complete call median ms |
|---|---:|---:|---:|---:|
| dense_int8 | 100.00% | 0.0238764 | 2.598 | 4.119 |
| sol_fp16_pv | 18.73% | 0.0238764 | 2.623 | 4.170 |
| sol_int8_pv | 18.73% | 0.0250638 | 2.842 | 4.892 |
| topk_12p5 | 12.50% | 0.0238764 | 0.536 | 1.829 |
| local_global | 12.11% | 0.0238764 | 0.384 | 1.761 |
| sol_tau0 | 43.70% | 0.0238764 | 1.859 | 3.341 |
| sol_tau0p5 | 28.44% | 0.0238764 | 1.874 | 3.338 |
| topk_25 | 25.00% | 0.0238764 | 0.347 | 1.743 |
| topk_50 | 50.00% | 0.0238764 | 0.570 | 1.951 |
| PyTorch FP16 SDPA | 100% | 0.000311143 | 1.487 | — |

## local8192_h28

| Approach | Selected blocks | Relative L2 vs dense | Kernel median ms | Complete call median ms |
|---|---:|---:|---:|---:|
| dense_int8 | 100.00% | 0.025767 | 112.832 | 141.605 |
| sol_fp16_pv | 16.35% | 0.025767 | 105.890 | 136.525 |
| sol_int8_pv | 16.35% | 0.0269231 | 112.977 | 145.189 |
| topk_12p5 | 12.50% | 0.025767 | 20.630 | 49.182 |
| local_global | 3.10% | 0.025767 | 11.205 | 39.160 |
| PyTorch FP16 SDPA | 100% | 0.000310109 | 72.742 | — |

## Interpretation and limits

These are warmed, untuned CUDA-event measurements, with compilation reported separately in JSON. Complete-call timing includes PyTorch preparation/routing, packing, allocation, validation and the TileLang kernel. Preparation currently builds V8 buffers for every approach; this is a common correctness scaffold and unnecessary overhead for FP16-PV modes. Peak memory is the benchmark process allocation while references and inputs are retained, not an isolated operator-memory claim.

The local-signal input gives each leaf a strong shared Q/K signature, making its own keys dominant. The random input has diffuse attention and makes omitted/pooled remote keys consequential. Equal outputs on the local-signal case do not demonstrate equivalent quality on a real video model. Denser tau/budget settings are included to expose the accuracy/work tradeoff.

The first hardware version passed 42/50 checks. Its eight D128 INT8-PV launch failures requested 69,120 bytes of shared memory. The corrected implementation reduces the summary-key tile to 32 columns and stages its distinct fragment layout through shared memory; this changes staging, not the mathematical pooling contract.

PTX exports confirm `.target sm_75`, INT8 `m8n8k16` and FP16 `m16n8k8` instruction families. No cp.async, WGMMA or hardware sparse-MMA path was found in those exports. Numerical checks ran on a physical T4; Compute Sanitizer and captured H3 tensor/generation qualification remain separate future checks.

## Reproduce

Run from the workspace with an authenticated Kaggle CLI:

```powershell
& 'C:\ProgramData\miniconda3\envs\comfyui\python.exe' -m unittest benchmarks.test_tilelang_attention_contract -v
& 'C:\ProgramData\miniconda3\envs\comfyui\python.exe' run_tilelang_t4_kaggle.py
```

The runner packages only the two experiment modules into a private notebook, pins TileLang 0.1.15, requests NvidiaTeslaT4, records hardware/source hashes and downloads the outputs. It does not modify existing model benchmark notebooks.

Minimal operator usage:

```python
from patches.tilelang_sparse_attention import prepare, make_routes, execute
p = prepare(q, k, v)  # FP16 [B,H,N,D] on an SM75 CUDA device
routes = make_routes(p, 'sol', tau=1.0)
out, lse = execute(p, routes, pooled=True, int8_pv=False)
```

Raw evidence directory: `C:\Users\Administrator\Documents\project\2xT4 minimax h3\tilelang_t4_results\attention_v2\tilelang_t4_results`. The source hashes in results.json must match the local operator and benchmark for this report generator to accept the run.
