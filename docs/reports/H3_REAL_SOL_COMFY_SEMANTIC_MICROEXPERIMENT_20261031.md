# Real MiniMax-H3 Sol v2 vs ComfyUI Built-in Sol — Cross-GPU Semantic Micro-Experiment

**Date:** 2026-10-31
**Status:** **GPU capture submitted / Kaggle accepted v1**. Real QKV artifacts and CK replay results **not yet collected**.
**Kaggle:** https://www.kaggle.com/code/pamcks/h3-real-sol-comfy-capture-20261031

## Why two GPUs?

The pinned ComfyUI node `ComfyUI/comfy_extras/nodes_sparse_attention.py` invokes `comfy_kitchen.sol_attn` and, in the MiniMax-H3 producer path, `comfy_kitchen.sol_attn_chunked`. Local `comfy_kitchen.sol_attn_is_available` requires **SM80+**. The workspace's restored Sol v2 is an SM75 native kernel that explicitly refuses other capabilities. Therefore physical same-device A/B between the two kernels is impossible without porting one of them.

- T4 (SM75, physical Kaggle 2×T4): capture real MiniMax-H3 post-RMS/post-RoPE Q/K/V and compute Sol v2 output.
- Local RTX 5060 Ti (SM120): replay **exact serialized FP16 bytes** through ComfyUI's low-level `comfy_kitchen.sol_attn`. Verified locally on SM120; tau 1.0/1.3 and token augmentation 0/256 all returned finite results in a *synthetic API smoke only*, not in the real H3 experiment.
- No cross-device performance/speedup claim; only numerical output comparison.

## Real data provenance

- Fixed previously completed 0.85MP anime Turbo Ref2VA model input, **1216×704, 243 frames, eight NFE, seed 20261020**, unchanged prompt, two reference images, exact pinned offline-baked regular Turbo LoRA.
- Hook: `patches.h3_operator_policy.prepare_dit_attention` at the first actual DiT attention. Input tensors are **real post-RMS/post-RoPE FP16 QKV**, not random tensors or inferred latents.
- Each TP rank contributes **one actual head** (head 0), retaining the entire **65,166-token** sequence and **D=128**.
- Shape per saved tensor `[1,1,65166,128]` BHND FP16. Four tensors per rank: `q,k,v,sol_v2`. Estimated uncompressed tensor bytes ≈ **63.6 MiB/rank**; actual file size/hash are recorded at execution.
- Capture artifacts planned: `real_h3_sol_comfy_capture/generation/rank_results/rank[01].json` and `rank[01]_real_qkv.pt` plus `generation/result.json`, `batch_results.json`.
- Tensor-level SHA256, file-level SHA256, true PackedLayout segments, GPU architecture, K-centroid topology and Sol parameters recorded.
- Frozen pre-math-fix Sol v2, `tau=1.0`, H3 topology v2 on, protected tokens start at video boundary (4974 in the previous measured layout), no fused KC threshold, workspace pool or bitmap LUT.
- Single first attention then a controlled CaptureStop; no model denoise continuation or video decode.

## Local CK replay variants on *identical* real QKV

ComfyUI's low-level entrypoint `ck.sol_attn(q,k,v,...)` accepts BTHD and does **not** reproduce the full H3 chunked QKV producer state machine. Four variants in `benchmarks/replay_real_h3_comfy_sol.py`:

| Variant | Tau | Sink KV blocks | Sink Q blocks | Token augmentation |
|---|---:|---|---|---:|
| Matched protected-prefix | 1.0 | 0:ceil(video_start/64) | 0:ceil(video_start/64) | 0 |
| Comfy H3 default sinks | 1.0 | same | floor(audio_start/64):ceil(video_start/64) | 0 |
| Comfy H3 sinks + augmentation | 1.0 | same | same | 256 |
| Comfy UI defaults | 1.3 | same | same | 256 |

All use `topk_ratio=0.0` and `tail=True`. This matrix separates sink-policy changes, token augmentation and tau from inherent algorithm/kernel differences.

## Metrics and interpretation

- Same real QKV bytes asserted by SHA256 at replay, and same first-layer TP rank/head.
- Compare to T4 Sol v2 output: rel-L2, MAE, max absolute deviation, cosine similarity, output norm ratio, finite; segmented by text, references, generated audio, video; **72 per-video-frame rel-L2** measurements.
- In addition, on local SM120 compute explicit **FP32 dense QK^T/softmax/PV** for ~85 **sampled real query positions**, including video-frame centers and audio/video/conditioning boundaries. Compare **both** T4 Sol-v2 and CK outputs to this common dense reference to help distinguish a departure from density vs simple cross-implementation disagreement.
- Numerical equality is *not* expected: Sol v2 adds H3-specific temporal/local routing and has a different K-centroid/INT8 implementation; ComfyUI has a different sink_q policy, optional token augmentation, and chunked H3 producer semantics.
- Low-level `ck.sol_attn` replay **does not** qualify `ck.sol_attn_chunked`'s QKV projection chunks, step-to-step `kmean/vscale` reuse, RoPE/pre-normalization flow or the full diffusion/video trajectory. The experiment is an **attention semantics microgate**, not a video quality E2E.

## Submission and code

- Version: **v1**; accepted on 2026-10-31T05:42:53.787564Z.
- `docs/data/h3_real_sol_comfy_capture_20261031/gpu_submission.json`
- `docs/data/h3_real_sol_comfy_capture_20261031/gpu_job/experiment.ipynb`
- Notebook SHA256 `d74e99ce27708a7d0ba25ad128af1f36beb86082b544760cf80ce17229d04590`.
- Source-binding SHA256 `4d595927940fa3d44e62e756accc43c1880e1f8be3021501548ab8e9c05d10a7`.
- Local **35/35** unit/preflight gates passed, including metadata, frozen Sol SHA, source closure, input identity and exact QKV export path.
- Real capture: `benchmarks/real_h3_cross_backend_capture.py`.
- E2E stage/disk-bake launcher: `benchmarks/run_real_h3_cross_backend_batch.py`.
- Local replay: `benchmarks/replay_real_h3_comfy_sol.py` (must receive real Kaggle artifact files first).
- No status polling, live logs, automatic retry or background collection requested or performed.

### Replay command once capture files have been collected

Using the previously verified version-bound final-artifact collector, obtain the rank JSON and `.pt` files under a local final output directory. Then from the repository root (adjust the capture directory to the actual collection location):

```powershell
& "C:\ProgramData\miniconda3\envs\comfyui\python.exe" -m benchmarks.replay_real_h3_comfy_sol --capture-dir "docs\data\h3_real_sol_comfy_capture_20261031\gpu_final_v1\real_h3_sol_comfy_capture\generation\rank_results" --output "docs\data\h3_real_sol_comfy_capture_20261031\local_comfy_replay"
```

**Current verdict: capture GPU job accepted, numerical comparison not yet available.** Do not report similarity scores or quality outcomes until both rank payloads are retrieved and replay has actually run.
