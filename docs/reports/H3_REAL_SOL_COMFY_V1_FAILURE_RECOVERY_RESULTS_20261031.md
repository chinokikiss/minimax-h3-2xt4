# H3 Real Sol Comfy Capture 20261031 — Failed v1 forensic recovery and completed real-QKV comparison

**Source Notebook:** https://www.kaggle.com/code/pamcks/h3-real-sol-comfy-capture-20261031
**Version:** 1 (Kaggle notebook **FAILED**, post-processing)
**Investigation:** one-time collection of version-bound final logs, metadata and payload artifacts; no status polling, no additional Kaggle job.
**Effective scientific status:** **both real-data captures complete; local ComfyUI Sol cross-GPU replay complete**.

## 1. Exact cause of Kaggle failure

The 2x T4 job successfully ran: 100 DiT TP2 base shard downloads, offline Turbo LoRA disk bake, 100 TE TP2 downloads, references/TE, real first denoise, interception of **65,166-token post-RMS/post-RoPE QKV**, execution of **Sol v2 on each T4**, and serialization of Q/K/V/Sol-v2 outputs. Both ranks logged:

`REAL_H3_SOL_CROSS_CAPTURE_COMPLETE 0 66732515`
`REAL_H3_SOL_CROSS_CAPTURE_COMPLETE 1 66732515`

After that, the reusable driver `benchmarks/real_h3_attention_microgate.py` still expected the **old benchmark** JSON schema:

```python
state['summary']=dict(
    old_rel_l2=[r['summary']['old_rel_l2'] for r in ranks],
    ...
)
```

The new capture intentionally produces `tensor_sha256` / `payload_file` / `sol` / `layout`, and does not produce `r['summary']`. Consequently the parent raised `KeyError: 'summary'` at line 382; the batch runner and Notebook then recorded `status=failed`. This happened **after both T4 captures completed**, not while running kernels. The GPU failure label is correct for the wrapper, but not proof the underlying experimental data are invalid.

**Fix for future captures:** the new driver should summarize `payload_file`, `tensor_sha256` and layout under a new `capture_summary` schema instead of invoking the old microgate's `old_rel_l2` assembler. Do not fabricate the old `summary` keys. This version has been recovered **without** rerunning the GPU.

## 2. Raw data evidence

- T4: both Tesla T4, SM75, rank 0 and 1, NFE 1.
- Real MiniMax-H3 8-step Turbo Ref2VA input from the prior successful 1216×704/243 frame/seed 20261020 experiment; identical references/prompt/baked LoRA.
- Exact **post-RMS/post-RoPE** Q/K/V after H3 projection, not random QKV.
- **One actual head per rank**, shape `[B=1,H=1,N=65166,D=128]` FP16. Four tensors per rank: Q, K, V, and restored **pre-math-fix Sol v2** output at tau 1.0.
- H3 PackedLayout: text 0–2528, two reference images 2528–3344 / 3344–4164, audio 4164–4974, video 4974–65166. Video: 72 latent frames × 836 tokens.
- File size **66,732,515 bytes per rank**. SHA256 checked against captured provider-side JSON (both PASS), and individual tensor SHA256 validated on local SM120 replay.
- Rank0 payload SHA256: `4c9116fd7f603539c4e49e7e440c034c893e246f8649371b21376483fafd34e8`.
- Rank1 payload SHA256: `9138f989a127295555ace8ca3914e007c0f3d6d9cd5f26f296c48789ad80bc91`.

Files:
- Final provider log: `docs/data/h3_real_sol_comfy_capture_20261031/failed_v1_provider_logs.txt`.
- Rank metadata: `docs/data/h3_real_sol_comfy_capture_20261031/failed_v1_artifacts/real_h3_sol_comfy_capture/generation/rank_results/rank[01].json`.
- Local actual QKV: `docs/data/h3_real_sol_comfy_capture_20261031/failed_v1_payloads/real_h3_sol_comfy_capture/generation/rank_results/rank[01]_real_qkv.pt`.
- Validated offline recovery receipt: `docs/data/h3_real_sol_comfy_capture_20261031/recovered_v1_result.json`.

## 3. Successfully completed ComfyUI built-in Sol replay

Local device **RTX 5060 Ti 16GB, SM120**, Torch `2.12.1+cu132`; direct low-level `comfy_kitchen.sol_attn` call, which actually supports this device. Exactly the T4-captured FP16 QKV were replayed. This is not the H3 `sol_attn_chunked` production pipeline.

**Output rel-L2 of CK Sol vs T4 restored Sol v2** (lower is closer to Sol v2):

| ComfyUI low-level Sol variant | Rank0 all | Rank1 all | Rank0 video | Rank1 video |
|---|---:|---:|---:|---:|
| τ=1.0, matched sink KV and dense prefix, token_aug=0 | **1.3055%** | **2.6213%** | 1.3799% | 2.9443% |
| τ=1.0, ComfyUI H3 sink_q, token_aug=0 | 4.2120% | 3.4228% | 1.3799% | 2.9443% |
| τ=1.0, ComfyUI H3 sink_q, token_aug=256 | 3.4064% | 2.9910% | 2.4919% | 2.8049% |
| τ=1.3, ComfyUI H3 sink_q, token_aug=256 | 5.2645% | 4.7708% | 4.7910% | 4.7967% |

Per generated-audio output rel-L2 vs Sol v2 was **0.3154%** rank0 and **0.5865%** rank1 for all four variants. This is because the audio query rows remain exact under all tested CK sink policies; the small residual reflects differences in numerical implementation. The distinct ComfyUI `sink_q` settings substantially change the conditioning-prefix outputs, even when generated video outputs are the same.

Worst video latent-frame rel-L2 vs Sol v2:

| Variant | Rank0 max of 72 frames | Rank1 max of 72 frames |
|---|---:|---:|
| matched sinks, τ=1, extra0 | 1.71% | 5.33% |
| CK H3 sinks, τ=1, extra0 | 1.71% | 5.33% |
| CK H3 sinks, τ=1, extra256 | 3.08% | 4.58% |
| CK H3 sinks, τ=1.3, extra256 | 6.11% | 7.94% |

### 85 real-query FP32 Dense reference

A **common numerical target** (not a cross-GPU speed claim) was computed locally from exactly 85 selected *real* queries covering conditioning/audio/video boundaries and centers of all 72 latent video frames. FP32 QK\u1d40 softmax PV on the full real K/V.

| Implementation | Rank0 rel-L2 vs sampled Dense | Rank1 rel-L2 vs sampled Dense |
|---|---:|---:|
| T4 Sol v2, τ=1.0 | **5.3483%** | **2.8443%** |
| CK Sol, matched sinks, τ=1.0, extra0 | 6.0101% | 4.0387% |
| CK Sol, CK sink, τ=1.0, extra0 | 7.1502% | 8.7556% |
| CK Sol, CK sink, τ=1.0, extra256 | 7.2490% | 8.8262% |
| CK Sol, CK sink, τ=1.3, extra256 | 9.2438% | 9.8780% |

These 85 selected queries and two rank-local heads are not a random or exhaustive statistical sample of all layers, time steps, or attention heads. On **this sample** the restored Sol v2 is closer to true Dense than any of the tested ComfyUI low-level Sol settings. This does **not** establish that whole-video Sol v2 has higher generation quality.

## 4. Interpretation and limits

- Backend math and routing are **not semantically identical**, even at τ=1.0, especially because Sol v2 has H3-specific temporal/local protections and its own K-centroid/INT8 quantization.
- ComfyUI H3 sink selection and `token_aug` strongly affect fidelity. The built-in typical τ=1.3 / token_aug256 configuration is farthest from Sol v2 in this comparison. τ is not an accuracy-equivalent parameter across the two implementations.
- Not an E2E video-generation comparison, and **cannot** prove the reported severe per-second semantic discontinuities come from either kernel.
- No cross-device speed comparisons: T4 and SM120 kernel times are not directly comparable.
- Both execution and outputs are captured real data: there are no synthetic QKV used in the reported numerical results.
- Neither built-in `comfy_kitchen.sol_attn_chunked` nor its step-wise reused kmean/vscale has been tested. Results apply only to low-level `ck.sol_attn` on the same input data.

## 5. Decision

**Kaggle v1 remains failed in provider history, but the scientific comparison is now fully recovered and locally completed.** Preserve v1 and do not resubmit identical work.

The reuse bug is in the *final summary schema*, not the Sol kernels or real H3 input. A future capture job should use a purpose-built `capture_summary`, and test the exact schema of its final report before consuming new Kaggle GPU time.

Local comparison output: `docs/data/h3_real_sol_comfy_capture_20261031/local_comfy_replay/result.json`.
Offline validation tool: `benchmarks/recover_real_h3_cross_backend_result.py`.
