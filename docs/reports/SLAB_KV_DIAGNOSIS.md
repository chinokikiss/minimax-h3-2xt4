# Local Video VAE KV-sharing diagnosis

KV sharing addresses missing attention context, but global RoPE coordinates and
Native INT8 query groups must also be preserved. A candidate using disjoint
Native-aligned query blocks and fresh global KV in every decoder block produced
Native bit-exact outputs on both saved workloads.

The full trained 36-block decoder ran on one RTX 5060 Ti with two independently
loaded replicas and two logical CUDA ranks/streams. ComfyUI is pinned to
`d49e888586dd8ae012c0667b33466b815fee07f7`; the loaded comfy-kitchen is 0.2.37.
Twenty-six cases completed. These are semantic experiments, not T4 performance
measurements. Production model code and the running Kaggle payload are unchanged.

## Controlled results

Relative L2 against fresh full Native raw decoding of the same saved input bytes.
The first four rows use 128-pixel overlap and full fade. The candidate uses no
overlap or fade. These are correctness ablations, not equal-work latency comparisons.

| Variant | 768×768, latent T=1 | 768×1344, latent T=2 |
|---|---:|---:|
| Independent slabs | 9.990% | 7.548% |
| Global RoPE only | 5.941% | 4.882% |
| Shared KV, local RoPE | 9.431% | 7.038% |
| Shared KV + global RoPE | 1.620% | 1.576% |
| Native-aligned query candidate | 0 — bit-exact | 0 — bit-exact |

Both tested overlaps, 64 and 128 pixels, give the same conclusions. All combined
KV/global-RoPE width-slab variants still fail the original 1% error gate. The
candidate and all four query-layout controls pass bit-exact final-output checks.
Seven selected intermediate blocks (1,2,3,6,12,24,36) also match Native exactly
on each candidate rank; all 36 blocks execute.

## Three causes

1. **Missing global context:** independent slabs attend only their own patches.
   KV must be exchanged afresh in all 36 blocks because hidden states change
   after each attention/FFN update.
2. **Changed positions:** Native normalizes width coordinates using the full
   width; cropped decoders renormalize local widths. Sharing KV while retaining
   local coordinates leaves roughly 7.0–9.4% error in the 128-pixel cases.
3. **Changed query quantization groups:** after restoring positions and context,
   first-layer floating Q/K/V are bit-exact. Quantized K/V and their scales also
   match. Quantized Q changes, and attention is the first divergent operation.
   Small differences then accumulate to about 1.6% final output error.

The pinned [Native Q quantizer](https://github.com/Comfy-Org/comfy-kitchen/blob/v0.2.37/comfy_kitchen/backends/cuda/sage_attention/quant_qk_int8.cu)
groups multiple sequence rows into a shared scale, including rows spaced eight
apart. Repacking width subsets changes group membership. Preserving whole
128-query blocks retains these groups. The measured byte comparisons establish
the difference independently of the source interpretation.

Two controls verify causality: evaluating full-query attention on each rank
restores exact outputs but duplicates attention work; splitting prequantized
Native query blocks also restores exact outputs but adds Q/output gathers.
Neither control is presented as an optimized runtime.

## Candidate and next implementation

The candidate maintains local hidden tokens, Native block/FFN operators, and
global coordinate slices. Each rank computes its owned Q/K/V, exchanges K/V,
and evaluates only its own queries against complete global KV. There are no
full-Q or full attention-output gathers inside blocks. Projected patch outputs
are joined once at the end. Four register tokens and the Native zero token are
owned exactly once; overlap duplication is absent.

The candidate currently performs the Native input projection once over the full
latent before distributing owned hidden tokens. That startup scope must remain
explicit in any performance test. The logical gather helper includes host
synchronization and size exchanges, so this remains a semantic prototype.

| Workload | Query ranges, including suffix | Ideal aggregate FP16 KV exchange across 36 blocks |
|---|---|---:|
| 768×768, latent T=1 | [0,1280), [1280,2309) | 0.634 GiB |
| 768×1344, latent T=2 | [0,4096), [4096,8069) | 2.216 GiB |

These are payload calculations, not observed PCIe/NCCL bytes. Restoring complete
attention increases work relative to independent cropped slabs. The old
approximate decoder's speedup does not predict this candidate's speed.

The next implementation steps are:

1. Keep the full token grid and five suffix tokens under Native-aligned query
   ownership, preserving complete query quantization groups on each rank.
2. Exchange current KV in canonical Native order every block. Preserve global
   K centering/anchor selection and V scales. Start with floating KV and the
   unchanged Native attention API.
3. Use fixed-shape communication with known ownership sizes and CUDA event
   dependencies. Remove diagnostic size exchanges, identity reorder copies,
   and redundant buffers.
4. Qualify sampler-produced latents, batch/temporal variation, and public-VAE
   normalization/cropping locally. Then run a bounded physical dual-T4
   correctness/latency comparison with identical timing boundaries.
5. Optimize packed INT8 KV transport only after its centering, scales and
   layout match Native and profiles show raw KV transfer is a bottleneck.

The diagnosis itself launched no additional Kaggle instance. Subsequently, a
separate user-authorized [physical VAE profiling job](https://www.kaggle.com/code/chinokiki9/minimax-h3-vae-kv-profile)
was submitted. See [profiling scope and cost hypotheses](../design/VAE_KV_PROFILE_PLAN.md).
That job subsequently completed: the aligned candidate is physically bit-exact
on all three workloads, with 1.388× / 1.479× / 1.825× raw decode speedups. See
[the physical report and public-decoding limits](VAE_KV_T4_RESULTS.md).
Its final local qualification additionally tests a synthetic 512×512, 124-frame
latent: the Native-aligned candidate remains bit-exact, while cropped-prefix
query controls fail at 1.608% relative L2 on that longer input. Their short-fixture
exactness must not be generalized. The existing P0/P1/P2 audit continues with
its qualified Native VAE path. Random benchmark latents establish numerical
behavior but do not establish visual quality on generated videos.

Evidence: raw measurements (`slab_results/kv_diagnosis.json`, local-only evidence),
verification summary (`slab_results/kv_diagnosis_summary.json`, local-only evidence),
[local diagnostic and candidate](../../benchmarks/diagnose_slab_kv.py).
