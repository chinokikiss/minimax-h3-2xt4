# Dual-T4 causal profiling: completed measurements, partial attribution

Both Kaggle jobs have finished. The continuation completed all six passes in
3.97 hours, including setup and postprocessing. This report uses measured
request latency from the unprofiled passes, not notebook runtime.

## Validated performance

Full 50-layer TE/DiT, Native Ref2VA reference encoding, 512×512/124 frames,
four actual denoising evaluations, Native-aligned query video decode and
Channel Parallel audio decode. Each continuous pass has one discarded warmup
and four consecutive measured requests. Inputs and inference/capture source
hashes match. No unrelated denoiser evaluation was added.

| Configuration | Mean request latency | Sample SD | Allocation |
|---|---:|---:|---|
| Direct Sync | 396.27 s | 1.40 s | Original |
| P1 | 399.21 s | 1.58 s | Continuation |
| P1+P2 | 387.10 s | 0.76 s | Continuation |
| Offline+P1+P2 | **379.20 s** | 1.87 s | Continuation |

On the continuation allocation, adding P2 reduced mean latency by 12.11 s
(3.03%); offline loading reduced it by another 7.90 s (2.04%). Offline+P1+P2
was 5.01% faster than P1. These are observed differences from four requests
per configuration, not statistically established causal effect sizes. Cases
ran in a fixed order; timing/profile order was counterbalanced. Direct Sync
used another allocation and cannot establish that P1 alone improves or harms
latency. No Native-to-TP end-to-end speedup is claimed.

All 32 continuous measured requests across the two jobs passed their output
checks. Conditioning, video/audio sampled latents and decoded video pixels
are bit-exact to the saved physical Native reference. Audio waveform is
explicitly not bit-exact: it passes the already accepted Channel Parallel
rounding contract. Neither VAE fell back. The continuation reused the original
Native tensors; it did not execute a fresh Native pipeline. Transfer checksums
record receipt integrity, not independently known upstream tensor checksums.

| Unprofiled mean stage time | P1 | P1+P2 | Offline+P1+P2 |
|---|---:|---:|---:|
| Reference encoding + TE | 116.81 s | 101.56 s | 98.01 s |
| Sampler, all four NFEs | 247.79 s | 247.86 s | 246.90 s |
| Video decode | 32.35 s | 33.77 s | 30.90 s |
| Audio decode | 2.26 s | 3.91 s | 3.38 s |

Stage boundaries are host request spans, including loading/dispatch in those
stages. P2 improves the following request's TE startup; it does not accelerate
the current video's mathematical decode. Decode time varies enough that the
offline gain cannot all be assigned to the loader. Handoff gaps are 0.31–0.54
ms with no deliberate inter-request sleep or validation I/O.

## Evidence beyond stage totals

**CPU lookahead is adaptive, but GPU depth did not change here.** Actual DiT
snapshots record CPU depth reductions from 16 to 15/14/13 at different block
positions for each policy and rank. GPU `lookahead_changes` is empty under
the configured maximum of two. Code predicts deadlines from next-use distance
and compute EMA (`patches/weight_residency.py`, `prefetch`), then submits to
the HostWeightCache. This proves implemented prediction and observed CPU
adaptation; it does not prove that deadlines were accurate or optimally met.

**DiT residency survives repeated evaluations, but does not eliminate reloads.**
For the final measured request, P1+P2 and Offline+P1+P2 loaded 59 blocks on
rank 0 and 55 on rank 1, versus 200 block uses/rank across four evaluations.
Thus 70.5%/72.5% of per-use reloads were avoided. This is a load-count comparison,
not an exact cache-hit denominator: entry lookups, speculation and readiness
waits have different counters. CPU caches still evict frequently; batch-wide
counts include the warmup and next-request preparation and cannot be divided
blindly into per-request hit rates.

**GPU work and CPU waits must remain separate.** In Offline+P1+P2 request 0,
physical GPU activity occupies 64.15%/69.27% of request time. NCCL kernel unions
are 47.66/60.27 s on GPU 0/1; they include device-side waiting, not just payload
transmission. CPU-future wait union is 150.64 s across concurrent threads/ranks
and is not 150.64 s of exposed request stall. Weight H2D payload is 49.20 decimal
GB across both ranks; it does not include every reference/activation/collective
transfer. Only 0.64/0.88 s of physical memcpy overlaps useful compute in this
request. Those unions prove limited overlap, not H2D/NCCL interference.
Evidence: `continuous/offline_p1_p2/kineto/causal_analysis.json`, `requests[0]`.
The request's common monotonic interval is
10,638,377,855,935–11,020,939,715,072 ns; its sampler occupies request-relative
99.976–348.415 s. The quoted whole-request unions are not sampler-only totals.

**Attention is a dominant compute target.** For the same request, rank 0
operator attribution gives 136.39 s to `aten::_efficient_attention_forward`;
the dominant kernel is `fmha_cutlassF_f32_...sm75`. This identifies actual FP32
attention work, not an inferred weight-loading cause. The original classifier
checked the word `cutlass` before `fmha`, so its GEMM/attention category totals
are wrong. That classifier has been fixed with a regression fixture. Existing
operator attribution, raw kernels, latency and correctness are unaffected;
category totals need re-analysis of raw traces and are not accepted here.

**The largest idle gaps still have incomplete causal attribution.** For P1,
request 3/GPU 1 has a 18.078 s inactivity interval with a TE embedding future
and CPU preparation overlapping it (batch timeline 1627.307502–1645.385038 s).
P1+P2's largest listed gap, request 1/GPU 0 at
907.723612–920.877174 s, instead overlaps
DiT CPU preparation during TE. In Offline+P1+P2, request 3/GPU 0 has a 21.649 s
gap at 1680.116039–1701.764906 s between D2H and H2D operations, with DiT
preparation in the TE context. Each batch timeline has its own zero point.
These facts identify the preceding/following operations and overlapping work;
they do not prove that the preparation thread was the sole blocker. Exact
monotonic intervals and candidate event IDs are in each case's
`causal_analysis.json/significant_stalls`; readable excerpts are in
`ROOT_CAUSES.md`. Do not add the two GPUs' idle gaps into end-to-end savings.

## Audit status and limitations

| Area | Status | Evidence / remaining issue |
|---|---|---|
| Actual full-model path and Native output checks | PASS | 50 layers, four NFEs, both default parallel VAEs, zero-gap requests; Native/CP contracts pass |
| Lookahead | PARTIAL | CPU depth changes and predicted deadlines observed; GPU depth capped at two; deadline success and cross-model launch causality incomplete |
| Cache efficiency | PARTIAL | DiT reuse confirmed; continuing CPU evictions and ~49.20 GB weight H2D/request; aggregate counters are not critical-path costs |
| Async/memory safety | PARTIAL | No OOM or output defect; event-gated leases, last-use retirement and shared HostBudget in code; not a proof of race freedom for all workloads |
| Transfer/NCCL contention | PARTIAL | Actual overlapping memcpy/NCCL intervals and rank imbalance; no isolated contention intervention |
| Fair performance comparison | PARTIAL | Same workload, synchronized output-ready timing, matched captures and warmup exclusion; Sync on another allocation, fixed case order, four-step workload only |
| Kernel breakdown and stall attribution | PARTIAL | Operator/launch correlation and interval unions present; attention category bug needs re-analysis and major idle gaps lack a sole-blocker proof |

During profiled measured requests, peak process-tree RSS is 24.02–24.15 GiB,
minimum system-available RAM is 5.90–6.14 GiB, and peak driver VRAM reaches
14.24 GiB on GPU 0 / 13.86 GiB on GPU 1. These include profiler memory and
retained CPU outputs; RSS can include shared pages. Shared HostBudget peak is
15.48 GiB against a 24 GiB limit, with no recorded admission denials. No OOM
occurred. This is not proof that RAM was maximally exploited or that larger
residency windows are safe. Nsight Systems is unavailable; Kineto/CUDA events,
NVTX-labelled ranges and NCCL logs provide the current evidence. Sampled NVML
bus traffic is approximate and includes collective traffic.

## Ranked next opportunities, without changing inference during this audit

1. **Native-preserving attention execution.** The mapped FP32 attention work
   is approximately 135–136 s/rank in representative requests. Inspect exact
   sampler kernel intervals and dispatch before testing a targeted kernel or
   layout change. This is the largest measured compute opportunity, not a
   promised latency reduction. Preserve masks, precision and Native output
   semantics; independent-shard quantization or lower precision cannot be
   accepted merely for speed.
2. **TE startup/vision residency and preparation priority.** P2 already removes
   15.25 s from the reference+TE stage on this allocation, while embedding waits
   and large handoff idle gaps remain. Audit the reference/startup consumer
   launch chain and prevent speculative DiT preparation from delaying it.
   Reuse safe CPU startup state under the existing shared budget. The observed
   12–22 s individual gaps are diagnostic opportunities, not additive or
   proven recoverable savings.
3. **NCCL dependency and rank-arrival overhead.** Approximately 45–60 s of NCCL
   device activity per rank warrants per-collective dependency inspection.
   Attribute rank-entry skew and the quantization/global-reduction dependency
   first, then test only changes preserving collective numerical semantics.
   The measured NCCL occupancy is an upper bound on activity, not removable
   communication latency. CPU/H2D optimizations alone did not reduce the
   roughly 247 s sampler stage substantially in this run.

These rankings concern measured opportunity size; actual improvements remain
unverified. Re-analysis and missing dependency attribution should precede new
GPU experiments. The current observed latency choice is Offline+P1+P2, with
P1+P2 as the simpler fallback; the measured additional offline benefit is 2.04%.

## Artifacts

- [Continuation notebook and raw output files](https://www.kaggle.com/code/chinokiki9/minimax-h3-causal-pipeline-resume).
- [Original Sync/pilot captures](https://www.kaggle.com/code/chinokiki9/minimax-h3-causal-pipeline-audit).
- Local continuation evidence: `kaggle_causal_resume_output/causal_audit_results/`.
- Local Sync/pilot evidence: `kaggle_causal_audit_output/causal_audit_results/`.
- Machine-readable comparison: `semantic_results/causal_pipeline_summary.json`.

Raw Kineto/custom traces and full timelines remain in Kaggle outputs. Local
summary/correctness/analysis records and NCCL logs have been downloaded; full
continuation raw traces are not yet all copied locally. Checkpoint reuse verified
all 200 objects (44.03 decimal GB), with no conversion or local weight upload.
