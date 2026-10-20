# MiniMax H3 P012 Spawn Repaired Audit: completed results

Kaggle **version 2 is COMPLETE**. All four configurations finished successfully
on two physical Tesla T4s, with PyTorch 2.11.0+cu130. Downloaded evidence was
revalidated locally by `analyze_loading_audit.py`; it accepts all four cases
and lists no pending configurations. No OOM or traceback appears in the four
case logs. Shared managed-host reservations return to zero after every case.

Notebook: [MiniMax H3 P012 Spawn Repaired Audit](https://www.kaggle.com/code/chinokiki9/minimax-h3-p012-spawn-repaired-audit).

## Request latency

Full 50-layer TE/DiT, 512x512 audiovisual output, four denoising evaluations,
Native vision/reference encoding and Native video/audio decoding. Output is
124 video frames and 165,600 stereo audio samples. The numerical reference is
reused from the previous verified physical session; its latency is not a
paired baseline for this allocation.

| Configuration | Cold request, seconds | Warm request, seconds |
| --- | ---: | ---: |
| Direct rank-local loading, synchronous | 392.893 | 385.477 |
| Direct + P1 CPU/GPU scheduling | 398.880 | 375.533 |
| Direct + P1 + P2 next-request TE preparation | 405.572 | 366.073 |
| Offline rank-local checkpoint + P1 + P2 | 404.386 | 358.179 |

Offline P1/P2 is the fastest observed warm configuration, 27.30 seconds
(7.08%) below direct synchronous. Direct synchronous is the fastest cold
configuration. P1 improves warm latency by 2.58%, P2 adds 2.52%, and offline
adds 2.16%, measured against their adjacent controls. Cold P1 and P2 regress
by 1.52% and 1.68%; offline improves its adjacent cold control by only 0.29%.
There is only one cold/warm pair per case. These are descriptive results,
not statistically established gains or a universally optimal configuration.

All **40 output comparisons are Native bit-exact**: conditioning, sampled
video/audio latents, decoded video pixels and waveform, for both requests
in every configuration. The spawn repair is limited to the launcher and
configuration transport; the analyzer verifies that exception explicitly.

## What the evidence establishes

- **P0 reuse works:** the 200 published offline shards were restored from
  their immutable HF revision, without conversion or uploading weights from
  this machine. Offline download was about 573 seconds, overlapping setup.
  The separate binding/verification stage took about 543 seconds; that cost
  is outside request latency and matters for short-lived instances. This run
  does not compare against the historical full-Native-block construction path.
- **P1 produces a modest warm gain:** warm sampler time falls from 238.66 to
  231.26 seconds. It does not reduce the observed number of DiT block loads:
  all configurations load 55 blocks per rank for 200 block uses, avoiding
  72.5% of potential loads. More residency is not the explanation for its gain.
- **P2 performs real CPU lookahead:** direct P1/P2 completes both startup
  preparation groups on each rank during decode, with about 11 seconds of
  cold overlap and 10.3 seconds of warm overlap. On the next request both
  groups hit, and startup wait is approximately zero. This prepares the next
  request, not the current video decode. Cold video decode rises from 28.82
  to 33.00 seconds, consistent with possible contention but insufficient to
  isolate its cause.
- **Memory remains tight on GPU:** the largest monitored driver usage is
  15,283,191,808 bytes on a 15,636,037,632-byte T4, leaving about 336 MiB at
  that observed peak. Minimum available system RAM is about 8.79 GiB. The
  managed-host ceiling in this audit is 24 GiB, with peak reservations about
  11.05 GiB; this is distinct from total RSS and system RAM usage.
- **Host cache efficiency remains limited:** reported aggregate block-cache
  hit fractions range from roughly 3% to 19%, depending on case/request.
  These counters include validation/background work and are not isolated
  timed-request hit rates. Startup-cache P2 hits are a separate measurement.

## Next actions

1. Keep direct synchronous available for cold single requests; offline P1/P2
   is the observed warm winner. Preserve offline artifacts across restarts
   rather than repeating restore/binding on every short run.
2. Profile and prioritize the remaining reference-encoding/TE stage: even the
   best warm request spends 95.8 seconds there. Split Native vision/reference
   compute from block preparation and exposed loading waits before choosing
   another loader change.
3. Gate or throttle P2 when no next request is known, and profile its decode
   contention. Warm readiness improves, while the current cold request gets
   slower. Do not infer current-request decode acceleration from overlap.

This job uses **Native VAEs**. It does not validate or time the subsequently
integrated Native-aligned video queries or Channel Parallel audio default.
An integrated comparison of those decoders must retain the observed GPU
headroom and account for additional parent/worker replicas.

Profiler traces cover an extra full-depth DiT evaluation outside request
timing, not TE/VAE. NVML PCIe traffic is sampled, not exact NCCL/weight wire
bytes. Runtime-cold does not establish a physically cold filesystem cache.
These four-step results cannot be extrapolated to a full longer sampling
schedule without measuring its actual evaluations and reuse.

Evidence: `kaggle_p012_spawn_repaired_v2_output/pipeline_audit_results/`
contains the manifest, hardware, four result JSONs, eight rank traces, NCCL
logs, `p012_analysis.json` and `P012_RESULTS.md`. Local validation output is
`kaggle_p012_spawn_repaired_v2_output/validation.log`.
