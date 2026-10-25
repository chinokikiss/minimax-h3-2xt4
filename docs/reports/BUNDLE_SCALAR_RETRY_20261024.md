# Single replacement of the bundle/Scalar profile — 2026-10-24

This replaces the [failed version-1 batch](BUNDLE_SCALAR_WORKER_PROFILE_20261023.md).
It retains the same 51,220-token inputs, base shards, adapter, operators, tensor
flow and pinned dependencies. Only benchmark instrumentation, failure isolation,
deadlines and duplicate-attempt guards change. No production operator changes
are included in this capsule.

**Current state:** all 67 tests pass from the exact isolated capsule and on
Kaggle CPU's native Python 3.13.15. CPU preflight version 3 completed in 113.90 s,
including the concurrent profiler checks. Kaggle accepted one replacement GPU
submission, version 2 of
[`pamcks/h3-dit-scalar-single-profile`](https://www.kaggle.com/code/pamcks/h3-dit-scalar-single-profile),
at 2026-10-24 09:25:33 Beijing time. Final artifacts were collected on the same
day: the periodic-flush failure-path gate failed on real dual T4s, stopping
before model downloads. Zero DiT calls ran. There is no new attention timing
comparison. The [evidence record](../data/bundle_scalar_retry_20261024.json)
tracks acceptance separately from execution and validation.

## Version-2 result

The actual stack was two distinct Tesla T4s, SM75, native Torch 2.11.0+cu128,
CUDA 12.8, Kitchen 0.2.37 and TileLang 0.1.13. The source/capsule binding and
all 28 returned artifact files were verified. Environment setup took 80.54 s;
the execution phase exited after 19.32 s, including 15.40 s for its two capture
cases. These are setup/probe durations, not DiT latency.

| Capture protocol | Observed result |
|---|---|
| `--cuda-flush-interval=1000`, three-second dwell, then forced worker termination | Failed: CUDA kernel table absent; both workers' four additions executed, but their kernel records were not preserved |
| Same periodic setting, explicit `torch.cuda.profiler.stop()`, graceful worker exit | Passed: four correlated additions per rank on separate GPUs |

The second report contains ten kernels in total, including initialization; eight
are correlated with the required four additions per worker. The first report
has NVTX and runtime API tables, but no CUDA kernel table. Local independent
coverage checks reproduce both outcomes.

The periodic-save change was insufficient to protect a force-terminated worker
on this stack. The expensive stage correctly remained blocked: no weight or
adapter downloads, TE/reference preparation, Scalar NFE, Kitchen NFE, or resident
replay started. The CPU profiler passed CPU qualification; it has no new live
DiT evidence from this run. The original Norm/Residual anomaly is unresolved.

The next capture protocol needs explicit worker-side flush checkpoints and
validated stop/restart semantics. It cannot assume that a periodic Nsight save
interval makes failure cleanup safe. Existing explicit-flush evidence can be
retained, but a changed checkpoint/restart protocol still needs a minimal gate
before model execution. No performance or accuracy conclusion is drawn from
this failed gate.

The initial CLI transfer disconnected after 20 files. Recovery used one
version-2 output inventory request, preserved those 20 validated files and
downloaded only eight missing files. No status/live-log polling, new GPU job or
automatic resubmission occurred. The initial transfer failure receipt is kept
alongside the completed recovery receipt. Local analysis is in
`D:/attention_eval/bundle_scalar_retry_20261024/analysis_v1/verified_results.json`.

## Avoiding duplicate experiments

The preceding GPU job is known to have finished: its final outputs were
collected, and the returned execution receipt records failure. The replacement
uses the existing notebook `pamcks/h3-dit-scalar-single-profile` rather than a
second notebook. A local exclusive submission claim and receipt permit one
replacement GPU push; an ambiguous response is not retried automatically.

There are exactly two planned denoiser cases, in this order:

1. Scalar Sparse, one NFE on both TP2 ranks.
2. Kitchen control, one NFE on both TP2 ranks.

Each case has an exclusive `profile.claimed` file containing config/input
hashes. A second invocation in the same output directory fails before launching
Nsight or a model process. A failed case is preserved and the other case is
attempted once; the batch never retries an NFE. This fixes version 1's behavior,
where a failed Kitchen control prevented Scalar from starting.

Conditioning/reference inputs are prepared once and shared between cases.
Weights and the adapter are downloaded once per job, directly from their
existing immutable HF locations. No new weight Dataset, baked model or adapter
upload is created. Each backend retains its own equivalent CPU on-demand LoRA
preparation, so its input weights follow the same numerical contract. This
protocol does not eliminate the existing rank-local CPU reconstruction work.

The representative one-block compilation warmup and the resident operator
replay remain outside the measured NFE. Neither runs another 50-block call.
There is no full sampler trajectory, Video decode, Audio VAE or media generation.

## Repairing evidence collection

| Version-1 failure | Replacement behavior |
|---|---|
| Worker NFE deadline was 600 s while CPU bundle preparation remained on the command path | Benchmark-only deadline is 1,500 s; production runtime defaults are unchanged |
| Timeout abort killed workers before final CUPTI flush | Nsight automatically saves CUDA buffers every 1,000 ms; normal explicit flush/graceful exit remains |
| Partial main trace was not exported by the failed launcher | Export coverage diagnostics even after a failed NFE; never generate timing summaries from a failed/incomplete trace |
| CPU stacks/call summaries were written only after NFE return | Atomic per-rank JSON checkpoints every 15 s, with current block/progress and completed-call records |
| Live cProfile used a per-thread clock and produced impossible timings | No cProfile during the live NFE; direct per-call wall/thread-CPU timers and Python stack sampling remain |
| Resident cProfile could overlap weight-cache work | Release the model and drain/close both worker CPU caches before the replay |
| Invalid profiler times could be treated as measurements | Use cProfile's default monotonic wall clock and reject negative/nonfinite/impossible timing entries |
| Coverage only considered runtime APIs | Correlate both runtime and driver launch APIs |

Resident cProfile function durations are explicitly **wall time**, not thread
CPU. Thread CPU is measured separately around each owner-thread call. The six
resident modes and three counterbalanced repeats remain; byte-exact output is
required. Both ranks write replay results, and the parent validates both.

The old four lifecycle cases are retained as historical proof and are not
repeated as a matrix. The replacement validates only the changed collection
protocol: forced exit after a three-second dwell with periodic saving, and
graceful explicit flush with periodic saving. Both must preserve four known
kernels per worker on separate GPUs before any model download. This tests the
failure path that version 1 left unprotected. Unsupported profiler flags or
missing worker CUDA records stop the expensive stage.

A complete result still requires successful NFE output, requested backend
counts, correlated kernels in all 50 block scopes per rank, two clean worker
exits, and explicit flush receipts. Partial data are useful only for diagnosing
progress and capture. Nsight Compute counter permissions are not reprobed;
there is no new Compute counter claim.

## CPU qualification and source identity

The 67-test CPU suite includes concurrent disk activity during resident
profiling, invalid-clock rejection, incremental output before `finish()`,
driver-only CUDA correlations, and an exclusive backend execution claim.
The Kaggle CPU receipt must explicitly confirm these tests ran on its native
Python version before the GPU bootstrap accepts it.

The capsule has 326 files and SHA256
`28af6446c16c416dcad2a4c03bcf85906e2489225fcb8c2b4a792ded3ba7a965`.
It is delivered from immutable Git commit
`750f8d64955174b1ba307213d6ed0b6f909a5bb9`.
The 11 changed files are all under `benchmarks/`; the frozen ComfyUI archive,
core wheel lock, production patches and node are unchanged from the preceding
submitted capsule. Kitchen remains 0.2.37 and TileLang 0.1.13, with Kaggle's
native Torch/CUDA family retained.

The GPU worker process cap is 1,850 s per case, the case launcher cap is 2,000 s,
the model/input/case batch cap is 4,600 s, and the outer execution cap is 4,800 s.
These are failure bounds, not promised latency. Input loading, compilation,
CPU preparation, trace export and resident replay are reported separately.

Local evidence is under `D:/attention_eval/bundle_scalar_retry_20261024/`.
No Kaggle status/list/live-log polling or timed final collector is used. Final
GPU artifacts will be collected once after the user indicates completion.
