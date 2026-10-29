# DiT trusted-bake hash ablation — 2026-10-29

Status: the two-T4 A/B completed. Trusting the bake reduced median eight-hop
selection + denoiser RPC time from **112.639 s to 65.233 s**, a **42.087%**
reduction and **1.727×** speedup in this reduced streaming fixture. All outputs
are byte-identical. This is not a full-model or full-pipeline speedup claim.

## Question and switch

Does trusting an already validated bake remove enough work from the inference critical path to matter, with the persistent queue and P0/P1 scheduling enabled?

`RuntimeConfig.verify_dit_weights` and the appended `MiniMaxH3TP2Loader` widget default to `True`. Setting the flag to `False` skips DiT safetensors payload SHA256 and loaded tensor-state hashes, including the PTD monolithic refiner/output-head overlays. It also covers ordinary DiT TP2 bundles and disk-baked LoRA replacements. TE verification remains enabled. Native safetensors readers have no newly introduced payload hashing to bypass.

The trusted arm still checks small descriptors, manifests, bake receipts, provenance, tensor schemas, and size/mtime signatures. This is a bulk payload/tensor hashing ablation, not removal of all metadata validation. Bake-time numerical/provenance checks and immutable download validation remain outside inference timing. A trusted file with changed bytes and unchanged size/mtime can be consumed without detection; the corruption test explicitly demonstrates that difference. The mode is therefore a deliberate trust assumption, not an equivalent integrity check.

The flag participates in reader identity to prevent cross-mode cache reuse. Queued hop readers copy the mode without mutating the admitted base reader. Existing workflows retain verification unless explicitly opted out; the new widget is appended after existing loader widgets.

## Why first use and reload differ

The current verified path already caches successful file and tensor proofs per reader. Later loads of that immutable identity use signature checks. PTD has eight hop-specific overlays/readers, so their first consumption can still perform substantial hashing even when the GPU window is persistent. P1 can overlap verification and refiner preparation with the preceding hop. Summing hash durations would overstate the possible latency reduction because work on different threads/ranks can overlap.

## Controlled T4 experiment

- Require two distinct physical T4 devices, SM75, and the qualified native cu128 environment. No SM120 substitute.
- Use public immutable base shards and a real trained PTD subset: two full-width DiT blocks, both refiners, full output heads, eight hops. No local LoRA download.
- Bake one bank once and use that exact bank for both modes. No full-model bank, full H3 generation, TE, or VAE execution.
- Keep P0 future-use reclamation, P1 refiner prefetch, persistent queue, current production operators, resource budgets, inputs, seed and FP32 sampler state carry identical.
- Rehearse once to fill binary caches. Then launch fresh services/readers for three trials per arm in order verified/trusted/trusted/verified/verified/trusted.
- Separate source/package setup, downloads, bake, worker startup, controller admission and AOT preparation from measured selection + denoiser RPCs. Report the separate phase timings as well.
- Compare all eight video/audio denoiser output hashes and final sampler-state hashes outside the timed RPCs. Trusted workers must record zero bulk payload/tensor hash calls; verified workers must record actual hash work.
- Add a sequential CPU reader probe on the T4 host: first load and two reloads for both owned ranks, using fresh readers and both arm orders. This isolates proof reuse; it does not represent physically cold disk I/O.

The main comparison reports raw trial samples, medians, ranges, percentage saved and speedup. Three repetitions support a practical effect-size estimate, not a strong statistical significance claim. The reduced fixture emphasizes streaming/host costs and cannot establish full-model or long-sequence E2E speedup.

## Evidence and interpretation

`benchmarks/dit_hash_observer.py` is benchmark-only. It records payload, tensor-state and metadata hash bytes, wall time, thread CPU time and thread identity in workers. Production runtime does not import the observer. Existing worker probes record CPU preparation spans, RPC dispatch times, CUDA default-stream intervals, RSS and VRAM peaks. Default-stream intervals include idle/waits; they are not kernel-only time or GPU utilization.

Use parent RPC latency to decide whether hashing is on the critical path. Compare per-rank hash spans and refiner preparation against those RPCs to explain the difference. Startup improvements are reported separately. Reload probes should show little hash-related benefit once the verified reader has cached proofs; a contrary result requires investigation rather than attribution to hashing alone.

No Nsight trace or full-pipeline result is implied by these counters. Artifacts remain under `dit_hash/`, with per-case JSON and durable per-rank worker reports. The job must fail if any arm is incomplete or outputs differ.

## Qualification and delivery

Local conda CPU regression: 207 passed, no failures/errors/skips. The extracted frozen capsule's dedicated gate passes 49 contracts, without fetching model weights or using a GPU. Frozen legacy ComfyUI, dependency lock bytes, Kitchen 0.2.37 and TileLang 0.1.13 are preserved; unrelated dirty worktree changes are excluded through a reviewed Git-index overlay.

`benchmarks/build_dit_hash_job.py` requires a matching Kaggle CPU setup and experiment receipt before GPU submission. The CPU job has a 600-second setup watchdog and 180-second contract watchdog. The T4 experiment has a 2400-second execution watchdog and 3300-second provider limit, zero dataset/output mounts, no credentials and no status polling. Artifact transfer may be retried under the repository's existing policy.

Local evidence: `D:\attention_eval\h3_dit_hash_ablation_20261029\`. T4 results
are recorded below. Verification remains the runtime default; this experiment
does not change that policy.

Kaggle CPU preflight version 1 failed; final artifacts were collected once.
Immutable source commit: `e1093f1c3c9aeba12d9d0f42d5b47815d8d7f268`; capsule SHA256: `1246b03ba71c886884d39b0cc91d5c18ce0f255efeb7ed63239ee6327b155e09`.
Source hashes and capsule identity match. Core imports and legacy API signatures
passed on native Torch `2.11.0+cpu`, Python `3.13.15`, Kitchen `0.2.37`, TileLang
`0.1.13`. The regression suite ran 207 tests: 206 passed and one errored.

The failing test mocked GPU execution to exercise the scaled-transport fused
epilogue contract with CPU tensors, but left the GPU NVTX label unmocked.
CPU-only Torch correctly raised `NVTX functions not installed`. The correction
replaces that label with `nullcontext` inside this single CPU test, matching
its other mocked GPU operations. Production operators, GPU NVTX instrumentation,
dependencies, test count and A/B items remain byte-exact. The retry capsule
changes only `benchmarks/test_h3_dit_fusion.py`; it passes 207 local regression
tests and 49 dedicated contracts again.

Version 1 never reached the dedicated hash gate and did not use a GPU or fetch
weights. Its failure is not a hash-ablation performance result. The matching
successful CPU receipt is still required before T4 submission. No Kaggle status
or live-log query has been made.

Kaggle accepted the corrected [CPU preflight, version 2](https://www.kaggle.com/code/pamcks/h3-dit-hash-ablation-cpu-20261029).
Retry source commit: `e1966aca5323d555cffc891de8bfff7812295987`;
capsule SHA256: `aded869fdade8e3d55aaca455d6a17345e2dc3431514a16e625548ed7ff11316`.
Its final artifacts were collected once: setup status `cpu_ready`, 207 regression
tests passed with no failures/errors/skips, and all 49 dedicated contracts passed.
Source hashes, capsule identity and downloaded artifact hashes match. Kitchen
`0.2.37`, TileLang `0.1.13` and the legacy attention APIs remain qualified on the
CPU host. No model weights were fetched and no GPU was used in that gate.

Kaggle accepted **version 1** of [H3 DiT Hash Ablation T4 20261029](https://www.kaggle.com/code/pamcks/h3-dit-hash-ablation-t4-20261029),
using the same CPU-qualified retry capsule. The prepared A/B items, three trials
per mode, persistent queue, P0/P1 settings and byte-exact output checks are
unchanged. The notebook requests two T4s and enforces physical SM75/cu128 checks
before fetching the trained PTD subset. The final artifacts now confirm device
execution and completion. No scheduling status or live logs have been queried. Receipts are under
`docs/data/h3_dit_hash_ablation_20261029/gpu_v1_*`, with CPU results under
`cpu_v2_final/` in the same directory.

## Physical T4 results

Both distinct devices identify as Tesla T4, SM75. Actual environment: native
Torch `2.11.0+cu128`, CUDA `12.8`, Kitchen `0.2.37`, TileLang `0.1.13`. Source
and capsule hashes match the CPU-qualified retry. The execution supervisor
exited with code 0 after 1,213.49 seconds, including downloads, one bank bake,
reader probes, rehearsal and six measured trials. The bake took 284.36 seconds
and is excluded from inference measurements.

| Measurement | Verification enabled | Trust bake |
|---|---:|---:|
| Trial 0, selection + denoiser RPCs | 112.639 s | 65.233 s |
| Trial 1 | 109.018 s | 41.903 s |
| Trial 2 | 113.179 s | 67.380 s |
| Median RPC time | **112.639 s** | **65.233 s** |
| Worker/service startup, median | 13.758 s | 6.814 s |
| Controller admission, median | 4.273 s | 0.028 s |
| AOT preparation after rehearsal, median | 1.653 s | 1.849 s |
| Whole fixture service lifetime, median | 138.836 s | 76.083 s |

The RPC median saves **47.406 seconds**. Every trusted trial is faster than
every verified trial, so the effect is practically substantial in this fixture.
The trusted arm nevertheless has a broad 41.903–67.380-second range. Three
samples per arm do not establish a formal significance level or explain that
variance. Whole-service lifetime includes startup/admission/preparation/shutdown;
it is not H3 generation latency. Medians of phases are not additive.

All six cases completed all eight hops with P0/P1 enabled and one persistent
queue. Video/audio hashes at every denoiser call and final FP32 sampler-state
hashes match the rehearsal exactly. Standalone per-case files match the aggregate
result, and both worker reports contain the expected 16 dispatch spans.

## Attribution: host hashing is on the critical path

Each verified worker performs **26 payload SHA256 calls over 17.415 GiB** and
**16 tensor-state hash calls over 2.926 GiB** across its service lifetime. Of
those payload calls, eight scan the monolithic hop auxiliary overlays: **11.505
GiB per rank**. Two base-auxiliary scans account for another 2.985 GiB; sixteen
owned block-shard scans account for the remainder. These counts include startup
and admission work as well as inference.

| Median worker hash attribution | Rank 0 | Rank 1 |
|---|---:|---:|
| Bulk hash thread CPU time, verified | 80.943 s | 78.122 s |
| Bulk hash span wall-time sum, verified | 102.781 s | 89.069 s |
| Overlay SHA256 thread CPU time, verified | 49.462 s | 44.919 s |
| Overlay SHA256 span wall-time sum, verified | 63.277 s | 47.203 s |
| Bulk hash calls, trusted | **0** | **0** |

The trusted arm still executes 20 metadata hash calls per worker, totaling
about 3.13 MiB; their measured thread CPU cost is approximately 0.016 seconds.
The large payload/tensor work disappears rather than being relabeled or moved
into timed GPU kernels. Numerical operators, bank bytes and compilation items
are unchanged, and the AOT medians show no compilation-related improvement.

Rank 1 illustrates the monolithic-overlay penalty particularly clearly: it
only consumes output-head tensors, but first verifies the entire shared overlay.
Its aggregate auxiliary-read span median drops from 47.247 s to 0.182 s when
bulk verification is bypassed. Rank 0 still needs the refiner payloads and
conversion/folding: its seven instrumented queued-refiner preparation spans sum
to median 85.202 s with verification and 56.884 s when trusted. Those nested
spans are attribution evidence, not independent latency terms.

Hash work competes with CPU preparation and cannot be fully hidden by P1 in
this experiment. The measured RPC reduction establishes critical-path impact.
Do not add hash span sums across ranks or subtract them from RPC latency:
prefetch, CPU work and GPU execution overlap. These counters do not measure
kernel-only time, GPU busy percentage or Nsight memory stalls. Remaining trusted
latency includes refiner preparation, loading, transfers, communication and
execution; this A/B does not isolate each remaining component.

## First consumption versus cached reload

The 24 sequential reader observations on the same T4 host confirm that the
gain is mainly first-use verification, not a faster tensor copy implementation.
All compared loaded weights are byte-exact.

| Median owned-shard load | Verified | Trusted |
|---|---:|---:|
| Rank 0, first consumption | 1.449 s | 0.163 s |
| Rank 1, first consumption | 1.270 s | 0.160 s |
| Rank 0, subsequent loads of same reader identity | 0.163 s | 0.165 s |
| Rank 1, subsequent loads | 0.162 s | 0.162 s |

Thus removing hashes from an already verified cached reader offers essentially
no benefit. PTD still pays first-use costs for each new hop-specific identity.
The six service trials intentionally reset these reader identities while reusing
the same bank and warmed binary cache. Physically cold storage and a second
complete trajectory in an already-running service were not measured.

Rank 0 peak allocated VRAM stays at 3.464 GiB in every trial, with median peak
reserved 3.820 GiB. Rank 1 median peak allocation remains 0.388 GiB, although
one trusted trial reaches 0.568 GiB. Rank 0 median process RSS rises from 4.643
to 4.850 GiB. This is a host-work optimization, not a demonstrated memory-saving
optimization. Long-sequence activation peaks were not tested here.

For an explicitly trusted immutable bake, `verify_dit_weights=False` now has
strong fixture-level performance evidence without changing numerical results.
There is no reason to remove the cached verification implementation: retain
the verified route for files that need consumption-time content checks. The
default remains `True` pending a separate policy decision. No full-pipeline
generation or additional GPU run was submitted during result collection.

## Final evidence collection

The first final-artifact transfer saved 49 files and then encountered an SSL EOF.
A bounded missing-file retry downloaded only the two remaining setup receipts
and the final kernel log. All 49 earlier files retain their hashes. All 52
remote artifacts are now complete, with matching source identity and successful
setup/execution supervisor receipts. This was artifact resumption, not a status
query, live-log poll or GPU resubmission.

Raw results: `docs/data/h3_dit_hash_ablation_20261029/gpu_v1_final/`.
The complete inventory is `gpu_v1_complete_download_manifest.json`; the initial
partial inventory and failure/attempt receipts are retained separately. Derived
per-trial CSV, attribution medians and source/hardware/output verification are
under `analysis/` in the same data directory.
