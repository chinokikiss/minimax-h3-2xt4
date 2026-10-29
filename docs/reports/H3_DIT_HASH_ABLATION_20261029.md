# DiT trusted-bake hash ablation — 2026-10-29

Status: implementation and local CPU qualification complete; real two-T4 timings pending. No speedup is claimed yet.

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

Local evidence: `D:\attention_eval\h3_dit_hash_ablation_20261029\`. Final T4 results and the default-policy decision remain pending.
