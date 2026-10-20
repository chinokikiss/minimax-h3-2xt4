# Manual Kaggle Native timeout

The supplied archive proves that the harness killed Native at its 600-second
limit (`return_code=124`). Native sampling had not completed, so the study
produced no accepted Native/TP or P0/P1/P2 performance comparison.

| Area | Finding | Evidence |
|---|---|---|
| Hardware | Two physical Tesla T4s, Torch 2.11.0+cu130, 31.35 GiB host RAM | `hardware.json` |
| Setup | Completed in 910.662 seconds | `manifest.json` |
| Offline retrieval | All 200 files, 41.009 GiB, verified; 515.476 seconds; no conversion | `offline_restore.json` |
| Native frontend | VAE construction 6.712 s; reference encoding + TE 104.093 s | `native.log` |
| Native DiT | All 50 resident blocks loaded; sampler did not finish before timeout | `native.log` |
| Host headroom | 22.434 GiB available at the last 570-second heartbeat | `native.log.progress.json` |
| Telemetry | Only 132.803 s of samples, ending at TE completion; sampler data absent | `native_pcie.json` |

Evidence is extracted under `semantic_results/manual_native_timeout`. The
archive SHA256 and structured diagnosis are in `diagnosis.json`.

The previous successful physical Native reference took 550.587 seconds total.
A 600-second ceiling left less than 9% margin. I set this limit too tightly.
The 800-second TP case ceilings were also optimistic: each case includes two
requests, validation, worker startup and a separate profiled evaluation.
Dependency warnings did not abort either install. No OOM is reported in the
provided logs. The underlying reason Native exceeded its earlier timing is
still unresolved; the missing sampler telemetry cannot prove normal progress,
a compute slowdown, or a blocked call.

## Repair and resume

Paste the complete contents of `kaggle_manual_resume/resume_cell.py` into a new
cell in the **same live Kaggle session**. Do not rerun the original setup cell.
The helper requires the existing `/tmp` models/shards and pinned source tree.
If those are gone, it stops instead of downloading another checkpoint set.

The helper keeps all 50 layers, four sampling steps, 512x512/124 frames,
identical conditioning, seed, Native operator dispatch and numerical gates.
It reruns the failed Native case with a 900-second ceiling, then runs the four
TP configurations with 1,200 seconds each. These are limits, not duration
estimates. The aggregate GPU-case budget is 5,700 seconds, with offline Native
binding separately bounded at 1,200 seconds. Completed matching cases may be
reused on a later invocation; the first failure stops the study.

Telemetry is atomically saved every 30 seconds, including GPU utilization,
so a timeout cannot erase the entire active stage. Native forward entry/host
return messages and periodic Python stack snapshots identify progress and
blocked call sites. Host-return durations are explicitly not synchronized
GPU latencies. Stack snapshots have a diagnostic `Timeout` heading and do
not themselves terminate execution. No CUDA fence or model operator is added.

Frozen qualified source hashes are checked before resuming. The cell embeds
only diagnostic scripts and qualification metadata, with no HF token, signed
download links, model weights or instance submission. All existing numerical
qualifications remain current. A local NVML probe verified that telemetry is
saved **during an unfinished stage**, before forcibly terminating its monitor;
this is instrumentation qualification, not dual-T4 performance evidence.

Results and the small downloadable `summary_evidence.zip` are written to
`/kaggle/working/pipeline_audit_results_resume`. Separate profiler traces are
packaged in `profiler_traces.zip` when available. Original failed-run evidence
is preserved. Download the summary archive even if the resume fails.

The standalone full-audit builder also uses the revised budgets. A fresh full
submission needs a kernel timeout of at least 9,000 seconds to accommodate
setup, the GPU matrix, binding and analysis. The resume cell avoids setup.
