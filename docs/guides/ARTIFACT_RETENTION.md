# Diagnostic artifacts and normal inference

Normal ComfyUI inference imports no benchmark observer or profiler. Detailed
host/window event histories are disabled by default. Numeric cache/load/wait
counters remain available; runtime stage history is bounded to 16 entries.
`MINIMAX_H3_PROFILE_DETAILS=1` enables up to 4096 recent detail records per
metric history. It does not launch a profiler. Dropped history is never
interpreted as a complete historical trace.

`benchmarks/causal_pipeline_audit.py` defaults to `--capture timing`. CPU/GPU
profiling requires an explicit capture argument. `analyze_cpu_path.py` emits
metrics by default; `--timeline` explicitly requests its HTML export. The TE
ablation runner captures kernels only for the actual reference/TE stage of
one real full request. Timing and profiler passes are separate. Profiler
durations are not subtracted from individual stalls or used as speedup data.

Thirty completed physical trace JSON/HTML files were archived with streaming
gzip: **11.873 GiB → 1.157 GiB**, freeing **10.716 GiB**. Each archive was fully
decompressed and SHA256 checked before the unchanged original was removed.
Receipts are in `semantic_results/artifact_retention/`. No checkpoints,
reference tensors, results/verdicts, or compact readable CPU/GPU timelines
were removed. Three obsolete local stop/debug marker files were removed.
Historical benchmark scripts remain because source identities and previous
results depend on them; they are separate opt-in entry points.

Inspect eligible generated traces without changing files:

```powershell
python manage_audit_artifacts.py
```

Compress a known completed output root, preserving checksum-verified evidence:

```powershell
python manage_audit_artifacts.py --compress --prune-verified --root kaggle_causal_resume_output
```

The analysis loader reads raw JSON or its `.json.gz` archive transparently.
Resumable Kaggle downloads recognize the verified archive receipt rather than
downloading a second raw copy. For tools requiring an ordinary JSON/HTML file,
restore one archive with a checksum check:

```powershell
python manage_audit_artifacts.py --restore "kaggle_causal_resume_output/causal_audit_results/continuous/offline_p1_p2/kineto/batch_r0_kineto.json.gz"
```

No automatic age-based deletion touches evidence. Large full-request captures
must be explicitly requested, with a bounded request count, and archived only
after a job completes. Small correctness verdicts, benchmark manifests and
source bindings stay uncompressed. Generated tensor outputs are not saved by
the new TE qualification gate; it reuses the preserved Native references.

The 2026-10-15 TE block diagnostic preserved six SHA-verified compressed traces:
129.54 MB raw → 10.75 MB archived. Only compact per-block metrics, CSV and the
readable timeline are expanded locally. Three raw causal files downloaded for
the earlier P0 reanalysis were also verified and compressed, freeing another
60,485,437 bytes; their checksum receipt is in the same retention directory.
Idle-residency timing defaults to counters only and generates no Kineto or tensor dumps.
