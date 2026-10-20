# Benchmark and diagnostic entry points

These modules are explicit developer tools, not runtime imports. The ComfyUI
node imports `patches`; it never imports this package. Numerical reference
loaders are isolated from the normal worker execution path.

Use the existing conda environment and pinned local ComfyUI. From the repository
root, the focused regression command is:

```powershell
$env:PYTHONPATH = "$PWD\benchmarks"
& C:\ProgramData\miniconda3\envs\comfyui\python.exe -m unittest `
  benchmarks.test_pipeline_review benchmarks.test_legacy_runtime `
  benchmarks.test_frozen_kaggle_setup benchmarks.test_video_query_fallback `
  benchmarks.test_h3_operator_enhancements benchmarks.test_h3_loading_stages `
  benchmarks.test_idle_window benchmarks.test_idle_lifecycle benchmarks.test_idle_identity `
  benchmarks.test_residency_policy -v
```

This suite checks ownership, cancellation, offline bake publication, dispatch,
frozen setup, diagnostic boundaries and launcher defaults. It does not run the
full H3 model. Avoid blanket test discovery: historical files named `test_*`
include top-level GPU experiments and require separate review.

The component scripts `test_h3_pipeline_wiring.py`,
`test_h3_operator_integration.py --cuda-preview`, `check_legacy_video_query.py`
and `qualify_h3_staged_vae.py` take `--output` paths. They are local component
checks, not SM75/NCCL timing or generation-quality evidence.

Profiling/capture tools such as `causal_trace.py`, `cpu_causal_trace.py`,
`capture_real_attention.py` and `profile_real_attention.py` must be invoked
explicitly. `MINIMAX_H3_PROFILE_DETAILS=1` enables bounded detail metrics only;
it does not start a profiler or dump tensors.

`te_compression_local.py` is a RAM-only lossless codec benchmark on real
production-loaded TP2 TE tensors. It uses a separate `--deps` directory with
`requirements-te-compression.txt`; it never installs packages into the active
runtime. `build_te_compression_codecs.ps1` compiles unchanged upstream ZigZag
and FSE library files for its C ABI. See the
[compression report](../docs/reports/TE_COMPRESSION_LOCAL_20261020.md) for
the results and the ZipNN INT8 blocker. No SSD or full-pipeline speedup is
implied by its decode throughput.

Historical builders/analysis scripts preserve earlier experiments. Their input
artifacts may be local-only. Do not submit an old notebook or run its setup as a
current deployment recipe. The frozen Kaggle setup requires a CPU-prepared Linux
wheelhouse and refuses version substitutions or unqualified platform additions.
