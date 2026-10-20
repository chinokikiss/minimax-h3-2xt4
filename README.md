# MiniMax-H3 on two NVIDIA T4 GPUs

Persistent TP2 workers for MiniMax-H3, with Native ComfyUI conditioning,
sampling and public VAE planning. Runtime code, numerical references, profiling
and historical experiments have separate entry points. Full-pipeline speedups
must not be inferred from operator measurements.

## Preserved runtime

- ComfyUI: local `ComfyUI` checkout at `d49e888586dd8ae012c0667b33466b815fee07f7`, with the validated attention interface.
- Local Python: `C:\ProgramData\miniconda3\envs\comfyui\python.exe`.
- Exact package snapshot: [runtime/local_conda.json](runtime/local_conda.json).
- Models: `D:\minimaxH3`, configured by `h3_legacy_model_paths.yaml`.
- No upstream code updates or implicit package substitutions during setup.

```powershell
.\launch_h3_legacy.cmd --preview --check
.\launch_h3_legacy.cmd --preview
```

The preview starts the selected ComfyUI on a one-GPU host; executing TP2 requires
two matching GPUs and NCCL. Node/API isolation is available with `--isolated-ui`
for smoke tests. Dynamic VRAM is disabled because workers own the weight window.
The selected checkout uses its own plugin and user directories.

**Kaggle status:** the Windows conda snapshot is preserved, but its Linux/T4
stack is not yet qualified. Linux-only CUDA/NCCL/Triton dependencies require an
explicit platform lock and ABI validation. The historical TileLang 0.1.15 gate
is retained; it does not qualify the snapshot's TileLang 0.1.13. Setup blocks on
these differences. See [runtime setup and limits](H3_LEGACY_RUNTIME.md).

## Operator defaults

| Component | Default | Optional choices |
|---|---|---|
| TE attention | TileLang Group32 dense | Native |
| DiT attention | Kitchen INT8 | TileLang dense, dropping, scalar pooling; Native |
| Token Refiner | Native SDPA | Native projections, norms and RoPE retained |
| Video VAE | Kitchen INT8, aligned query partitioning | Native decode fallback |
| Row projections | TileLang packed, native INT32 SUM | Native; approximate scaled FP16 SUM |
| SwiGLU | Fused padding | Disable fusion |
| LoRA B GEMM | Selective B for qualified FC1 geometry | Full B for other sites |
| Audio VAE | FP32 stereo channel parallel | Native; small rounding differences are expected |

LoRA loading defaults to online; offline baking is optional and assigns whole
layers to up to two GPU workers. Reference VAEs load encoders first and decoders
at output time. Idle residency optionally preloads encoders and TE weights.
See [operator integration](H3_OPERATOR_INTEGRATION.md),
[enhancements](H3_OPERATOR_ENHANCEMENTS.md), and [loading stages](H3_LOADING_STAGES.md).

## Code boundaries

- `custom_nodes/minimax_h3_tp2`: ComfyUI node schema and session selection.
- `patches/comfy_tp_runtime.py`: worker protocol, phase transitions and ownership.
- `patches/comfy_tp_models.py`: Native TE/DiT model construction.
- `patches/comfy_sampler.py`, `staged_vae.py`: Native sampler/VAE adapters.
- `patches/parallel_vae.py`, `audio_channel_decode.py`: active VAE rank execution.
- `patches/h3_operator_policy.py`, `tilelang_*.py`: operator dispatch and kernels.
- `patches/runtime_diagnostics.py`, `diagnostic_metrics.py`: explicit snapshots and bounded counters.
- `patches/reference_loaders.py`, `offline_lora_reference.py`: explicit numerical qualification references.
- `benchmarks/`: tests, profilers, captures and experiment analysis; never imported by runtime code.

Legacy PP/SP and tiled decoder interfaces remain for compatibility and historical
experiments. They are not selected by the TP2 loader. The production audio path
does not import the old tiled-decoder manager. Profiler/tensor-capture scripts
run explicitly; normal inference does not start them.

## Validation and experiment entry points

Use the frozen conda interpreter for local checks. The focused regression suite
is documented in [benchmarks/README.md](benchmarks/README.md). Tiny trained VAE
and operator checks are component previews, not full H3 generation or physical
T4/NCCL performance qualification.

The active builders are `build_h3_operator_integration_job.py` and
`build_ref2va_e2e_job.py`; use `--validate-only` to inspect generated code without
running or submitting it. New jobs require a frozen runtime dataset with a
CPU-prepared Linux wheelhouse. GPU jobs install verified wheels offline and do
not resolve/download dependencies. Older builders/notebooks must not be
submitted directly with their historical setup.

[Pipeline review](H3_PIPELINE_REVIEW.md) records cleanup, checks and remaining
blockers. [Provenance audit](H3_RUNTIME_PROVENANCE_AUDIT.md) distinguishes source
selection from changes to the user's original ComfyUI directory.
[Historical overview](HISTORICAL_README.md) retains earlier reports and caveats.
Generated datasets, traces, submissions and private upload capabilities stay
local and are excluded from Git; no model weights are published.
