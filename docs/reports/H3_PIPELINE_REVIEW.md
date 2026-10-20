# H3 pipeline review — 2026-10-20

The review covered the ComfyUI node, sampler adapters, persistent rank protocol,
TE/DiT construction, residency/cache ownership, VAE stages, operator dispatch,
frozen setup and publication boundaries. No dependency or Native ComfyUI source
was updated. No full H3 pipeline or Kaggle GPU experiment was run.

## Findings addressed

- Native TE/DiT factories, staged vision and startup state moved into
  `patches/comfy_tp_models.py`. Existing runtime import names are re-exported.
- The full-tensor Native reference loader moved into `reference_loaders.py`.
  The historical entry point remains a lazy compatibility shim; normal rank
  preparation uses the owned-range loader.
- Explicit `stats`/`idle_stats` snapshot assembly moved into
  `runtime_diagnostics.py`. Counters are bounded; profilers and tensor capture
  remain explicit tools under `benchmarks`, with no runtime imports.
- Persistent-worker audio decode now imports `audio_channel_decode.py` directly.
  The old tiled/temporal benchmark manager is no longer loaded by this path.
  Its historical function name re-exports the same function object.
- Service abort unregisters its bound atexit callback, preventing replaced,
  closed sessions from remaining reachable until interpreter exit.
- Node/API isolation is opt-in through `--isolated-ui`; normal launcher defaults
  no longer disable other custom/partner nodes. Dynamic VRAM remains disabled
  because TP2 owns its weight window.
- Removed an unreachable validation branch and corrected stale FP16 SUM comments.
  Operator math, INT32 defaults, sparse choices and attention selection remain.
- Node-to-policy construction now uses named fields, avoiding positional coupling.
- The freezer no longer writes a duplicate maintained setup implementation into
  `runtime`; exported helper copies remain hash-bound generated artifacts.
- The main README now describes the active pipeline and limitations. Its earlier
  historical material is preserved in `HISTORICAL_README.md`.
- Generated submissions/results, traces and private upload/download capabilities
  are kept local. Publication uses an explicit reviewed source list rather than
  adding the entire workspace.

## Compatibility intentionally retained

`KitchenRefinerAttention` is a historical import alias for Native SDPA. The hidden
`lora_concat_a` parameter still rejects the retired mode clearly. PP/SP/tiled
interfaces remain for older callers and are not selected by this loader. Both
input-packing adapters remain because existing qualification callers use them.
These are compatibility paths, not duplicate default execution paths.

The intentional Token Refiner SDPA change, offline LoRA option, staged VAE
loading, idle preloading, Group32/Kitchen policies and optional scaled FP16
transport remain. Historical benchmark sources/results were not rewritten.

## Verification and remaining limits

The focused suite passes 56 tests, including session collection, launcher node
availability, runtime/benchmark import isolation and read-only snapshots. Tiny
Native pipeline wiring, operator integration, trained 36-block video query and
trained staged Video/Audio VAE checks also pass locally. Query output is
byte-exact with Native using two logical ranks on one 5060 Ti; this is not
physical TP2/NCCL or T4 performance qualification.

The frozen conda stack still has a Linux platform dependency/ABI qualification
blocker. The historical TileLang 0.1.15 gate is not silently relaxed for the
snapshot's 0.1.13. Source setup has not been deployed as a new Kaggle job.

Detailed local logs and source hashes are under
`D:\attention_eval\h3_pipeline_review_20261020`. Existing ComfyUI startup emits
three UI-package recommendation warnings; packages are deliberately preserved.
