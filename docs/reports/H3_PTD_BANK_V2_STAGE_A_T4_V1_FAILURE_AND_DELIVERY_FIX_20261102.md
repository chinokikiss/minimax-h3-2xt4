# H3 PTD Bank v2 Stage A — T4 v1 Failure and Delivery Fix (2026-11-02)

## Final v1 result

Kaggle kernel `pamcks/h3-ptd-bank-v2-stage-a-two-block-t4-20261102`, version 1, finished with `KernelWorkerStatus.ERROR`.

The failure occurred before any Stage A phase started. `result.json` contains an empty `stages` object and reports:

```text
ImportError: cannot import name 'check_state_schema' from 'patches.rank_local_int8'
```

The worker exited with code 1 after approximately 2.274 seconds. No trained PTD range download, shared/hop Bake, Bank assembly, dual-T4 runtime, eight-hop denoise, backend dispatch, VRAM sampling or PTD numerical measurement was executed. This is a delivery/package failure, not a PTD algorithm or T4 runtime failure.

Final evidence is archived under:

`docs/data/h3_ptd_v2_stage_a_t4_20261102/gpu_final_v1/`

## Root cause

The Stage A builder overlaid the current `patches/tp2_bundle.py` onto the frozen Kaggle source, but retained the frozen `patches/rank_local_int8.py`. The current bundle reader imports `check_state_schema`, while the frozen dependency did not expose that symbol.

The local worktree tests did not reproduce the failure because they imported a mutually consistent current source tree. The defect existed only in the frozen-baseline-plus-selective-overlay delivery composition.

The same risk also existed in `build_ptd_v2_bake_job.py`.

## Fix

Both PTD v2 Kaggle builders now derive their payload from the immutable frozen baseline commit and recursively close the overlay over every changed internal Python dependency using `benchmarks/source_overlay_closure.py`.

The manually reviewed PTD entry files remain explicit seeds. Non-Python PTD files such as `patches/ptd/reference_sources.json` remain explicit seeds as well.

The builders fail closed if `patches/rank_local_int8.py` is absent from the derived overlay.

A fresh-process source import gate is embedded in each generated notebook after overlay extraction and before the actual GPU worker is launched. The gate imports the critical PTD/bundle modules in a new Python interpreter, so modules cached by the frozen notebook setup cannot hide API skew.

Delivery metadata is now versioned as:

- Stage A: `h3-ptd-v2-stage-a-delivery-v2`
- Full Hop0 Bake: `h3-ptd-v2-hop0-kaggle-delivery-v2`

## Local verification

The corrected Stage A delivery contains 103 source files, including 61 recursively discovered changed dependencies. The corrected full-Hop0 Bake delivery contains 87 source files, including 45 recursively discovered changed dependencies. Both include current `patches/rank_local_int8.py`.

Generated notebooks parse successfully and contain their corresponding fresh-process import gates.

With the pinned local ComfyUI checkout on `PYTHONPATH`, the PTD regression suite completed:

```text
Ran 84 tests
OK
```

This includes the new `benchmarks.test_ptd_v2_delivery_closure` coverage for both builders.

## Status

The source-delivery defect is locally fixed and regression-covered. Additional exact-delivery qualification reconstructs commit `d7c7528515d2a7961de5e6f961232d81d21cdc8b` into an isolated directory, applies the actual tarball embedded in each generated notebook, checks every source SHA256 against its binding, and launches a new local `comfyui` Python interpreter. **Both Stage A (103 files) and full Hop0 Bake (87 files) pass cold imports**, including `check_state_schema`.

A separate end-to-end delivery regression (`benchmarks.test_ptd_v2_delivery`) also passes. The selected PTD and delivery suite completed **46/46** tests, including both rebuilt-notebook cold imports. Kaggle metadata `id` and title-derived slug are now equal: the Stage A candidate points at the existing provider-returned kernel slug, and the Hop0 builder points at its own requested slug. Both candidates are private, have no attached Kaggle datasets or kernels, and are preserved locally:

- `docs/data/h3_ptd_v2_stage_a_t4_20261102/gpu_job_v3_local/` (notebook SHA256 `bcf7d9766c5500bc4413c873c2e660f1ece08472a8fc8a8354dfa97b54f777ab`)
- `docs/data/h3_ptd_v2_hop0_t4_20261102/gpu_job_v2_local/` (notebook SHA256 `9984104c00479324c114aa960e78d24e01458f69d0be88b0b37b7bfb80709a90`)

**Update — 2026-11-02 corrected submission:** The user explicitly authorized one resubmission of the lightweight Stage A candidate. Kaggle returned `Kernel version 2 successfully pushed` with process exit code 0. The provider link is `https://www.kaggle.com/pamcks/h3-ptd-bank-v2-stage-a-two-block-t4-20261102`. The immutable v2 submission receipt is at `docs/data/h3_ptd_v2_stage_a_t4_20261102/gpu_submission_v2.json` (notebook SHA256 `bcf7d9766c5500bc4413c873c2e660f1ece08472a8fc8a8354dfa97b54f777ab`; overlay SHA256 `71a1e11129d7d459aa18da9a31f13b4d02d971868cd3676ecb218888212c4fe4`). No provider execution status or final GPU artifact has been collected for v2, and no automatic polling/retry is authorized. Physical dual-T4 PTD correctness, timing, numerical output, Bake and memory remain unqualified.
