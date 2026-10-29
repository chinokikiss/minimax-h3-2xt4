# PTD cleanup: original overlays; P1 remains optional

The FP32 refiner sidecar and P2 auxiliary Split implementation have been removed
from active source at the user's request. P1 next-hop CPU refiner prefetch
remains available and **off by default**, including newly generated E2E plans.
The graph node and `PTDOptions` retain the same false default. Selecting P1 still
requires the future-use cache and queue handoff.

## Removed code

- Sidecar writer/reader, offline layout publication, tensor checksum path and
  `runtime_layout` / `baked_refiner` switches.
- Split auxiliary units, reader, offline repack and `split_auxiliary` switch.
- Dedicated Split analyzer/tests, sidecar-only tests, profiling hooks and
  sidecar/Split benchmark submission/case selection.

Both source-delivery builders explicitly remove retired files inherited from
historical capsules. Native ComfyUI, attention operators, dependency pins,
FP16 tensor-flow arithmetic and ordinary INT8 baking remain unchanged. Existing
monolithic banks retain their original reader, including banks with unused
historical sidecar metadata. Split banks are rejected with a monolithic-layout
error; use the archived runtime or bake a monolithic bank.

## Verification and the P1 question

The original auxiliary overlay is verified **once per hop reader/rank, on first
tensor consumption**. Metadata admission and construction no longer trigger an
eager full-overlay scan. Concurrent first readers share the lock and checksum;
subsequent reads retain file-signature guards. No per-tensor full hash or Native
forward comparison was added. Ordinary bake provenance and Native merge checks
are retained.

Hashing is a substantial measured cost, but it does not explain every P1 stall.
The [physical scheduling report](H3_PTD_CPU_SCHEDULING_T4_RESULTS_20261029.md)
records 66.15 seconds of nested sidecar checksum time inside 150.82 seconds of
sidecar read/validation spans. Those spans also include I/O, allocation and
other checks; they are not independent critical-path sums. P1 without a sidecar
has scheduling/ready-time variation whose exact cause was not isolated. Removing
the sidecar removes its extra read/checksum stream, rather than demonstrating
that P1 becomes stall-free.

Numerical equivalence is exercised in controlled tests and the existing bake
validation. Inference retains one-time file integrity checking; checks are not
silently disabled after an experiment passes.

## Records and running experiment

Historical reports, raw results, receipts and immutable submitted capsules are
preserved. Reproducing the removed paths requires their recorded source commit,
not the current benchmark launcher.

The already accepted [full PTD E2E version 1](https://www.kaggle.com/code/pamcks/h3-ptd-long-e2e-t4-20261029)
uses the prior immutable capsule and an explicitly enabled P1 experiment option.
This cleanup does not mutate, stop, query or resubmit it. No additional Kaggle
GPU time or API interaction is used for the cleanup.

## Validation

Local CPU tests use the normal conda comfyui interpreter, frozen Native source
and hidden GPUs. Coverage includes lazy verification and checksum reuse,
concurrent consumption, corruption with unchanged size/mtime, mutation after
verification, receipt tampering, disabled-role fallback, rejection of retired
layouts, source-delivery pruning, Native refiner equivalence, full Native sampler
integration and P1 default/explicit-enable behavior. Final counts and source
preservation evidence are recorded under
[`docs/data/h3_ptd_cleanup_20261029`](../data/h3_ptd_cleanup_20261029/).

This is CPU functional validation, not a new T4 performance measurement.
