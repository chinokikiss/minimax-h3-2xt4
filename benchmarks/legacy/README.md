# Historical H3 implementations

This directory contains retired PP/SP, multi-GPU Audio/Video VAE
prototypes, and benchmark-only collective/CUDA diagnostic implementations.
Production H3 TP2 code must not import these modules.

The import paths and source-manifest entries of their historical
benchmark consumers have been updated to `benchmarks.legacy`. They are
preserved for reproducibility, not runtime qualification. Some experiments
require their original pinned packages and physical GPUs.

PTD Bank v1 regression scripts are isolated in `ptd_v1/` and require
an old source commit because the current runtime supports Bank v2 only.
