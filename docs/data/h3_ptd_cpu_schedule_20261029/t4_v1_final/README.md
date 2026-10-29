# Final physical T4 CPU-scheduling ablation

All six eight-hop cases completed on two distinct Tesla T4 GPUs. Original Native source/core pins are preserved. `summary.json` independently checks source binding, per-hop/final hash equality, stable window identities, complete worker spans, no duplicate CPU preparation and fully drained host ownership. `ptd_result.json.gz` is a lossless compressed original; its uncompressed SHA256 is recorded in the summary. Raw downloads remain on D:.

Original setup, AOT, worker reports and final supervision are retained. All files are public fixture metadata; no trained weight payload or private activation is included. The user-authorized third artifact attempt succeeded after two SSL-EOF failures. No status/list/live-log polling or resubmission occurred. Earlier failed transfers remain separately preserved.

Timings are selection-plus-denoiser RPC sums, excluding setup, bake, AOT, diagnostics and teardown. The two-layer 24-token fixture cannot establish full-model speedup or generation quality. Baseline/prefetch have two repetitions; P0-only/sidecar have one. Host-budget reservation is not RSS. CUDA and nested CPU spans overlap and do not provide Nsight attribution.
