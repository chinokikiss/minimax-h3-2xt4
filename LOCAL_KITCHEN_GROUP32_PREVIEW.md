# Local Kitchen-inspired Group32 preview — 2026-10-17

This is a completed local RTX 5060 Ti (SM120) screen, not a T4 result. Real captured H3 inputs were sliced to the current TP2 contracts. Both logical ranks ran sequentially on one GPU. No H3 generation, sparse policy change, NCCL or full-pipeline test ran.

Preparation fusion is the promising candidate. The larger query tiles / 256-thread warp variants regressed locally. Do not use this architecture-specific ranking to reject or select T4 kernels.

## Reproducible scope

Hardware: NVIDIA GeForce RTX 5060 Ti; driver 616.56; Torch 2.12.1+cu132, CUDA runtime 13.2, TileLang 0.1.13, Kitchen 0.2.33. Local nvcc is CUDA 13.3 with MSVC. These differ from queued T4 dependencies (Torch 2.11/cu130, TileLang 0.1.15, Kitchen 0.2.37, CUDA compiler 12.8).

TP2 shapes (BHND, batch 1): TE Q=(32,353,128), KV=(4,353,128), causal GQA8; DiT QKV=(28,6814,128) with V token stride 10752 matching packed QKV; Video Q=(32,1024 or 773,64), global KV=(32,1797,64). Captures: TE layer0, DiT layer25/call9, Video layer0/call0. Fixture hashes and actual strides are recorded in JSON. These are native-trajectory captures sliced for TP2, not newly generated TP2 trajectories.

The screen used five warm-up calls and ten individual CUDA-event / synchronized-wall samples per phase. API timings were repeated in reverse order without compilation. A separate rank0 confirmation used fifteen samples and CUDA Graph replay as a Python-dispatch ablation. First-call JIT timings were excluded from steady-state samples. Windows was not an exclusive benchmark environment; small timing differences and inter-round drift should not be overinterpreted.

## Short confirmation (logical rank 0 only)

Median CUDA-event API interval in ms. The range spans first and reversed repeat-round medians; graph replay removes per-operation Python dispatch and excludes graph capture. It is diagnostic, not an enabled TP2 production feature.

| Component | Current API | Fused API | Current graph | Fused graph | Graph reduction |
|---|---:|---:|---:|---:|---:|
| te | 0.318–0.366 | 0.256–0.276 | 0.107 | 0.095 | 12.0% |
| dit | 18.216–18.269 | 17.528–17.535 | 18.234 | 17.478 | 4.1% |
| video | 0.638–0.646 | 0.608–0.608 | 0.591 | 0.560 | 5.3% |

The TE eager-vs-graph gap supports a substantial Python/launch contribution. DiT remains dominated by GPU work: fused API 17.53 ms and graph 17.48 ms. This ablation is not Nsight evidence of a particular memory stall or instruction bottleneck. Video Kitchen confirmation is 0.383–0.392 ms API / 0.275 ms graph; fused Group32 is 0.608 / 0.560 ms, so it still has a substantial local gap to Kitchen.

## Full screen: both logical TP2 partitions

Phase timings are separately measured event intervals and need not add to API latency. They include applicable launch gaps; they are not pure GPU kernel durations.

| Component | Logical rank | Candidate | API first / repeat ms | Prep ms | Attention + output ms | Bitwise current |
|---|---:|---|---:|---:|---:|---|
| dit | 0 | current | 18.461 / 18.600 | 3.770 | 14.945 | True |
| dit | 0 | fused_prepare | 17.893 / 17.693 | 2.989 | 14.907 | True |
| dit | 0 | q64_row256 | 21.540 / 21.304 | 2.980 | 18.297 | True |
| dit | 0 | q64_square256 | 23.901 / 24.011 | 2.975 | 21.001 | False |
| dit | 1 | current | 21.811 / 19.426 | 3.994 | 17.052 | True |
| dit | 1 | fused_prepare | 18.149 / 18.424 | 3.235 | 16.455 | True |
| dit | 1 | q64_row256 | 21.348 / 24.735 | 3.042 | 18.641 | True |
| dit | 1 | q64_square256 | 30.050 / 25.591 | 3.405 | 24.747 | False |
| te | 0 | current | 0.328 / 0.334 | 0.292 | 0.097 | True |
| te | 0 | fused_prepare | 0.263 / 0.282 | 0.231 | 0.088 | True |
| te | 1 | current | 0.327 / 0.347 | 0.258 | 0.095 | True |
| te | 1 | fused_prepare | 0.326 / 0.269 | 0.204 | 0.087 | True |
| video | 0 | current | 0.641 / 0.636 | 0.280 | 0.396 | True |
| video | 0 | fused_prepare | 0.613 / 0.681 | 0.327 | 0.403 | True |
| video | 0 | kitchen | 0.422 / 0.381 | — | — | different algorithm |
| video | 0 | q128_qreuse256 | 0.697 / 0.717 | 0.280 | 0.488 | False |
| video | 0 | q128_row256 | 0.709 / 0.716 | 0.279 | 0.509 | False |
| video | 1 | current | 0.580 / 0.582 | 0.291 | 0.333 | True |
| video | 1 | fused_prepare | 0.724 / 0.688 | 0.349 | 0.329 | True |
| video | 1 | kitchen | 0.349 / 0.382 | — | — | different algorithm |
| video | 1 | q128_qreuse256 | 0.657 / 0.724 | 0.256 | 0.441 | False |
| video | 1 | q128_row256 | 0.656 / 0.660 | 0.255 | 0.442 | False |

## Numerical checks and limits

All 20 Group32 cases had exact QI/KI/QS/KS/V/count identity against the baseline at their own tile geometry and passed full-output rtol=0.001, atol=0.0002. Fused preparation alone was bitwise output-identical in all six component/partition cases. 14/20 complete Group32 outputs were bitwise identical; tile/warp changes can alter rounding. Kitchen is a separate numerical method, not part of this equivalence gate.

The optional dense-reference check samples 32 evenly spaced real queries per head using every key with FP32 math and TF32 disabled. It is a smoke check, not full-distribution or generation-semantic validation. Fused preparation leaves these errors unchanged. In Video rank0, median-head relative RMSE is 0.0934% for Group32 vs 0.1412% for Kitchen, but worst-head error is 0.6822% vs 0.6811%; this does not establish uniformly higher precision.

## Compatibility and representativeness

The production/submitted sources are untouched and all eight normalized submitted-source hashes still match. Isolated copies change only the package namespace, SM75 guards and target to SM120. Installed TileLang 0.1.13 generated an invalid cutlass::half_t → __half conversion in vectorized V padding. The final run repairs the generated pack_float16x4 helper to accept any 2-byte type, using the identical 16-bit shift/OR packing. Original and repaired CUDA sources are retained; vectorization and attention scheduling are preserved. The earlier scalar-vectorization-disabled diagnostic run is excluded from the final timings.

Representative for real tensor distribution, TP2 partition sizes/strides, operand contracts and local implementation screening. Not representative of T4 instruction mapping, occupancy, cache/memory behavior, relative kernel ranking, dual-GPU concurrency or NCCL/PCIe cost. The T4 job remains the authority for deployment decisions; its last status in this turn was QUEUED.

Next on T4: confirm fused preparation first; retain the small scheduling matrix to measure actual SM75 behavior. Keep existing sparse drop/scalar semantics and numerical contracts unchanged. No production switch is justified by this local preview alone.

## Artifacts

- Screen: `D:\attention_eval\operator_optimization_tp2_20261017\kitchen_inspired\local_preview_vectorized\results.json`
- Confirmation/graph ablation: `D:\attention_eval\operator_optimization_tp2_20261017\kitchen_inspired\local_preview_graph\results.json`
- Adaptations/CUDA sources: `D:\attention_eval\operator_optimization_tp2_20261017\kitchen_inspired\local_preview_vectorized`
- Runner: `C:\Users\Administrator\Documents\project\2xT4 minimax h3\benchmarks\local_kitchen_preview.py`
