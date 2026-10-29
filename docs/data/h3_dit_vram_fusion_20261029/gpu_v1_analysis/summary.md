# DiT VRAM fusion operator results

Batch status: **partial_or_failed**. Physical two-T4 receipt: **True**.

Seeded long-shape operator/block fixture; no generation or full-model window qualification.

| Section / variant | CUDA ms, slower-rank median | Speedup | Worst rank allocated GiB | Allocated GiB saved |
|---|---:|---:|---:|---:|
| norm_gate/gate_baseline | 51.748 | 1.000× | 5.852 | 0.000 |
| norm_gate/gate_fused | 13.566 | 3.815× | 2.582 | 3.270 |
| norm_gate/norm_baseline | 53.603 | 1.000× | 4.358 | 0.000 |
| norm_gate/norm_fused | 5.635 | 9.512× | 2.582 | 1.777 |
| column/chunk_16k | 308.449 | 0.858× | 5.443 | 1.296 |
| column/chunk_4k | 294.614 | 0.898× | 4.786 | 1.952 |
| column/chunk_8k | 290.572 | 0.911× | 5.005 | 1.734 |
| column/prepared | 358.442 | 0.738× | 7.430 | -0.691 |
| column/production | 264.634 | 1.000× | 6.739 | 0.000 |
| row_epilogue/baseline | 68.045 | 1.000× | 6.444 | 0.000 |
| row_epilogue/fused | 13.451 | 5.059× | 2.582 | 3.862 |
| attention/baseline | 1613.754 | 1.000× | 3.775 | 0.000 |
| attention/split | 1614.074 | 1.000× | 3.381 | 0.395 |
| block: incomplete | — | — | — | — |

Numerical results, phase peaks, collective counts and trace coverage are retained in the JSON summary. No production default is promoted.

- Component peak savings overlap and must not be added
- CUDA time uses the slower rank for each aligned repeat, then median
- Sampled full-device usage is not the full-device peak
- Compilation/loading and error checks are outside warmed timing
- Missing CUDA trace coverage blocks attribution even if latency samples exist

Blockers: Column candidates failed byte qualification; timings do not authorize promotion; block incomplete; no speedup claim.
