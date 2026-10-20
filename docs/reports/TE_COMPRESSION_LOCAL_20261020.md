Local TE compression experiment, 2026-10-20
==========================================

**ZSTD is the first candidate to carry into a future loading experiment.**
Direct FSE is a useful alternative when compressed size matters more. These
are CPU codec results: they do not establish T4 loading latency, disk savings
on the critical path, or full TE speedup. Per the user's clarification, no
timed SSD/read-plus-decode experiment was run. The active H3 loader is unchanged.

**Input and method.** The production `rank_local_int8.load_rank_block` loader
prepared layers 0, 24 and 49, both TP2 ranks, from
`D:\minimaxH3\qwen3vl_32b_minimax_h3_int8_convrot.safetensors`. This preserves
the current row/column ownership, replicated scales, quantization metadata
validation, and Native dtype casts. The corpus contains 108 final module
tensors, 1,463,872,512 bytes (1.363 GiB), and is overwhelmingly INT8 weights.
Embeddings and the vision tower are outside this corpus; these ratios must
not be presented as a measurement of the entire 27.14 GB checkpoint.

The CPU is an AMD Ryzen 7 7800X3D, with 8 physical/logical CPUs exposed and
31.1 GiB RAM. Execution used the validated conda Python 3.11.15. All 94 frozen
runtime package versions still match `runtime/local_conda.json`. Experimental
packages live only under `D:\attention_eval\te_compression_20261020\deps`.

Each independently decodable frame stays within one tensor and TP2 rank.
Chunk sizes were 1, 4, 16 and 64 MiB; the last chunk may be smaller. Compression
uses one library thread. Decompression uses 1, 2, 4 or 6 outer workers, with
one library thread per job. Input frames are resident in RAM before timing;
each decode materializes its output, which is released in the worker. No disk
read, hashing, compilation, loading, pinning, H2D, tensor assembly or GPU compute
is included in the decode timer. Contexts/pools receive one full warmup pass.
The main 4 MiB cases have five repeated measurements; other chunk cases have
three or five. Results use median elapsed time and decimal GB/s of raw output.
All samples, including variation, are retained in `screen.json`.

**Measured results at 4 MiB.** Ratio is raw bytes divided by stored bytes;
stored bytes include codec frames and their JSON side index. Encode throughput
includes per-frame context creation and framing; decode throughput includes
output allocation. Blosc results must be interpreted alongside their ratios.

| Scheme | Ratio | Bytes saved | Encode, 1 worker GB/s | Decode, 1 worker GB/s | Decode, 2 workers GB/s | Decode, 4 workers GB/s | Decode, 6 workers GB/s |
|---|---:|---:|---:|---:|---:|---:|---:|
| ZSTD level 1 | 1.1240× | 11.03% | 0.914 | 1.057 | 2.068 | 3.989 | 5.648 |
| ZSTD level 3 | 1.1240× | 11.03% | 0.867 | 1.107 | 2.130 | 3.817 | 5.400 |
| Blosc2 + BitShuffle + ZSTD level 1 | 1.0072× | 0.71% | 1.342 | 1.799 | 3.614 | 6.571 | 9.497 |
| Blosc2 + BitShuffle + ZSTD level 3 | 1.0001× | 0.015% | 1.848 | 7.593 | 11.849 | 16.122 | 15.643 |
| ZipNN INT8 | Blocked | — | — | — | — | — | — |
| OpenZL ZigZag + FSE | 1.1317× | 11.64% | 0.455 | 0.601 | 1.223 | 2.459 | 3.577 |
| Direct FSE, without ZigZag (ablation) | 1.1317× | 11.64% | 0.604 | 0.884 | 1.716 | 3.427 | 5.072 |

**Chunk size results, six decode workers.** Different methods prefer different
chunk sizes; 64 MiB is not an improvement here. The tiny ratio changes are
consistent with amortizing frame/index overhead, rather than a new source of
compressibility.

| Chunk | ZSTD-1 ratio / GB/s | ZigZag + FSE ratio / GB/s | Direct FSE ratio / GB/s |
|---|---:|---:|---:|
| 1 MiB | 1.1238× / 4.943 | 1.1314× / 3.567 | 1.1313× / 4.655 |
| 4 MiB | 1.1240× / 5.648 | 1.1317× / 3.577 | 1.1317× / 5.072 |
| 16 MiB | 1.1241× / 5.849 | 1.1318× / 3.656 | 1.1318× / 4.810 |
| 64 MiB | 1.1241× / 5.419 | 1.1318× / 3.425 | 1.1319× / 4.507 |

**Why BitShuffle disappoints on these INT8 weights.** A controlled 16 MiB
sample from layer 24, rank 0, `gate_proj.weight`, has empirical byte entropy
7.0979 bits/value, but its eight individual bit entropies sum to 7.9998 bits.
This supports the explanation that a substantial part of the exploitable
structure is correlation between bits within each signed byte. Separating the
bit planes makes the streams harder for this backend to compress. This is an
inference about this sample, not a general claim that BitShuffle is unsuitable
for all INT8 or floating-point weights.

On the same sample, Blosc ZSTD-1 without a filter achieves 1.1141×; adding
BitShuffle gives 1.0019×. Prepending the upstream ZigZag transform before
BitShuffle improves it to 1.0621×, still below the unfiltered result. For the
full tested corpus, 99.06% of raw bytes in Blosc3 frames have the documented
`BLOSC_MEMCPYED` flag. Its headline decode throughput largely measures stored
payload copies. Blosc1 has no whole-frame memcpy flags, but still saves only
0.71%. See `entropy_probe.json`, `blosc_mapping_probe.json`, and
`blosc_fallback.json` for the evidence.

**What ZigZag contributes.** ZigZag permutes the 256 INT8 symbols; it cannot
reduce their order-0 entropy. Direct FSE already models those byte frequencies.
The two FSE variants differ by only 0.0022 percentage points of saved bytes at
4 MiB, while direct FSE decodes about 42% faster with six workers. A separate
single-thread, preallocated-buffer probe on a real 16 MiB frame measured median
18.17 ms for FSE decoding and 7.45 ms for the inverse ZigZag pass (two warmups,
ten repetitions). This demonstrates a material transform cost in this build;
it does not assert that every implementation of ZigZag has the same cost.

**Existing libraries and the ZipNN blocker.** No compression or signed-mapping
algorithm was implemented for this experiment. Python adapters call:

- [python-zstandard 0.25.0](https://pypi.org/project/zstandard/0.25.0/), using its native ZSTD backend.
- [Python-Blosc2 4.14.1](https://pypi.org/project/blosc2/4.14.1/), with the native BitShuffle filter and ZSTD codec; GIL release enabled.
- [OpenZL v0.3.0](https://github.com/facebook/openzl/tree/32246b48faee46807f84183dac4db479089f5445/src/openzl/codecs/zigzag), unmodified upstream INT8 ZigZag C kernels.
- [FiniteStateEntropy](https://github.com/Cyan4973/FiniteStateEntropy), version 0.9.0 sources bundled in the pinned ZipNN 0.5.4 sdist. FSE is an ANS implementation. Sources were compiled with MSVC `/O2 /arch:AVX2`; `ctypes.CDLL` exposes the existing C ABI and releases the GIL. The adapter adds only mode/length framing and handles the library's documented raw/RLE return cases.

[ZipNN 0.5.4](https://pypi.org/project/zipnn/0.5.4/) has no Windows wheel in
its published artifacts. More decisively, its actual compression dispatch
does not accept INT8: the enum exists, but the non-floating branch in
`compress_torch_numpy_byte` admits only UINT32, and INT8 reaches a `ValueError`.
Its native core also includes POSIX `pthread.h`. Therefore no ZipNN INT8 speed
or ratio is reported. Its published floating-point numbers and generic-byte
workarounds are not substitutes for this requested INT8 path. Source and PyPI
metadata are retained for inspection. The descriptive integer discussion in
the repository is not evidence that this release implements that path.

The complete OpenZL Python extension build encountered an MSVC C11 static
initializer error. That is recorded in `openzl_build*.log`; the experiment
instead successfully built and validated the unchanged standalone ZigZag
kernels together with FSE. This does not qualify the full OpenZL Python API.

**Scheduling candidate for later integration.** Start with independently
decodable 4 MiB ZSTD-1 frames, with 16 MiB as the bulk-throughput option. Locally,
16 MiB gains only 3.6% decode throughput over 4 MiB with six workers; 4 MiB offers
finer scheduling and smaller individual outputs. Keep a process-wide worker
budget shared by both TP2 ranks, instead of giving each rank a full CPU-sized
pool, and keep internal codec threading at one. Prioritize both ranks of the
foreground layer, then the next layer, while preserving the existing shared
host budget and residency ownership. These are scheduling proposals, not a
measured TE-latency improvement or an implemented loader change. Worker counts
must be retested on the actual T4 host CPU. Direct FSE at 4 MiB is the secondary
candidate; the extra ~0.60 percentage points of saved bytes are measured, but
their deployment value is not yet established.

**Validation and limits.** All 15 compressed configurations passed byte-exact
round trips for every tested chunk. All 108 corpus tensor SHA256 values were
checked against the production-loaded states. All eight-bit symbols, constant
data, and incompressible data also exercised the FSE raw/RLE paths. The source
checkpoint, ComfyUI, operators and default dependencies were not modified.
No full H3 pipeline, GPU workload, Kaggle submission, semantic comparison, or
T4 SSD extrapolation was performed.

Windows CPU accounting failed a separate calibration: a one-second busy loop
reported much less than one CPU-second. Raw `cpu_ms` observations remain in the
artifact, but no utilization/effective-core conclusion is drawn from them.
Reported decode scaling uses wall-clock time and configured worker counts.
These tests materialize individual frames; a future loader still needs to
measure assembly into contiguous tensor destinations or use supported
destination-buffer library APIs. The compressed input corpus is RAM resident
during this benchmark; it is not a measurement of bounded streaming RSS.

**Reproduction and evidence.** Full local artifacts (~25.5 GiB, including raw
and packed variants) are retained at
`D:\attention_eval\te_compression_20261020`. `screen.json` has all measurements;
`summary.json`, `strata.json`, `corpus.json`, `provenance.json`, install reports,
source archives and native build logs preserve the inputs and environment.
The frozen H3 runtime requirements remain separate from
`benchmarks/requirements-te-compression.txt`. Use a fresh experiment directory
to repeat preparation; existing corpus data is protected against replacement.

```powershell
$python='C:\ProgramData\miniconda3\envs\comfyui\python.exe'
$experiment='D:\attention_eval\te_compression_20261020'
& $python benchmarks/te_compression_local.py screen `
  --output $experiment --deps "$experiment\deps" `
  --chunks 4 --workers 1 2 4 6 --repeats 5
```

Completed cases are retained/skipped on replay. For an independent repeat,
use a fresh directory, install the exact benchmark-only requirements with
`--target ... --no-deps`, obtain the pinned upstream sources recorded in
`provenance.json`, run `benchmarks/build_te_compression_codecs.ps1`, then run
the `prepare` stage followed by `screen`. An exclusive benchmark lock prevents
overlapping runs or concurrent result writers. Historical sources and results
are retained; nothing is published or installed into the active runtime.
