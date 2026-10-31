# H3 Sol All-INT8 T4 Benchmark Report — 2026-10-30

## Scope

This report evaluates the Star7-derived Sol All-INT8 SM75 attention path against
`comfy-kitchen==0.2.37` dense INT8 attention on two NVIDIA Tesla T4 GPUs.

The Sol implementation is isolated under `experiments/sol_all_int8/`. Production
attention dispatch is unchanged.

The benchmark measures complete raw-input FP16 BHND calls, including Sol routing,
centroid construction, Q/K/V quantization, LUT preparation, output allocation,
and attention execution. The baseline is the normal complete Kitchen
`int8_attention` call.

Geometry:

- Batch: 1
- Heads: 28
- Head dimension: 128
- Dtype: FP16 input/output
- Sol threshold: `tau=1.0`
- GPUs: 2 × NVIDIA Tesla T4
- Torch: 2.11.0+cu128
- CUDA runtime: 12.8
- NVCC: 12.8.93

## Porting correction

Version 1 completed lengths 512 through 4096, then failed at sequence length 6814
with:

```
ValueError: SM75 Sol Q scale must contain four 16-row scales per Q64 block
```

The failure was in the adapter, not the CUDA kernel. The initial port omitted the
upstream ragged-length Q-scale padding:

```python
required_q_scales = query_blocks * 4
if q_scale.shape[-1] < required_q_scales:
    q_scale = torch.nn.functional.pad(
        q_scale,
        (0, required_q_scales - q_scale.shape[-1]),
        value=1.0,
    )
```

Length 6814 requires 426 generated Q-scale slots but 428 Q64-layout slots; length
51220 similarly requires two padding slots. The corrected version 2 restored the
upstream behavior and completed all remaining long-sequence cases.

## Results

The table combines the valid 512–4096 results from version 1 with the corrected
6814–51220 results from version 2. Reported values are the median across the two
T4 ranks after each rank's two A/B ordering rounds.

| Sequence | CK dense INT8 | Sol All-INT8 | CK throughput | Sol throughput | Sol / CK | Route density |
|---:|---:|---:|---:|---:|---:|---:|
| 512 | 0.665 ms | 0.904 ms | 770k tok/s | 566k tok/s | 0.736× | 45.15% |
| 1,024 | 1.453 ms | 1.383 ms | 705k tok/s | 741k tok/s | 1.051× | 30.54% |
| 2,048 | 3.059 ms | 2.289 ms | 670k tok/s | 895k tok/s | 1.347× | 22.76% |
| 4,096 | 6.773 ms | 3.786 ms | 605k tok/s | 1.082M tok/s | 1.800× | 19.22% |
| 6,814 | 19.830 ms | 9.939 ms | 344k tok/s | 686k tok/s | 1.995× | 17.66% |
| 8,192 | 24.842 ms | 9.735 ms | 330k tok/s | 842k tok/s | 2.552× | 17.29% |
| 12,288 | 54.468 ms | 18.908 ms | 226k tok/s | 650k tok/s | 2.881× | 16.59% |
| 16,384 | 93.982 ms | 31.819 ms | 174k tok/s | 515k tok/s | 2.954× | 16.27% |
| 32,768 | 370.359 ms | 141.142 ms | 88.5k tok/s | 232k tok/s | 2.628× | 15.78% |
| 51,220 | 903.924 ms | 367.963 ms | 56.7k tok/s | 139k tok/s | 2.459× | 15.58% |

### Per-rank stability at H3-relevant lengths

At 6,814 tokens:

- Rank 0: 1.860×
- Rank 1: 2.130×

At 51,220 tokens:

- Rank 0: 2.535×
- Rank 1: 2.384×

The gain therefore survives across both physical T4s and is not a single-device
outlier.

## Interpretation

The crossover occurs around 1K tokens. Sol is slower at 512 because routing,
centroid, and quantization overhead dominates. At 1K it is effectively parity.
At 2K the gain becomes material, and from 4K onward the sparse path clearly wins.

The best region in this synthetic matrix is 8K–16K, where Sol reaches roughly
2.55–2.95× the complete-call throughput of Kitchen dense INT8.

The gain falls moderately at 32K–51K but remains large: 2.63× at 32K and 2.46×
at 51,220. This is still highly relevant for H3 long-sequence DiT attention.

Route density falls from 45.15% at 512 to 15.58% at 51,220. The performance
curve is therefore consistent with the expected increasing advantage of sparse
routing over quadratic dense attention as sequence length grows.

The benchmark's reported temporary allocation peaks are equal between the two
paths at the tested lengths. This experiment should therefore be treated as
throughput evidence, not as evidence for a VRAM reduction.

## Production recommendation

Do **not** promote Sol All-INT8 to the unconditional production default yet.

Recommended next step:

1. Keep the current implementation opt-in.
2. Add a length gate only for a real-H3 qualification branch:
   - below 2K: Kitchen dense INT8;
   - 2K–4K: optional Sol evaluation region;
   - 4K and above: primary Sol candidate.
3. Run a real H3 DiT A/B using identical latent trajectory inputs and compare:
   - output error against CK dense INT8;
   - per-block / per-step drift;
   - final video quality;
   - temporal consistency;
   - reference adherence.
4. If quality is acceptable, promote a conservative production dispatch threshold
   at approximately 4K tokens rather than using Sol universally.

For the two especially relevant lengths already measured:

- 6,814 tokens: ~1.995× faster complete-call latency.
- 51,220 tokens: ~2.459× faster complete-call latency.

Performance alone therefore supports continuing toward production qualification.

## Evidence

- Version 1: `pamcks/h3-sol-all-int8-t4-20261030/1`
  - valid completed cases: 512, 1024, 2048, 4096
  - failed at 6814 due to the adapter Q-scale padding omission
- Version 2: `pamcks/h3-sol-all-int8-t4-20261030/2`
  - corrected ragged Q-scale padding
  - completed: 6814, 8192, 12288, 16384, 32768, 51220

Version 2 native SM75 artifact:

- NVCC: CUDA 12.8.93
- Artifact SHA256:
  `b6c2a3edb855517531a0af32cf5fb5d2f797a633b300bf756b2072ecad4c8886`
- Artifact size: 1,790,296 bytes

No production dispatch was changed and nothing was pushed.
