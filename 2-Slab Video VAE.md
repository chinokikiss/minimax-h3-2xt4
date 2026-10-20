Add one new Video VAE benchmark experiment to the existing `minimax-h3-2xt4` benchmark suite:

## New experiment: 2-Slab Spatial Decode

Implement a **2-way spatial slab decode** for the Video VAE.

For a latent tensor `[B, C, T, H, W]`:

- split along the **width dimension** into two large slabs;
- add configurable overlap/halo between the two slabs;
- send the left slab to GPU 0 and the right slab to GPU 1;
- decode both concurrently with `_decode_pixels`;
- transfer one decoded slab back if necessary;
- blend/cross-fade only the overlapping output region;
- reconstruct the full output.

Start with:

- width split: 50/50;
- overlap: 64 px output-space;
- also test 128 px overlap if practical;
- preserve the full temporal dimension;
- do not use NCCL unless actually necessary.

Benchmark at least:

- `768x768`
- `768x1344`

You may reuse all existing benchmark results for:

- 1-GPU normal;
- 256px tiled;
- 384px tiled;
- 2-GPU tiled.

Do not rerun those unless needed for validation.

For the new `2-slab` method measure:

- end-to-end latency;
- speedup vs 1-GPU normal;
- speedup vs current best 2-GPU tiled result;
- GPU 0 / GPU 1 utilization;
- peak VRAM on both GPUs;
- PCIe transfer time;
- stitching/blending time;
- MAE;
- relative L2;
- cosine similarity against full non-tiled decode.

Also report the exact latent slab shapes used.

Verify that overlap coordinates and final output dimensions are correct and that there are no visible or numerical discontinuities caused by incorrect cropping/stitching.

If appropriate, also test a simple **halo + center-crop / narrow cross-fade** variant instead of blending the entire overlap.

The goal is to determine whether:

`2 large slabs + one boundary`

is faster and/or higher quality than:

`many 384px tiles + many boundaries`.

Do not redesign unrelated benchmarks. Keep the change minimal, reuse existing benchmark infrastructure and existing saved results, then update the benchmark report with the new comparison and conclusion.