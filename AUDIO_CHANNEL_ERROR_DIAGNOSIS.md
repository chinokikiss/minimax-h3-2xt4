# Channel Parallel audio VAE: source of the small error

The trained decoder's stereo channels are mathematically independent. The
observed discrepancy is reproduced by changing its convolution batch shape,
even when both mono decodes run sequentially on the same GPU. The first local
divergence is **`decoder.conv_pre`**, after exactly matching latent normalization
and `dec_in_proj`. The local kernel trace identifies a concrete dispatch change:
cuDNN FFT convolution uses complex GEMM for batch 2 and complex GEMV for batch 1.
Those operations use different FP32 accumulation paths.

This establishes the mechanism on the local 5060 Ti and reproduces the class of
error seen in the physical T4 results. The historical T4 run did not capture
per-operator audio traces, so its exact first divergent operator/kernel remains
unverified on T4. No new cluster instance or performance sweep was started.

## Evidence and inputs

Downloaded only the completed T4 audit's round-zero audio latents and Native /
Channel Parallel waveforms for 5, 10 and 20 seconds. All input byte hashes match
the original results. The checkpoint is the same 605,254,808-byte FP32 file used
by the full pipeline, SHA256
`8e505d95dd1561d47abd43d4238fd40d9bb1ae9e147ed0a4cba778d76ae4db48`.
Models load strictly; there is no INT8 quantization in this decoder.

The local reproduction uses torch 2.12.1+cu132 and cuDNN version integer 92000 on
one RTX 5060 Ti. TF32 is disabled, matching the T4's FP32 capability; convolution
benchmarking and deterministic selection are initially disabled. These are
numerical diagnostics, not dual-T4 timing results.

| Input | Recomputed saved T4 CP / Native relative L2 | Local sequential mono / Native relative L2 | Local same-batch dummy control |
|---|---:|---:|---|
| 5 s | 1.94788×10⁻⁵ | 1.27990×10⁻⁵ | Bit-exact |
| 10 s | 1.32745×10⁻⁵ | 1.23711×10⁻⁵ | Bit-exact |
| 20 s | 1.37809×10⁻⁵ | 1.20486×10⁻⁵ | Bit-exact |

The saved T4 5-second waveform has maximum absolute error 1.58817×10⁻⁴, consistent
with the earlier report. These results still fail the established 1×10⁻⁵ L2 gate.
Small error magnitude does not by itself establish perceptual acceptability.

An initial local reproduction regenerated seeded inputs, whose hashes differed
from Linux's historical tensors. It is retained as a separate exploratory run;
all results above and the convolution traces use the subsequently downloaded,
hash-verified original inputs.

## First divergence and accumulation

Native `decode` reshapes `[B,32,2,T]` into `[2B,32,T]`, decoding stereo as a
batch. Channel Parallel calls `decode` twice with `[B,32,1,T]`, so each mono
decoder sees `[B,32,T]`. There are no decoder operations that mix batch rows,
and the normalization constants and trained weights are unchanged.

On the saved 5-second input (B=1, T=200), the left channel gives:

| Stage | Output relative L2 / corresponding Native channel |
|---|---:|
| `dec_in_proj` | 0, bit-exact |
| `decoder.conv_pre` | 1.08062×10⁻⁶ |
| First residual block | 1.10192×10⁻⁶ |
| End of upsampling stage 3 (`resblocks.11`) | 2.01675×10⁻⁵ |
| `decoder.activation_post` | 1.19653×10⁻⁵ |
| `decoder.conv_post` | 1.26277×10⁻⁵ |

`decoder.conv_pre` receives **bit-exact input values** in both executions. Its
input changes from `[2,2048,200]` to `[1,2048,200]`; its left-channel output
already differs by up to 6.31809×10⁻⁶. Subsequent convolutions add their own
rounding differences, and nonlinear Snake/anti-alias/residual operations propagate
the perturbations. The error does not increase monotonically through all stages.

One-operation replays feed both batch shapes the **same captured Native inputs**,
isolating each operator from errors propagated from earlier layers:

- `dec_in_proj`: exact across batch shapes.
- `decoder.conv_pre`: differs across batch shapes, exact on repeated same-shape
  calls. Both replay inputs are contiguous, isolating batch size from input strides.
- First transposed upsampler and first anti-alias upsampler: exact across batch
  shapes in these replays.
- `decoder.resblocks.0.convs1.0`: independently differs with identical input
  values across batch shapes, while repeated same-shape calls are exact.

The convolution traces contain 17 FFT kernel events in each shape. Native batch
2 dispatches eight `cutlass_100_simt_cgemm_32x32_16x5_nt_align1` complex GEMM
events; mono batch 1 dispatches eight `internal::gemvx::kernel` complex GEMV
events. Thus this is an observed kernel change, not an assumption based only on
the final waveform discrepancy.

[PyTorch documents that batched and sliced floating-point calculations can differ
bitwise](https://docs.pytorch.org/docs/2.11/notes/numerical_accuracy.html#batched-computations-or-slice-computations).
The local same-input convolution replays establish that behavior for this model.

## Scheduling and candidate fixes

The unchanged Channel Parallel wrapper was also run with two independently
loaded trained replicas and two logical CUDA streams. For all three original
inputs, its output is bit-exact against sequential mono decoding and repeats
exactly. Consequently, local thread scheduling, channel order and stitching do
not add error. This does not test physical inter-GPU transfer, but that transfer
is not required to reproduce the discrepancy.

| Control | Observation | Implication |
|---|---|---|
| Decode two mono channels sequentially on one GPU | Same small mismatch | Communication and concurrency are not necessary causes |
| Original wrapper, two logical ranks | Exact against sequential mono | No extra local scheduling/stitch error |
| Keep Native stereo batch shape; zero unused channel, select owned channel | Exact for 5/10/20 s | Preserving batch shape restores the tested Native arithmetic |
| Pad only `conv_pre` to Native batch shape | Final L2 remains 1.24830 / 1.19204 / 1.16959×10⁻⁵ | Other convolutions also need matching arithmetic |
| Set `cudnn.deterministic=True` | Split still differs; Native itself changes | Determinism does not preserve the original Native numerical contract |

The deterministic-mode split errors are 1.28276 / 1.24117 / 1.22669×10⁻⁵ against
that mode's own Native outputs. Its Native outputs also differ from the original
baseline by 2.02879 / 1.95044 / 6.85526×10⁻⁶. It is therefore not an exact fix.
PyTorch distinguishes [convolution algorithm selection and algorithm
determinism](https://docs.pytorch.org/docs/2.11/notes/randomness.html#cuda-convolution-determinism).

## Recommendation

Keep the current Native audio path as the strict fallback. The existing Channel
Parallel implementation is structurally valid but is **not Native bit-exact**
and has not passed the established numerical gate. Do not silently relax it.

If bit-exactness is required, preserve Native's batch geometry or reproduce the
Native convolution accumulation path across mono calls. Padding a dummy channel
through the whole decoder is a working **local control**, but doubles each
rank's decoder rows and forfeits the intended compute saving. Its physical T4
exactness and latency have not been measured; it is not a production optimization.

The narrow next physical experiment is the isolated `conv_pre` batch-2/batch-1
replay and representative later convolutions, using these saved inputs and the
original T4 runtime. Trace the actual kernels; only then investigate a batch-
independent kernel implementation or constrained kernel selection. PyTorch's
deterministic flag alone cannot supply that contract. Full-decoder exactness must
be checked again because fixing the first convolution is insufficient.

If a numerical tolerance is acceptable instead, characterize sampler-produced
audio using waveform and spectral error plus listening tests before proposing a
new quality gate. The current random-latent fixtures establish the arithmetic
cause, not audibility or generated-audio quality. No production model/wrapper or
running Kaggle payload was modified during this diagnosis.

Evidence: [saved-input diagnosis](semantic_results/audio_channel_saved_diagnosis.json),
[trained logical-rank comparison](semantic_results/audio_logical_ranks.json),
[artifact hash manifest](semantic_results/audio_saved_references/manifest.json),
[batch-2 convolution trace](semantic_results/audio_conv_pre_traces/native_batch2.json.gz),
[batch-1 convolution trace](semantic_results/audio_conv_pre_traces/mono_batch1.json.gz).
