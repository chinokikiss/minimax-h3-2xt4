# FP16 residual fix and weight folding: local operator study

**Result: worth retaining as an isolated candidate, not ready for a default change.** Scaled FP16 removes a real first-block overflow, but it does not remove residual-update absorption or ordinary FP16 rounding. Fixed scale folding can avoid changing INT8 weight bytes. Full channel-wise RMSNorm-gain folding adds requantization error.

## Scope and reproducibility

- Actual hardware: one RTX 5060 Ti, SM120; Torch 2.12.1+cu132, Kitchen 0.2.37 and the frozen conda environment. All 94 package versions match the preserved snapshot. No package or source API was updated.
- The current pinned Native front end reconstructs block-0 inputs from retained real Tokyo challenge conditioning, reference latents and initial noise at sigma=1. The source workload records 512x512, 90 frames, seed 20261013. This is the challenge fixture, not the earlier 640x352 workload.
- Only conditioning projection/text-refiner/input-packing operators execute during preparation. Execution is intercepted before the first main DiT block. No TE or VAE inference, 50-block DiT, sampler trajectory, full generation or Kaggle submission is run.
- Full retained shape is **[7895,5376]**: 417 text, 266 reference-image, 300 audio and 6912 generated-video tokens. All 7895 tokens participate in the main block-0 screen. A separate LoRA test takes 513 order-preserving, uniformly selected tokens; that subset changes attention context and is explicitly not full-sequence validation.
- Two logical TP ranks execute real rank-owned trained weights on the same GPU with synchronized mock collectives. Rank agreement is checked, but neither those collectives nor SM120 timings qualify T4/NCCL performance.
- The active SM75-only TileLang row backend falls back to Native on this GPU. These are current logical TP2/Native-Kitchen arithmetic probes, not execution tests of FP16 TileLang SM75 kernels. Current packed/SwiGLU/owned-FP32 kernels also have FP32 guards; an FP16 deployment needs separate dispatch/kernel work.
- There are 18 complete-sequence and 18 LoRA-subset configurations. 15 shared configurations reproduced their output hashes across two fresh full-sequence processes. Compile/loading and diagnostic reductions are not timed; no speed or measured pipeline-VRAM claim is made.

## Exact gauge, including epsilon

For a positive constant `c = 2**k`, store `y = x/c`. A pre-normalized residual block becomes:

```text
RMSNorm_eps(x) = RMSNorm_(eps/c^2)(y)
h = AdaLN(RMSNorm_(eps/c^2)(y))
y_next = y + gate * branch(h) / c
```

The same epsilon transformation applies to LayerNorm; its affine beta stays unchanged. RMSNorm, AdaLN, timestep curves, Q/K norm/RoPE and residual-update arithmetic remain FP32 islands in the prototype. The FP16-stream variant rounds at storage boundaries. The FP16-branch variant additionally uses FP16 projection/attention/MLP inputs and outputs, with INT32 base GEMM SUM retained.

Leaving epsilon unchanged is wrong near zero. Storing adjusted epsilon in FP16 is also wrong: `1e-5 / 256^2` rounds to zero in FP16. Nonlinear SiLU, AdaLN shifts, output biases and LoRA contributions cannot be scaled away without compensation.

The FP32 gauge control, FP32 output-scale fold and FP32 scalar norm-gain fold all match the current TP2 block **byte-for-byte** in the tested full-sequence and LoRA-subset cases. This validates the transformation controls for these cases, not FP16 equivalence or universal floating-point identities.

## Overflow, underflow and lost updates

The unscaled input maximum is 113.518494. After attention, the baseline residual maximum is 1049.691406; after the MLP it is **110585.664062**, beyond FP16's maximum 65504. Exactly one baseline output element exceeds that range. Unscaled FP16 storage produces one infinity after the MLP in each logical rank.

| Residual divisor | Nonfinite output elements/rank | Initial nonzero values cast to zero | Attention updates rounded away | MLP updates rounded away | Block relative L2 |
|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 2 | 1,241,626 | 1,402,694 | not finite |
| 4 | 0 | 3 | 1,241,625 | 1,402,716 | 0.2670% |
| 16 | 0 | 12 | 1,241,775 | 1,402,606 | 0.2550% |
| 64 | 0 | 42 | 1,241,624 | 1,401,652 | 0.2578% |
| 256 | 0 | 208 | 1,241,426 | 1,402,122 | 0.2672% |
| 4096 | 0 | 3,295 | 1,257,266 | 1,426,205 | 0.2620% |

At c=16, approximately **2.926%** of finite nonzero attention updates and **3.305%** of MLP updates leave the stored residual unchanged. The corresponding FP32 control counts are only 211 and 241. These counters use each candidate's own finite FP32 increment, not a semantic importance score or a matched Native-update mask.

Power-of-two scaling changes exponent range, not the mantissa budget or update/residual ratio. It therefore does not solve ordinary FP16 update absorption. Larger divisors trade overflow headroom for more subnormal/zero values: c=4096 loses 3295 initially nonzero values versus 12 at c=16. There is no globally selected divisor from this one-layer, one-step screen.

The L2 column is implementation diagnostics only. It is not a generation distribution/semantics acceptance gate, and finite output is insufficient for promotion. Later layers/steps and error accumulation remain unmeasured.

## Weight Fold A: fixed residual output compensation

Instead of dividing a dequantized row result by c, divide the FP32 `out_proj` and `fc2` weight-scale vectors by c. INT8 weights, activation quantization and INT32 collective payloads are unchanged. Online LoRA requires its separate post-B strength divided by c as well; adapter order and A/B GEMM shapes stay unchanged. A schema with row bias is rejected until that bias has its own compensation.

Folding happens before the output cast, so it is not universally byte-equivalent to dividing an already-rounded FP16 result. In the full-sequence c=16 case, folded/unfolded FP16-branch output hashes match. At c=256 they differ; the LoRA subset can also differ at c=16. Those distinctions are retained, not hidden behind an equality claim.

## Weight Fold B: scalar norm-gain redistribution

A safer fold than changing every input-channel gain is a common power-of-two factor p:

```text
norm_gain_new = p * norm_gain
AdaLN_shift_new = p * AdaLN_shift   # includes learned weight AND bias rows
following_QKV_or_FC1_weight_scale_new = old_scale / p
following_online_LoRA_post_B_strength_new = old_strength / p
```

Modulation scale/gate outputs remain unchanged. Compensating the shifts ensures the whole modulated input is multiplied by p; scaling only the norm gain is wrong for DiT. Compensation occurs before downstream nonlinearities. The implementation rejects an AdaLN adapter wrapper rather than silently omitting its row-wise correction.

With p=64, the FP32 whole-block control is byte-identical in these fixtures. In the FP16 TE projection probe, input casts that lose nonzero values fall from 2 to 0 and subnormal outputs from 6755 to 114. Relative projection L2 barely changes (0.33357% to 0.33353%): underflow prevention is useful range engineering, not a general precision improvement.

In the full-sequence FP16 DiT flow, adding the scalar-gain fold changes block L2 from 0.26682% to 0.26723%, with no nonfinite values in either. Reduced norm-output underflow alone does not establish better block accuracy or generation quality.

## Weight Fold C: full channel-wise RMSNorm gain

The separate TE probe uses learned layer-0 RMSNorm gain and 512 Q-projection output rows with full input width 5120, on 257 seeded synthetic normalized inputs. It does not represent real TE activation distributions.

Checkpoint INT8 weights live in a ConvRot basis. To fold a nonuniform gain correctly, Native-dequantize/inverse-rotate the weight, multiply input columns in the original basis, then Native-requantize the full input width. Directly multiplying rotated INT8 columns by gamma is generally wrong because diagonal gamma and Hadamard rotation do not commute.

Dense algebraic folding has relative L2 1.688e-07. After Native INT8 requantization, the measured projection difference is **1.5198%**. This is a different quantized model, not a free exact fold. Static full-gain folding through dynamic DiT AdaLN shifts is rejected by the helper.

## Memory meaning and next decision

The [7895,5376] residual payload is 161.91 MiB in FP32 and 80.95 MiB in FP16, per tensor/rank. That is a byte ledger only. The prototype intentionally materializes FP32 normalization/update islands and stores diagnostic observations; no 50% whole-block or pipeline-VRAM saving follows.

Retain c=4/c=16 as calibration candidates and fixed-output/scalar-gain folding as separate switches. Keep FP32 residual accumulation as the conservative comparator. Before any runtime option is introduced, calibrate real later-layer/later-step ranges and update absorption, validate online LoRA/AdaLN/final-head compensation, then implement a guarded SM75 FP16 preparation path and measure it on physical T4. Generation distribution/semantic acceptance remains a later full-pipeline gate, not something inferred here.

## Reproduce

Run from the repository root with the existing conda `comfyui` Python. No setup/install command is required:

```powershell
& C:\ProgramData\miniconda3\envs\comfyui\python.exe benchmarks/run_fp16_residual_fix.py prepare
& C:\ProgramData\miniconda3\envs\comfyui\python.exe benchmarks/run_fp16_residual_fix.py screen --tokens 0
& C:\ProgramData\miniconda3\envs\comfyui\python.exe benchmarks/run_fp16_residual_fix.py screen --tokens 513 --lora D:/minimaxH3/loras/dmad_minimax_h3_4step_full_critic_comfyui.safetensors
& C:\ProgramData\miniconda3\envs\comfyui\python.exe benchmarks/run_fp16_residual_fix.py norm-fold
& C:\ProgramData\miniconda3\envs\comfyui\python.exe -m unittest benchmarks.test_fp16_residual_fix
```

The default cached-input location is local historical evidence; a clean clone needs those private fixture files or another explicitly supplied `--inputs` directory. Artifact tensors, prior source/output versions and logs stay under `D:/attention_eval/fp16_residual_fix_20261020`.

Source: [experimental gauge/folding helpers](../../benchmarks/fp16_residual_fix.py), [operator driver](../../benchmarks/run_fp16_residual_fix.py), [analytical tests](../../benchmarks/test_fp16_residual_fix.py), [compact results](../data/fp16_residual_fix_local_20261020.json). Ten new analytical/contract tests plus eight existing pipeline/runtime tests pass. Runtime source, model files, environment versions and defaults are unchanged.

Mathematical references: [PyTorch RMSNorm definition](https://docs.pytorch.org/docs/stable/generated/torch.nn.RMSNorm.html), [NVIDIA floating-point formats](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/mathematical-functions.html). These explain the formula/format, not qualification of a different Torch/CUDA version. Execution uses the frozen local packages.
