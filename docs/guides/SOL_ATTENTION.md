# Sol Attention Production Integration

## Status

Sol All-INT8 is an explicit optional attention backend for the MiniMax H3 TP2
runtime on NVIDIA SM75/T4. Kitchen remains the default backend.

The former TileLang sparse DiT attention implementations (drop, scalar, and
their shared sparse-routing stack) have been removed from the production
patches package. The former TileLang Group32 TE attention path has also been
removed. TileLang remains in use only for unrelated qualified operators.

## DiT

The loader exposes three related controls:

- dit_attention: kitchen, sol, or native
- dit_attention_schedule: optional 1-based per-sample NFE schedule
- sol_tau: global Sol routing threshold multiplier, default 1.0

A Sol schedule entry may override the global threshold with `sol@tau`:

    1:kitchen,2-7:sol@1.0,8:kitchen

An entry without an override inherits `sol_tau`:

    1:kitchen,2-7:sol,8:kitchen

Ranges may be omitted. An omitted NFE uses the dit_attention default. Tau is
accepted only on Sol entries and must be finite and nonnegative. Sol routing applies only to DiT.

The NFE counter is reset by begin_sample and incremented only for actual
denoiser calls, so schedules do not leak across requests.

### Condition-token protection

For Sol DiT attention, the protected prefix ends at the start of the final
target-audio segment in Native PackedLayout. Text, reference and other true
conditioning tokens are protected; generated target audio and generated video
remain on normal Sol routing.

Protection is exact at the native Q64/K64 routing granularity:

- every query attends all protected key blocks;
- every protected query block attends all key blocks.

The protected token boundary is rounded outward to native 64-token blocks.
This deliberately spends additional dense work around a ragged boundary to
avoid dropping conditioning information.

## Video VAE

Video VAE attention supports Kitchen INT8 (default) or Native SDPA.
Its Sol routing and D64 wrapper have been retired; DiT Sol remains supported.

## Native build and admission

Production Sol uses only the vendored source under
patches/sol_attention_native/.

The shared object is compiled for sm_75 with the preserved CUDA 12.8 compiler
and cached by the exact source/compiler identity. When a session selects Sol,
it is included in the normal AOT preparation stage before DiT execution; Video
VAE Sol is AOT-prepared during vae_prepare before decoder weights are loaded.
An externally supplied H3_SOL_ALL_INT8_LIBRARY does not override production
source selection.

Explicit Sol selection is fail-closed on unsupported hardware or build/runtime
errors. It does not silently substitute Kitchen.

## Qualification

The earlier complete-call T4 operator benchmark for the unprotected D128 Sol
path showed the following representative speedups versus Kitchen dense INT8:

- 6,814 tokens: approximately 2.0x
- 16,384 tokens: approximately 2.95x
- 51,220 tokens: approximately 2.46x

The current production native source additionally enforces bidirectional
protected-block routing and has been successfully compiled for sm_75 with
CUDA 12.8.93.

The remaining DiT-specific qualification work covers protected-routing
performance, real-H3 trajectory drift, and final-generation quality by
NFE policy. DiT Sol remains opt-in; Video VAE Sol is retired.
