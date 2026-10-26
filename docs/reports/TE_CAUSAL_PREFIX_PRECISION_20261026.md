# TE causal prefix invariance and FP32 precision — 2026-10-26

This audit checks the current `kitchen_causal` default at the attention boundary,
after Q/K normalization and RoPE. It uses the pinned Kitchen **0.2.37**, the
current production adapter, compact GQA and synthetic inputs. It does not load
trained weights or run an H3 generation.

**Both physical T4s confirmed that fixed-packed causal masking passes, while
the floating-input API fails strict prefix invariance. FP32 input also does
not preserve FP32 output precision: it returns INT8 attention values rounded
through BF16 storage.** The production policy and pinned dependencies were
preserved during this audit.

## Physical T4 results

The two T4s each completed 84 precision cases, 168 prefix comparisons and 72
fixed-packed controls. The raw metrics and exact submitted source modules are
retained in the [evidence manifest](../data/te_causal_precision_20261026/manifest.json).

| Check | Result across both T4s |
|---|---|
| Future packed Q/K/V edits with unchanged scales | **144/144** prefix outputs byte-equal |
| Floating-input prefix comparisons | **288/336** changed, including ordinary-scale future edits and truncation |
| FP32 reference, fixed-length future edits | Maximum prefix difference **0**, all controls |
| FP32 reference versus independent CPU FP64 | Maximum absolute difference **1.021e-6** |
| Current TE adapter versus pinned Kitchen | Byte-equal in all **168** precision cases |
| FP32-input returned values | All **84** cases exactly on the BF16 representable grid |

### Ordinary-scale future-token edits

N353/H32/Hkv4/D128, FP32 input, first 31 query outputs. Earlier Q/K/V values
and all lengths were held fixed; only the named future input was resampled
at its original unit Gaussian scale. Relative L2 is the median across two
seeds, boolean/additive masks and both ranks; maximum absolute error is the
largest observed value.

| Edited future input | Prefix relative L2 drift | Max absolute drift |
|---|---:|---:|
| Q | 0.141% | 0.01953 |
| K | 0.669% | 0.03125 |
| V | 1.010% | 0.03125 |

The FP32 reference prefix stayed exactly unchanged for these edits. The packed
diagnostics also identified the corresponding changed dequantized prefix
operand. This establishes quantization-induced future dependence independently
of a causal score-mask error.

Some aligned cuts remained unchanged: future Q resampling at cut128 passed
all eight FP32 comparisons, while future V resampling still changed outputs
by a median **0.911%** relative L2. At cut257 the future-V drift was **0.654%**.
FP16 inputs also failed invariance; future-V drift at cut31 was **0.983%**.
Alignment does not provide a general invariance guarantee.

Running only the first128 tokens versus taking that prefix from the N353 call
changed FP32 outputs by a median **1.545%** relative L2, maximum absolute
**0.046875**. This comparison includes length-dependent quantization choices.

The separate 32× future-V stress test produced **22.84%** median prefix
relative L2 drift at cut31, maximum absolute **0.52344**. This is a stress
counterexample and is not an estimate of real H3 activation behavior.

### Precision against FP32 attention

Unit-standard-deviation Gaussian Q/K/V, boolean causal mask. Values are the
median relative L2 error across two seeds and both ranks; maximum absolute
error takes the largest FP32-input result.

| Tokens | FP16-input error vs FP32 | FP32-input error vs FP32 | FP32-input max absolute error |
|---|---:|---:|---:|
| 65 | 1.067% | 1.077% | 0.03638 |
| 129 | 1.131% | 1.147% | 0.04150 |
| 353 | 1.234% | 1.245% | 0.04914 |
| 513 | 1.276% | 1.289% | 0.04699 |
| 1025 | 1.326% | 1.336% | 0.04821 |

At N353/FP32, rounding the ideal FP32 reference to BF16 alone gave **0.166%**
relative L2 error. FP32 attention on dequantized Kitchen operands differed
from the original-input reference by **1.167%**. Kitchen output differed from
that dequantized reference by **0.433%**, which includes probability/kernel
and output-store effects. These are separate comparisons, not additive error
components. Input quantization accounts for much more error than the ideal
BF16 store alone.

Boolean and finite additive causal masks gave the same N353 aggregate errors.
The padding/bias case was similar: **1.245%** for FP32 input. Scaling all inputs
to amplitude0.2 gave **0.742%** error; the amplitude4 stress case gave **4.147%**,
with maximum absolute error **3.565**. These measured synthetic errors do not
establish a trained-model quality threshold.

## Contracts checked separately

| Property | Check |
|---|---|
| Causal masking on fixed packed inputs | Keep scales fixed, change future packed Q, K or V, require earlier outputs to remain byte-equal |
| Prefix invariance of the floating-input API | Keep lengths and all prefix inputs fixed, edit future Q/K/V, requantize, compare earlier outputs |
| Prefix-only execution | Compare the prefix of a full causal call with an independently quantized truncated call |
| FP32 arithmetic reference | Explicit FP32 GQA QK, softmax and PV with TF32 disabled; independently check a small case against CPU FP64 |
| Storage precision | Check output dtype and whether FP32-returned values lie exactly on the BF16 lattice |
| Adapter behavior | Require the current TE adapter and pinned raw/prequantized APIs to return equal output values |

Future-only edits use both fresh seeded Gaussian values at the original scale
and a separate **32× stress** mutation. The stress result is not representative
of ordinary TE activations. Prefix-only execution is reported separately:
changing lengths can also change quantization/packing choices.

## Precision interpretation

Kitchen accepts FP32 input, but its pinned implementation chooses BF16 output
storage for that input dtype and then converts the result back to FP32.
Returning `torch.float32` therefore does not mean the output has FP32 numerical
precision. The retained CUDA attention template accepts only half/BF16 output
stores; its floating accumulation state does not undo operand/probability
quantization or the final store rounding.

The audit reports error against the original floating-input FP32 reference,
against an FP32 reference on Kitchen's dequantized packed operands, and the
error of ideal output-storage rounding alone. Those separate comparisons
distinguish input quantization from remaining probability/kernel/store error;
their norms are not additive.

Q/K scales are shared across query/key groups, and packed V has one scale per
batch/KV-head/channel across the sequence. Future inputs can therefore change
the representation of earlier inputs. Changing a K anchor can also shift
dequantized keys; such a common shift is mathematically softmax-invariant, so a
key-representation difference alone is not treated as proof of output leakage.
The audit measures output changes directly and includes a fixed-packed control.

## Method and reproducibility

Precision cases use B1/H32/Hkv4/D128, N65, 129, 353, 513 and 1025, FP16/FP32
inputs and two seeds. Causal masks are boolean, finite additive, or additive
with a blocked earlier key and a small allowed-position bias. At N353 the
amplitude sweep is 0.2, 1 and 4; the unit-standard-deviation Gaussian case is
the main normalized-scale diagnostic, not a trained activation distribution.
Every mask leaves at least one valid key per query.

Prefix tests use N353 and cuts 31, 128 and 257. They isolate future Q, K and V
changes. A cut aligned to one quantization group is not assumed to isolate
other groups or V's sequence-wide scales. The FP32 reference uses the same
input values after their selected input-dtype conversion.

Five CPU oracle tests and seven existing TE adapter/dispatch tests passed. A
six-case local SM120 API smoke passed but is not T4 evidence. The T4 notebook
reuses the exact frozen source/core bootstrap already qualified on Linux CPU;
two supplemental source modules are embedded with SHA256 bindings. It does not
edit that capsule, install new core versions, mount outputs or download models.

The setup deadline is 240 seconds and the operator deadline is 300 seconds.
One final self-contained JSON is requested after the fixed window plus startup
allowance, without status/list/live-log polling. Its batch includes actual
runtime versions, both physical GPU identities, raw per-case metrics, source
hashes and the immutable delivery binding.

Implementation: [audit](../../experiments/kitchen_attention/te_precision.py),
[reference tests](../../experiments/kitchen_attention/test_te_precision.py),
[offline analysis](../../benchmarks/analyze_te_causal_precision.py).

Actual environment: **Torch 2.11.0+cu128 / CUDA 12.8 / NumPy2.1.3**, Kitchen
**0.2.37**, TileLang **0.1.13**, two distinct Tesla T4 UUIDs. Each rank's operator
audit took about five seconds, excluding imports/bootstrap. The self-contained
final JSON was collected successfully in one operation; no extra artifact or
status request was made. The legacy ComfyUI commit remains
`d49e888586dd8ae012c0667b33466b815fee07f7`, and local runtime/policy/adapter
hashes match the preceding validated state.

The immutable base capsule SHA256 is
`dcbc4c4da405fc8092654e62ef51bf1b7bc15098fac236391774121199d9c5f4`; supplemental
source hashes and the final notebook SHA256 are recorded with the raw batch.
No activation tensors, model weights or credentials are in the evidence.

## Consequence for TE

The current `kitchen_causal` default satisfies the tested packed causal masking
contract. A requirement for strict causal prefix invariance of floating Q/K/V,
or for true FP32 output precision, needs a different numerical path. INT8
retention would require future-independent quantization/calibration and a
deliberate output precision contract. Casting the current BF16-rounded result
to FP32 cannot recover lost precision. This audit leaves the selected TE
default unchanged; model-level accuracy and repeated-layer amplification were
not measured.
