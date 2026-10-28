# Optional PTD TP2 adaptation — implementation and validation state

## Delivered on the isolated branch

Branch: `codex/h3-ptd-refasymopd`; implementation commit
`80039fbc88618a9ce1ea2023e41ea6c235869213`. Main is unchanged.

The adaptation adds eight per-hop offline INT8 TP2 expert sets, resumable
layer-parallel baking, shared unchanged auxiliary weights, hash-bound hop
overlays, the existing streaming residency window, and a dedicated eight-hop
PTD sampler. Native ComfyUI and the supplied reference files are not edited.
The original MODEL/sampler can be selected freely; PTD defaults to disabled.

The current pruned AdaLN bundle is retained only with explicit
`allow_pruned_adaln=True`. Per-hop refiner, PDD heads, audio chord correction and
streaming each have an independent switch. Turning off a trained PTD component
is an ablation, not a claim of equivalent generation semantics. See the
[architecture, graph wiring and switch table](../design/H3_PTD_TP2_ADAPTATION.md).

## Verified locally without model execution

* All 633 Python files in the delivery capsule parse successfully.
* Native ComfyUI files match the parent capsule byte for byte.
* Frozen environment files and hashed direct-pip core lock match byte for byte.
* The supplied reference mapping and schedule are copied verbatim.
* The public immutable source archive was downloaded and its complete size and
  SHA256 checked after Git push. This was a **source-only** download.
* `git diff --cached --check` passed for the implementation commit.
* No new LoRA weight payload was downloaded locally; no local GPU execution
  was used for qualification.

The source-only archive is 18,634,073 bytes, SHA256
`d29b4d1c8d831a688e96eec7b16e11abf17e83a91c226b1819151614bb4e9f1d`.
Its source capsule SHA256 is
`35c4e226685c5f152f924bf9ad2749de1476801f76ae1f6867098c3fe8f20680`.
Unrelated dirty-worktree attention/VAE changes are excluded from this delivery.

## Kaggle state

CPU API preflight **version 1 accepted**:
[H3 PTD TP2 CPU 20261028](https://www.kaggle.com/code/pamcks/h3-ptd-tp2-cpu-20261028).
Acceptance is not a test pass. At this report's submission checkpoint, final
outputs have not been collected and no new PTD T4 numerical result exists.
No Kaggle status/list or live-log endpoint was polled.

The new GPU job builder requires both matching frozen-setup and PTD CPU API
receipts before submitting its bounded T4 fixture. The prepared GPU experiment
uses two real trained DiT blocks, both trained refiners, complete PDD head
banks, all eight experts and Native TP2 calls on small seeded inputs. It
independently verifies every merged target against the Native weight formula,
checks ordered hop selection and finite FP32 state carry, and rejects the
reduced bank for production admission. It is not a full H3 generation or a
quality comparison. No private inputs or credentials are needed.

## Deployment limits still open

1. The full current bank has a conservative metadata-based estimate of
   **170.18 GiB**, requiring **184.53 GiB free** including download and reserve,
   plus any base storage not already present. Full-bank admission must fail
   before download on smaller disks. The reduced T4 qualification does not
   establish that a full bank fits the provider's `/tmp` capacity.
2. The supplied reference itself labels the `.999` schedule endpoint as
   unverified against the Diffusers scheduler. Pruned AdaLN adds a separate
   approximation. Generation quality remains unqualified.
3. The RMS-RoPE AOT plan is corrected to Native H3's **96/128** rotary layout.
   Its cache coverage must be confirmed in the new T4 experiment; historical
   128/128 preparation traces do not prove that coverage.

Small source-preservation and submission receipts are in
[`docs/data/h3_ptd_tp2_20261028`](../data/h3_ptd_tp2_20261028/).
