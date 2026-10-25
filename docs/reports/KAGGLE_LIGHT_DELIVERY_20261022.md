# Kaggle source/dependency delivery redesign

The user reports both prior GPU instances stopped while investigating silent
data preparation. Heavy mounted inputs are a plausible cause, **not a confirmed
root cause**: the cancelled runs returned no worker log or device receipt.
This change removes those mounts rather than assuming a kernel was running.

## New delivery contract

| Resource | Previous route | New route |
|---|---|---|
| Frozen ComfyUI + current source | Frozen Kaggle dataset plus embedded capsule | One 12,884,750-byte source-only archive, fetched from an immutable GitHub commit and SHA256-verified |
| Core libraries | CPU notebook outputs containing three Python-ABI wheel archives, mounted by GPU jobs | Direct PyPI pip install with exact versions and approved hashes; reuse already matching native packages |
| CPU/device reference | Mount complete historical notebook outputs to read a receipt | Small source-bound CPU qualification embedded in the new submission |
| Component weights | Mount the 5.66 GB CPU component-output notebook | Fetch byte-verified ranges from the original immutable HF model revisions after startup |
| VAE checkpoints | Duplicate full VAE files inside component notebook outputs | Fetch original HF bytes/ranges, checking the existing fixture hashes |
| Retained real input | Kaggle retained-input dataset / CPU output copy | Private HF dataset; 158,861,503-byte compressed input; the uncompressed 175,840,517-byte tensor remains hash-identical |
| E2E model weights | Already downloaded from immutable HF revisions | Unchanged; no checkpoint reupload or Kaggle model-data mounts |

Kaggle metadata has **empty `dataset_sources` and `kernel_sources`**. Source and
libraries arrive after the first notebook cell starts, where deadlines,
progress and logs are observable. Provider boot/provisioning remains outside
the notebook's control. A zero-mount run can still be delayed by Kaggle; no
startup speedup is claimed without returned receipts.

## Version preservation

The source archive preserves the original frozen ComfyUI bytes and legacy
attention API; no upstream clone/pull occurs. Current production patches and
the operator/E2E workload are preserved. Source publication uses a dedicated
`codex/` artifact branch, without altering the working checkout, main branch
or local index.

The 78 existing core pins remain selected, especially Kitchen **0.2.37** and
TileLang **0.1.13**. Pip uses `--no-deps --only-binary=:all: --require-hashes`
and the public PyPI index. Requirements are exact and use the already reviewed
wheel SHA256s for the actual Python minor. Packages already at the required
version are reused. Other core packages install into an isolated overlay.
Native Python/NumPy/Torch/Torchvision/Torchaudio/CUDA are not replaced. Unknown
ABIs, unavailable wheels, hash changes and API mismatches block explicitly.

CPU import/signature validation precedes GPU replacements. The CPU probe checks
the unchanged legacy attention signature, Kitchen entrypoints and runtime
imports; CPU tests cover current loading/LoRA bridges and native-platform
preservation. CPU success is not CUDA ABI, T4 numerical/performance or E2E
quality acceptance.

## Privacy and authentication

The public source archive contains source, dependency hashes and model range
plans; it contains no captured tensor payload, model weights, wheels or tokens.
Private activation data is sent only to a private HF dataset. Secrets remain
the default credential route. For this operator retry, the user explicitly
authorized a temporary token in the **private** notebook instead. That private
worker supplies it once in memory and removes its generated source file at
startup. The token is absent from public distributions, receipts and persisted
environment handoffs. This exception does not apply to other credentials/jobs.

The earlier queued/submitted sources, wheel archives, empty artifact attempts
and results remain intact. Model checkpoints remain unchanged. The original
conda snapshot and local environment are not modified.

## Publication and experiment state

- Source published at GitHub commit
  `a203a8074dece58be92b3d812362ff5f75cf6579`, artifact SHA256
  `44072cef35d063a90427fa5ae02fb1959a451e8351193fe40a091151878539d8`.
  An independent raw download matched both its length and checksum.
- New private CPU-only notebook accepted as **pamcks/h3-light-cpu, version 1**.
  This has zero Kaggle input mounts and creates no wheel archive outputs.
  Its initial requested metadata slug differed from the provider-returned URL;
  the returned slug is recorded as authoritative and future titles were fixed.
- CPU version 1 returned a matching `cpu_ready` receipt: **23 tests passed**,
  including the preserved APIs/loading/LoRA bridges and direct-pip guards.
  Linux pread also passed. The supervised worker took **120.02 seconds**;
  setup itself recorded 119.02 seconds. Pip installed 61 mismatched/missing
  packages and reused 17 already matching versions. Native base inventory
  remained unchanged. This measures notebook execution, not provider startup.
- E2E **version 2** returned real dual-T4 evidence. Direct delivery worked;
  offline baking verified all 208 targets, and reference/TE encoding completed.
  The first denoiser call failed in FC1's eager GPU epilogue at 51,220 tokens,
  before any sampling step completed. See the
  [version-2 result and memory diagnosis](H3_L1_CURRENT_OFFLINE_E2E_20261022.md).
- The private input is now committed in `cnmds/h3-kaggle-private-inputs`,
  revision `c2e165af2b7c2c8ca176d26732a42b56014315af`. Compressed and unpacked
  sizes/hashes match the retained local input. Local upload attempts stalled;
  a one-off CPU bridge read the existing 158.9 MB Kaggle input and uploaded it
  to private HF in 8.95 seconds. That bridge used one existing dataset mount,
  no new Kaggle data upload, and no GPU. Replacement GPU jobs remain zero-mount.
- Operator **version 3** was accepted using that fixed private HF revision,
  the original source publication and the passed CPU qualification. Its test
  matrix is unchanged. Acceptance does not establish execution or correctness.
- A separate source extension contains the two column-memory candidates and
  their CPU tests. CPU-only qualification version 2 was accepted; no GPU setup
  check or second full generation was launched. The source extension preserves
  all existing production runtime, model-plan and core-lock bytes. Its result
  does not supersede the original passed qualification until artifacts return.
- No status/list/live-log polling is used. CPU evidence and accepted submission
  receipts are in [the delivery record](../data/kaggle_light_delivery_20261022.json).
  The E2E failure is physical T4 evidence; there is no completed generation or
  quality acceptance. The new column-memory candidates have local-only evidence.

Local working root: `D:/attention_eval/h3_light_delivery_20261022/`. It preserves
the distribution, publication receipt, CPU notebook and binding. Builders are
`build_kaggle_light_delivery.py` and `build_kaggle_light_job.py`; setup and
execution helpers are `prepare_kaggle_direct_pip.py` and
`launch_kaggle_light.py`. Historical builders are unchanged and should not be
submitted as a workaround.
