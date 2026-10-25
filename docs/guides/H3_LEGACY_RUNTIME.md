# Preserved ComfyUI and conda runtime

Local execution uses `C:\ProgramData\miniconda3\envs\comfyui\python.exe`.
The source is this repository's `ComfyUI` checkout at
`d49e888586dd8ae012c0667b33466b815fee07f7`. Retain its legacy attention API:

```python
Attention.forward(x, rotary_pos_emb, pre_norm, residual, residual_scale)
```

The API-different sibling `ComfyUI-master` is not substituted. Its upstream
revision and chronological ordering are unverified; see
[the provenance correction](../reports/H3_RUNTIME_PROVENANCE_AUDIT.md). The old Query Parallel
attention implementation, Native fused normalization/RoPE, and Kitchen INT8
attention remain intact. Models remain under `D:\minimaxH3`.

## Default local execution

```powershell
& C:\ProgramData\miniconda3\envs\comfyui\python.exe -m tools.runtime.setup_h3_legacy
& C:\ProgramData\miniconda3\envs\comfyui\python.exe -m tools.runtime.launch_h3_legacy --preview --check
& C:\ProgramData\miniconda3\envs\comfyui\python.exe -m tools.runtime.launch_h3_legacy --preview
# Equivalent launcher using that exact interpreter:
.\launch_h3_legacy.cmd --preview
```

[`tools/runtime/setup_h3_legacy.py`](../../tools/runtime/setup_h3_legacy.py)
only verifies the environment; it installs nothing and
creates no venv. `--preview` permits local UI/checks on the single 5060 Ti.
Physical TP2 still requires two GPUs, NCCL, the compiler, and operator ABI
qualification. Local previews do not establish T4 correctness or performance.
Node/API isolation is now explicit with `--isolated-ui` for smoke tests; normal
launches preserve the selected checkout's available custom and partner nodes.
Dynamic VRAM remains disabled because TP2 workers own the weight window.

The frozen user environment is `runtime/local_conda.json`: 94 relevant package
versions, including transitive dependencies. Representative versions:

| Package | Preserved conda version |
|---|---|
| Torch | 2.12.1+cu132 |
| Torchvision | 0.27.1+cu132 |
| Torchaudio | 2.11.0 |
| TileLang | 0.1.13 |
| comfy-kitchen | 0.2.37 |
| Transformers / Tokenizers | 5.9.0 / 0.22.2 |
| AV | 18.1.0 |
| Frontend / workflow templates / docs | 1.52.7 / 0.11.59 / 0.5.11 |

ComfyUI emits recommended-version warnings for those three UI packages;
startup succeeds and they are deliberately preserved. The new snapshot is
not a claim that every optional addon in the conda environment is qualified.

## Kaggle preparation

**Current authorized exception (2026-10-21):** Kaggle may use its native Torch
and matching platform packages. Core libraries retain their validated pins,
especially Kitchen 0.2.37 and TileLang 0.1.13, and the frozen ComfyUI source keeps
the legacy attention API. The local conda lock remains unchanged. CPU wheel
preparation/import gates passed; CUDA/T4 qualification is a separate bounded
component batch. See [the current report](../reports/T4_NATIVE_TORCH_VALIDATION_20261021.md)
and [the profile contract](../../runtime/README.md).

The instructions below describe the original strict Windows-snapshot-to-Linux
profile and its historical blockers. They are retained for reproducibility,
not used to override the user's native-Torch exception.

The bundle at `D:\attention_eval\h3_conda_runtime_20261019` contains the frozen
source archive, hashes of 1,260 tracked source files, exact requirements,
constraints, and the standalone setup helper. No model weights are included.
`python -m tools.runtime.freeze_h3_runtime` refuses to overwrite a baseline whose source or package
versions have drifted. Re-running it preserves the existing source archive.
Upload only its `kaggle_bundle` subfolder, which contains five runtime files.
The source archive has a `.bin` suffix there to prevent automatic unpacking by
the dataset service. Do not upload the evidence/preflight folders.

The current operator and REF2VA builders require `--runtime-dataset OWNER/SLUG`.
They bind the manifest digest into the experiment and use the attached archive;
there is no git clone/pull or upstream requirements installation in their setup.
Historical builders and already published notebooks retain their historical
contents; do not submit those directly.

Prepare Linux wheels on a **CPU-only Linux host**, using Python 3.11:

```bash
python frozen_kaggle_setup.py --bundle /path/to/frozen-bundle \
  --root /tmp/h3-runtime-preflight \
  --export-wheelhouse /path/to/frozen-bundle/linux-wheelhouse
```

This resolves exact versions under the frozen constraints, rejects unknown
platform dependencies and version substitutions, and downloads hash-verified
wheels. A missing wheel or ABI/resolver conflict is a blocker. There is no
fallback to newer packages or historical T4 package versions.

GPU jobs require that prepared wheelhouse and install offline with
`--no-index --no-deps --require-hashes`; they do not resolve/download packages
while using GPU quota. Wheel versions, inventory, hashes and manifest identity
are checked before installation. Linux environment preparation and T4 operator
qualification are separate steps.

**Historical remote status before 2026-10-21:** this corrected bundle had not been uploaded or submitted as
a new Kaggle job. Linux wheel preparation remains unverified. Historical T4
Torch 2.11 / TileLang 0.1.15 results do not qualify the current conda stack on
Linux. The pipeline's historical operator ABI gate still blocks an unqualified
TileLang version; its error directs qualification instead of package changes.
No full H3 generation or Kaggle GPU setup experiment was run for this change.

An official-wheel metadata audit found a concrete platform gap. The matching
Linux Torch `2.12.1+cu132` wheel exists, but declares seven Linux dependencies
absent from the Windows snapshot: `cuda-toolkit`, `cuda-bindings`,
`nvidia-cudnn-cu13`, `nvidia-cusparselt-cu13`, `nvidia-nccl-cu13`,
`nvidia-nvshmem-cu13`, and `triton`. The audit is saved in
`torch_linux_metadata.json`. They require a separate explicit platform lock and
ABI validation; the current strict preflight rejects their implicit addition.
Public Linux wheels also exist for TileLang `0.1.13` and Torchaudio `2.11.0`;
wheel availability alone does not establish ABI or numerical compatibility.

## Evidence

`D:\attention_eval\h3_conda_runtime_20261019` contains the conda runtime check,
ComfyUI startup/custom-node import log, 49 passing regression tests, and the
successful local archive/package preflight. This preflight used Windows and
installed nothing; it is not Linux wheel or physical T4 validation.

`D:\attention_eval\h3_legacy_runtime_20261019` retains the earlier experimental
Torch 2.11 venv evidence, including trained VAE and logical-rank checks.
`D:\h3_tp2_legacy_env` is historical and no longer the default. It was created
to avoid modifying conda while testing another stack; choosing it as the default
was unnecessary. The original conda installation was not downgraded.
