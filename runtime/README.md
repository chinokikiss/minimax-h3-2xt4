# Frozen user runtime

The latest authorized Kaggle delivery uses an immutable GitHub source archive
and exact, hashed **direct pip installs**, with zero dataset/kernel-output
mounts. It preserves this snapshot and the native-Torch exception below;
libraries already at the required version are reused. Historical wheelhouse
instructions describe the earlier route. See
[the delivery redesign](../docs/reports/KAGGLE_LIGHT_DELIVERY_20261022.md).

`local_conda.json` preserves the conda `comfyui` package versions, the selected
ComfyUI source inventory and the source archive digest. Exact requirements and
constraints accompany it. These files contain versions/hashes, not credentials
or model weights.

Do not regenerate this lock from a different environment. The exporter refuses
package/source drift. A separately reviewed baseline is needed for deliberate
version changes. `python -m tools.runtime.freeze_h3_runtime` exports the publishable dataset under
`D:\attention_eval\h3_conda_runtime_20261019\kaggle_bundle` on this machine.

The setup helper's only source is `benchmarks/frozen_kaggle_setup.py`; copies in
exported datasets are generated artifacts. Generated copies are not maintained
as a second implementation here.

The Windows package snapshot is not a Linux dependency or ABI qualification.
Seven Linux-only requirements in the matching Torch wheel still need an explicit
platform lock. GPU jobs require a verified CPU-prepared wheelhouse and install
it offline. Historical T4 results on Torch 2.11 / TileLang 0.1.15 are separate
from this snapshot's Torch 2.12.1 / TileLang 0.1.13.

## Authorized Kaggle native-Torch profile (2026-10-21)

The user explicitly allows Kaggle's native Torch and its matching platform
packages. This does not change the local conda lock. Kitchen **0.2.37**,
TileLang **0.1.13**, the remaining core pins, the frozen ComfyUI source and its
legacy attention API remain preserved.

`benchmarks/prepare_kaggle_native_torch.py` prepares exact hashed core wheels on
a CPU host. A GPU job installs them offline into an isolated overlay and records
the actual native Torch/Python/CUDA stack in `runtime_profile.json`. CPU import
success does not qualify CUDA ABI or T4 operators; those have separate physical
device gates. Native NumPy/Torchvision/Torchaudio stay with the platform stack;
UI-only Comfy distributions are omitted from this operator-only profile.

`benchmarks/build_t4_native_validation_batch.py` builds the bounded component
batch. Never submit historical notebooks with divergent setup. Do not poll
Kaggle status/list/live logs; collect version-bound final artifacts once after
the execution window. See [the experiment report](../docs/reports/T4_NATIVE_TORCH_VALIDATION_20261021.md).

The 2026-10-22 physical operator batch completed on two T4s with native Torch
2.11.0+cu128, Kitchen 0.2.37 and TileLang 0.1.13. The service core gate now
matches those preserved core pins. Its default policy selects FP16-fix,
fused_stored preparation and scaled FP16 row SUM. No local package or original
snapshot was changed; full-pipeline quality remains unqualified. See
[the promotion record](../docs/reports/H3_FP16_FUSED_DEFAULT_20261022.md).
