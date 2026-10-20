# Frozen user runtime

`local_conda.json` preserves the conda `comfyui` package versions, the selected
ComfyUI source inventory and the source archive digest. Exact requirements and
constraints accompany it. These files contain versions/hashes, not credentials
or model weights.

Do not regenerate this lock from a different environment. The exporter refuses
package/source drift. A separately reviewed baseline is needed for deliberate
version changes. `freeze_h3_runtime.py` exports the publishable dataset under
`D:\attention_eval\h3_conda_runtime_20261019\kaggle_bundle` on this machine.

The setup helper's only source is `benchmarks/frozen_kaggle_setup.py`; copies in
exported datasets are generated artifacts. Generated copies are not maintained
as a second implementation here.

The Windows package snapshot is not a Linux dependency or ABI qualification.
Seven Linux-only requirements in the matching Torch wheel still need an explicit
platform lock. GPU jobs require a verified CPU-prepared wheelhouse and install
it offline. Historical T4 results on Torch 2.11 / TileLang 0.1.15 are separate
from this snapshot's Torch 2.12.1 / TileLang 0.1.13.
