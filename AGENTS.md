# H3 runtime preservation

- Preserve validated source, attention API, operators and dependency versions by default. Do not update upstream code or dependency versions as part of adaptation or testing.
- Local execution uses `C:\ProgramData\miniconda3\envs\comfyui\python.exe`. The experimental `D:\h3_tp2_legacy_env` is historical, not the default.
- Retain the legacy attention API in the local `ComfyUI` checkout (`d49e888586dd8ae012c0667b33466b815fee07f7`). Do not replace it with the newer sibling `ComfyUI-master` implementation.
- `runtime/local_conda.json` is the user environment snapshot. Kaggle must use the frozen local source archive and exact package versions; never clone upstream HEAD, run git pull, install unpinned dependencies, or silently substitute platform versions.
- Windows/Linux wheel or ABI differences are blockers to report and validate explicitly. Historical T4 Torch 2.11 / TileLang 0.1.15 measurements do not qualify the current conda Torch 2.12.1 / TileLang 0.1.13 stack on Linux.
- Older Kaggle builders/notebooks are historical and may contain divergent dependency setup. Do not submit them directly. Use the frozen setup preflight first, and replace their setup before any new submission.
- Preserve historical submitted sources, results and environments. Do not run a full H3 pipeline or consume Kaggle GPU time for environment setup checks.

## Authorized Kaggle exception — 2026-10-21

- The user explicitly permits Kaggle's native Torch rather than reproducing the local Windows Torch/CUDA package stack. Keep its matching native Python/NumPy/Torchvision/Torchaudio/CUDA environment and record actual versions; do not modify the local conda environment or original snapshot.
- Core libraries whose APIs/operators H3 relies on remain pinned, especially comfy-kitchen 0.2.37 and TileLang 0.1.13. Preserve the frozen ComfyUI source and legacy attention API. Validate core imports/signatures on CPU before operator tests; do not upgrade upstream source or core libraries to resolve compatibility problems.
- Do not poll Kaggle status/list endpoints or live logs, including through an API or browser workaround. Minimize interactions: batch submissions, use bounded runs, and collect final outputs once per batch.

## Authorized delivery change — 2026-10-22

- The user authorizes GitHub/HF delivery instead of large Kaggle datasets or mounted notebook outputs, and direct pip installation instead of shipping wheel archives.
- Preserve the frozen source and exact core pins. Use immutable commit URLs and content hashes; never fetch upstream HEAD or install unpinned libraries. Retain Kaggle's native Torch/CUDA/NumPy family.
- Prefer zero dataset/kernel-output mounts. Install only version differences into an isolated overlay with exact approved wheel hashes; retain matching native packages.
- Captured activations stay private on HF. Authenticate on Kaggle through HF_TOKEN (or the historical HF_token) Secrets; never include tokens in notebooks, archives, receipts or logs.
- Both earlier GPU instances were stopped by the user. Qualify the new direct-pip/source bootstrap on CPU before GPU replacements. The separately authorized long E2E retry remains in scope.
- For the current operator retry only, the user explicitly permits their temporary HF token in the private Kaggle submission instead of Secrets. Keep it out of public distributions, logs and receipts; do not broaden this exception to other credentials or jobs.
