# H3 runtime preservation

- Preserve validated source, attention API, operators and dependency versions by default. Do not update upstream code or dependency versions as part of adaptation or testing.
- Local execution uses `C:\ProgramData\miniconda3\envs\comfyui\python.exe`. The experimental `D:\h3_tp2_legacy_env` is historical, not the default.
- Retain the legacy attention API in the local `ComfyUI` checkout (`d49e888586dd8ae012c0667b33466b815fee07f7`). Do not replace it with the newer sibling `ComfyUI-master` implementation.
- `runtime/local_conda.json` is the user environment snapshot. Kaggle must use the frozen local source archive and exact package versions; never clone upstream HEAD, run git pull, install unpinned dependencies, or silently substitute platform versions.
- Windows/Linux wheel or ABI differences are blockers to report and validate explicitly. Historical T4 Torch 2.11 / TileLang 0.1.15 measurements do not qualify the current conda Torch 2.12.1 / TileLang 0.1.13 stack on Linux.
- Older Kaggle builders/notebooks are historical and may contain divergent dependency setup. Do not submit them directly. Use the frozen setup preflight first, and replace their setup before any new submission.
- Preserve historical submitted sources, results and environments. Do not run a full H3 pipeline or consume Kaggle GPU time for environment setup checks.
