# Repository tools

Run Python commands **from the repository root**, using the existing conda
environment. The 95 former root-level Python files moved here on2026-10-25;
the [path map](path_migration.json) records every old and new location.

| Folder | Purpose |
|---|---|
| [`runtime/`](runtime/) | Four maintained commands: launch, environment verification, frozen snapshot export and workflow conversion |
| [`experiments/`](experiments/) | Historical notebook builders, collectors and analysis scripts, preserved for experiment provenance |
| [`../benchmarks/`](../benchmarks/README.md) | Current operator experiments, profiling, CPU qualification and tests |

```powershell
& C:\ProgramData\miniconda3\envs\comfyui\python.exe -m tools.runtime.launch_h3_legacy --preview --check
& C:\ProgramData\miniconda3\envs\comfyui\python.exe -m tools.runtime.build_comfy_tp2_workflow --help
```

The root `launch_h3_legacy.cmd` remains the convenient local launcher and uses
the same exact interpreter. Other commands use `python -m tools.runtime.NAME`
or `python -m tools.experiments.NAME`, omitting `.py`.

Historical experiment tools can depend on local evidence and older dependency
setup. They are **not approved Kaggle submission templates**. Use the frozen
CPU preflight and current exact core pins before preparing a new experiment;
do not run their old polling helpers. Submitted archives and their immutable
commit URLs remain unchanged. Dated reports and evidence JSON retain historical
source paths/hashes; consult the path map for the current checkout.

The branch cleanup keeps `main` and preserves each former experiment head as
an `archive/20261025/...` tag. This removes branch-list clutter without losing
the exact frozen sources used by the recorded experiments.
