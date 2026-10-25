# Main synchronization and script layout — 2026-10-25

This delivery synchronizes the current H3 runtime, integrated optimizations,
experiment tools and reports to `main`, including the completed
[0.5MP dual-T4 E2E](H3_L1_TURBO_05MP_E2E_20261024.md). Its generation quality
remains pending review. The layout change itself is CPU-qualified; it does not
claim a new GPU performance result.

Publication and cleanup completed: **16 remote branches and12 local branches
deleted; only `main` remains**. All16 original heads are verified on GitHub as
archive tags. The runtime/layout delivery commit is `e2cea4a`; this follow-up
records the completed cleanup.

## Layout

All95 former root Python scripts now live under `tools/`:

| Location | Contents |
|---|---|
| `tools/runtime/` | Launch, environment verification, frozen runtime export and workflow conversion (4 commands) |
| `tools/experiments/` | Historical builders, collectors and analysis scripts (91 files) |
| `benchmarks/` | Existing operator experiments, profiler tools and tests |
| `patches/`, `custom_nodes/` | Runtime implementation and ComfyUI integration |
| `docs/` | Guides, reports, design notes and aggregate evidence |

Use `python -m tools.runtime.NAME` or `python -m tools.experiments.NAME` from
the repository root. The small root `launch_h3_legacy.cmd` remains supported
and works from another current directory. Its interpreter is unchanged.
Imports, repository-root calculations, source packaging and active setup
instructions were updated. The complete [old-to-new path map](../../tools/path_migration.json)
preserves the correspondence with dated reports and source hashes. Original
submitted archives are not rewritten.

## Branch retention

The16 remote experiment branches differ from the previous `main` only by
frozen source archives. Each head is retained under an
`archive/20261025/<former-branch-name>` tag before deleting its branch.
`main` is the only retained development branch. This preserves the exact
commits used by immutable GitHub delivery URLs without putting hundreds of
megabytes of old capsules into the current source tree.

Publication is ordered: update `main` and archive tags, verify the remote
references, then delete the exact recorded branch heads. The
[machine-readable record](../data/repository_sync_20261025.json) records the
mapping and final cleanup state. Local pre-delivery file and Git bundles are
retained in `D:/attention_eval/repo_sync_20261025/`.

## Validation

- 163 CPU review tests passed, zero failures/errors/skips.
- 6 migrated analysis tests passed.
- 562 Python source files parsed successfully.
- The root launcher passed `--preview --check` when invoked from a different
  directory; the frozen ComfyUI revision and legacy Video attention API passed.
- Workflow conversion's module CLI/help passed. Runtime/dependency lock files
  and the ComfyUI checkout remain unchanged.
- No Kaggle interaction or full-pipeline execution was performed for this sync.

Historical builders may contain old setup or polling routines. Their relocation
does not make them current submission templates; the preserved environment
and final-only Kaggle collection rules still apply.
