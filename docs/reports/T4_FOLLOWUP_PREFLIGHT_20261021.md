# T4 follow-up execution attempt — 2026-10-21

> Historical strict-profile attempt. The user subsequently authorized Kaggle's
> native Torch while preserving Kitchen and other core pins. That separate
> profile and its T4 evidence are recorded in
> [the native-Torch follow-up](T4_NATIVE_TORCH_VALIDATION_20261021.md). The four
> CUDA constraint proposals below were not applied to the original lock.

**The current T4 experiments remain blocked before GPU submission.** Two private CPU-only Kaggle jobs ran. No T4 GPU job, full H3 generation, or replacement SM120 benchmark was submitted. The retained historical T4 results are unchanged.

## Access and interaction policy

The new OAuth login was verified active for `pamcks`. This account initially had no notebooks and could not access the older private notebooks owned by other accounts; the earlier `Not found` list result was not proof of failed authentication.

New private resources:

- [Unchanged frozen runtime dataset](https://www.kaggle.com/datasets/pamcks/h3-frozen-runtime-20261021)
- [CPU metadata preflight, version 1](https://www.kaggle.com/code/pamcks/h3-frozen-linux-cpu-preflight) — use the retained version-1 download/local evidence, not an inferred latest-version result
- [CPU exact-interpreter preparation, version 1](https://www.kaggle.com/code/pamcks/h3-frozen-linux-cpu-candidate)

After the user's instruction to minimize interactions, CLI status/list polling and live-log monitoring stopped. The second CPU job's evidence was retrieved in one version-bound collection, rather than checking its status repeatedly. Do not substitute API/browser polling for CLI polling. Future work should batch submissions, use bounded execution, and collect final artifacts once per batch. Do not launch the old workflow runner's polling loop.

## CPU evidence

The first job verified the unchanged frozen source archive and fetched metadata for all 94 locked package versions, targeting Linux x86_64 CPython 3.11 wheels. It recorded:

- Kaggle's default interpreter: **3.13.15**, rather than the required 3.11.
- Seven Linux-only Torch requirements absent from the 94-package Windows dependency closure: CUDA Toolkit, CUDA bindings, cuDNN, cuSPARSELt, NCCL, NVSHMEM and Triton.
- No missing wheel candidate in that metadata screen. This is not a resolver, import, ABI, compiler or GPU qualification.

The second job successfully bootstrapped a separately isolated **CPython 3.11.15** on the CPU host, using a fixed publisher release URL and SHA256. It then ran pip's dry-run resolver with **both unchanged frozen requirements and unchanged full constraints**. It did not reach package installation, wheelhouse export, or import/ABI qualification.

Pip returned `ResolutionImpossible`. Its explicit first conflict was:

```text
cuda-toolkit 13.2.1 depends on nvidia-cublas==13.4.0.1.*
The user requested (constraint) nvidia-cublas==13.6.0.2
```

The distinction matters: the 94-package manifest is the selected H3 dependency closure; `constraints-local-conda.txt` also preserves other installed local packages. Checking only those 94 versions misses conflicting CUDA constraints that become active on Linux. We kept the full constraints and rejected the resolution rather than ignoring those pins or installing with `--no-deps` to bypass the conflict.

An additional read-only audit of CUDA Toolkit 13.2.1 publisher metadata identified four distinct incompatible pins. Only cuBLAS is the first conflict printed by the actual resolver; the other three are metadata-proven intersections, not additional completed resolver runs.

| Package | Frozen local constraint | Linux Torch → CUDA Toolkit 13.2.1 requirement | Exact proposed exception |
|---|---:|---:|---:|
| `nvidia-cublas` | 13.6.0.2 | `==13.4.0.1.*` | 13.4.0.1 |
| `nvidia-cuda-nvrtc` | 13.3.33 | `==13.2.78.*` | 13.2.78 |
| `nvidia-cuda-runtime` | 13.3.29 | `==13.2.75.*` | 13.2.75 |
| `nvidia-curand` | 10.4.3.29 | `==10.4.2.55.*` | 10.4.2.55 |

These exceptions are **proposed, not applied or approved**. Publisher wheel URLs/hashes and the original constraints digest are in [the machine-readable proposal](../data/t4_linux_constraint_proposal_20261021.json). Applying them would be a deliberate Linux-platform exception to the freeze, not an identical dependency environment. It would preserve ComfyUI `d49e888586dd8ae012c0667b33466b815fee07f7`, Torch 2.12.1+cu132, Kitchen 0.2.37, TileLang 0.1.13, and all 94 locked H3 package versions.

The exception would not itself qualify anything. The complete added Linux dependency closure still needs exact resolution, hashes, CPU imports/ABI checks and compiler preparation before any GPU job. Further conflicts may surface after these four are resolved. The historical TileLang 0.1.15/nvcc 12.8 production gate has not been relaxed.

## Work prepared locally

The following are new, explicit benchmark tools. They are not runtime imports and have not executed on T4:

- `build_t4_followup_preflight.py` creates fresh private CPU-only submissions and preserves historical source snapshots.
- `run_frozen_linux_preflight.py` verifies the frozen archive and reads publisher metadata without installing packages.
- `prepare_frozen_linux_candidate.py` uses exact CPython 3.11.15, verifies the unchanged base versions and dependency reachability, and would install only hashed resolved wheels if the resolver succeeds. The current run stopped at resolution.
- `run_h3_physical_fp16_flow.py` prepares physical NCCL TE/DiT component comparisons against the existing Native FP32 activation flow, retaining INT8 weights/operators and mixed dtype islands. It skips BF16 execution on SM75, traces optimized preparation/fallback calls, separates first-call/loading overhead from repeated steady-state timing, and optionally exports PyTorch profiler traces. This is a prepared entry point, not a numerical/performance result or Nsight evidence.
- `prepare_t4_followup_fixtures.py` exported unmodified trained TE block 0, DiT blocks 0/24/49, the corresponding LoRA tensors, refiner-attention weights, and the retained first-block input. The approximately 2.1 GiB component fixture set remains local; it was not uploaded. No full H3 checkpoints were uploaded.

Fixture limitations are explicit: TE block inputs would be seeded synthetic inputs; DiT layers 24/49 would reuse the retained first-block input as a stress test, not their own native trajectory. The older Tokyo TE/DiT QKV capture paths named in historical manifests are not present at those retained local locations. Do not relabel these component inputs as a fresh complete real-distribution study.

Local verification: seven focused frozen-setup/platform-closure unit tests passed; the new scripts compile and the physical harness CLI parses. Production runtime source, package constraints, operator defaults and dependency gates were not changed during this attempt.

## Retained evidence and remaining status

Local root: `D:\attention_eval\t4_followup_20261021`.

- `cpu_output/preflight_results/preflight.json`: first CPU host/metadata/source receipt.
- `candidate_output/linux_candidate/candidate_receipt.json`: second CPU result, `status=blocked`.
- `candidate_output/linux_candidate/resolution.log`: original resolver failure.
- `candidate_output/download_manifest.json`: version-1 evidence filenames, sizes and SHA256 digests.
- `cuda_constraint_conflicts.json`: publisher metadata analysis of the active CUDA extras.
- `linux_candidate_job/submission_binding.json`: submitted CPU source binding.
- `component_fixtures/fixture_manifest.json`: selected trained tensor identities and input provenance.

All T01–T09 dimensions in [the backlog](T4_VALIDATION_BACKLOG_20261021.md) remain uncompleted on the current frozen target stack. T09 is still a CPU/Linux gate and should run without a GPU. Full pipeline, sparse semantic acceptance and concurrency remain deferred under the existing scope.

The applicable [repository instruction](../../AGENTS.md) requires Kaggle to use “exact package versions” and says to “never … silently substitute platform versions.” Proceeding requires an explicit decision on the four CUDA constraint exceptions above; the package/version preservation rule was not bypassed.
