# H3 runtime preservation

- Preserve validated source, attention API, operators and dependency versions by default. Do not update upstream code or dependency versions as part of adaptation or testing.
- Local execution uses `C:\ProgramData\miniconda3\envs\comfyui\python.exe`. The experimental `D:\h3_tp2_legacy_env` is historical, not the default.
- Retain the legacy attention API in the local `ComfyUI` checkout (`d49e888586dd8ae012c0667b33466b815fee07f7`). Do not replace it with the newer sibling `ComfyUI-master` implementation.
- `runtime/local_conda.json` is the user environment snapshot. Kaggle must use the frozen local source archive and exact package versions; never clone upstream HEAD, run git pull, install unpinned dependencies, or silently substitute platform versions.
- Windows/Linux wheel or ABI differences are blockers to report and validate explicitly. Historical T4 Torch 2.11 / TileLang 0.1.15 measurements do not qualify the current conda Torch 2.12.1 / TileLang 0.1.13 stack on Linux.
- Older Kaggle builders/notebooks are historical and may contain divergent dependency setup. Do not submit them directly. Use the frozen setup preflight first, and replace their setup before any new submission.
- Preserve historical submitted sources, results and environments. Do not run a full H3 pipeline or consume Kaggle GPU time for environment setup checks.

## Authorized trusted-bake default — 2026-10-29

- The user promotes trusted DiT bake loading to the runtime/loader default: `verify_dit_weights=False`. Keep explicit `True` available. TE verification, metadata/schema/file-signature guards and download/bake-time validation remain unchanged.
- Low-level bundle readers retain verified defaults for offline baking and integrity tools. Inference receives the runtime flag explicitly; A/B experiments must select both arms explicitly.
- The user authorizes one regular Turbo LoRA long E2E challenge at approximately 1 MP and 10 seconds, with persisted offline TP2 baking. This is not PTD. Preserve the existing source/core pins, CPU-before-GPU qualification and no-status-polling policy.

## Authorized 0.9 MP resolution retry — 2026-10-29

- After the 1 MP GPU OOM, the user explicitly requests one 0.9 MP resubmission and skipping the CPU smoke test. Reuse the successful prior CPU qualification for the unchanged runtime/core. No new CPU smoke instance is required for this resolution retry.
- Preserve the qualified source capsule and core setup; record a content-hashed benchmark-only canvas/export-validation overlay separately. Do not fabricate a fresh CPU receipt for that overlay or change production operators, dependency versions, model/LoRA, prompts, references, seed or sampler settings.
- Retain failed-job artifacts and receipts. No status/live-log polling or automatic duplicate submission.

## Authorized 0.85 MP resolution retry — 2026-10-29

- After the 0.9 MP OOM, the user requests another direct GPU retry, finally selecting approximately 0.85 MP (1216×704). Skip a new CPU smoke test and reuse the successful unchanged runtime/core qualification with the separately hashed benchmark contract overlay.
- Keep the 10-second regular Turbo workload, offline bake, model/LoRA, references, seed, sampler and dependency pins unchanged. Retain prior failure artifacts and submit only one GPU job; no status/live-log polling.

## Authorized 0.8 MP resolution retry — 2026-10-29

- After reporting a 0.85 MP failure, the user initially requested one 0.8 MP direct GPU retry (1184×672) and skipping the CPU smoke test. This request was superseded before GPU submission: final artifacts showed HF HTTP 503, not OOM, and the user requested 0.85 MP again.
- Preserve the ten-second regular Turbo workload, offline LoRA bake, inputs, sampler, source and core pins. Retain the previous failure artifacts and receipts. No status/live-log polling or automatic duplicate submission.

## Authorized 0.85 MP download-failure retry — 2026-10-29

- The user requests retrying 0.85 MP after the collected HTTP 503 weight-download failure. No 0.8 MP job was submitted. Keep skipping CPU smoke, reusing the unchanged qualified runtime/core.
- Benchmark-only download handling may retry transient HTTP/transport failures up to three attempts with bounded backoff, preserve partial files, and retain exact revisions, byte sizes and SHA256 validation. Record this additional benchmark overlay separately; production operators and dependency pins remain unchanged.
- Preserve the prior failed version and submit one new version only. No Kaggle status/live-log polling.

## Authorized seed / dataset CPU batch — 2026-10-30

- The user requests exactly one GPU E2E repeat with a different seed and two CPU studies. Use seed 20261021 for the same 1216×704 ten-second regular Turbo workload; preserve TE Kitchen causal, DiT, decoder, LoRA, references and all other settings. Reuse the previously CPU-qualified and physical-T4-executed runtime/core with separately hashed benchmark overlays; do not allocate an extra H3 environment-smoke instance.
- The read-only CPU study may mount six public opaque-binary datasets and measure 1/2/4/6 datasets together, with 1–64 readers, bounded reusable buffers, RSS guards and cache caveats. This is not authorization to remount model datasets in the GPU run or to claim CPU-storage results are T4 results.
- The upload CPU study may inspect native KaggleHub/CLI/API and attempt small private synthetic dataset uploads using only notebook-native authentication or attached Secrets. Do not embed local Kaggle credentials, install unpinned clients, upload H3 captures/weights or publish probe datasets. Report missing authentication as a blocker.
- Submit each job once, retain durable receipts and bounded watchdogs, and use final artifacts only. No status/list/live-log polling or automatic resubmission.

## Explicit upload retry and read-study collection — 2026-10-30

- The user's "Submit and check H3 Dataset Read Sweep CPU 20261030" authorizes exactly one corrected upload-study CPU retry and final read-study artifact collection. Upload version 2 was submitted; it does not authorize automatic further retries or another GPU run.
- The user subsequently reports upload-study success; collect its final artifacts once and distinguish accepted upload requests from completed processing/readback proof. Existing artifact-transfer retry permission remains in force.

## Authorized TE O_DIRECT profiling — 2026-10-30

- The user requests exactly one additional GPU instance to compare buffered TE weight I/O with O_DIRECT on real T4s, including I/O speed and page-cache/reclaim behavior. The follow-up explicitly requires hash-verification profiling too.
- Run TE only, using the previous 0.85 MP prompt/reference presentation; do not run DiT, VAE encoders/decoders, sampling or LoRA baking. Keep the qualified frozen runtime/core and production defaults unchanged, and retain all TE file/tensor verification.
- Reuse the matching already-passed CPU source/core qualification. Bind the new benchmark-only adapter/observers separately, test them locally and run a Linux alignment/byte/cache-bypass gate before TE execution. This authorizes no extra CPU instance or environment-only GPU test.
- Use one private zero-mount submission with bounded execution, durable receipts and final artifacts only. O_DIRECT must not silently fall back to buffered weight reads. Do not alter global drop_caches, poll status/live logs, resubmit automatically or embed credentials.

## Explicit TE O_DIRECT retry — 2026-10-30

- After the frozen-helper import failure, the user explicitly answered yes to submitting the prepared corrected version 2. Submit that source-bound private T4 retry once, retaining the same workload, frozen runtime/core, operators, TE verification and measurement plan.
- Version 2 was accepted. This authorization does not permit another automatic resubmission, status/live-log polling or an extra CPU instance. Collect final artifacts after the user reports completion; bounded transfer retries remain authorized.

## Authorized Kaggle exception — 2026-10-21

- The user explicitly permits Kaggle's native Torch rather than reproducing the local Windows Torch/CUDA package stack. Keep its matching native Python/NumPy/Torchvision/Torchaudio/CUDA environment and record actual versions; do not modify the local conda environment or original snapshot.
- Core libraries whose APIs/operators H3 relies on remain pinned, especially comfy-kitchen 0.2.37 and TileLang 0.1.13. Preserve the frozen ComfyUI source and legacy attention API. Validate core imports/signatures on CPU before operator tests; do not upgrade upstream source or core libraries to resolve compatibility problems.
- Do not poll Kaggle status/list endpoints or live logs, including through an API or browser workaround. Minimize interactions: batch submissions, use bounded runs, and perform one logical final-output collection per batch; artifact-download retries are permitted as clarified below.

## Authorized artifact-download retries — 2026-10-29

- Artifact downloads may be retried after network/transport failures or incomplete downloads. The user has explicitly authorized these retries; do not require renewed permission for each attempt.
- Retrying/resuming final artifact collection is distinct from polling job status or live logs. Keep status/list/live-log polling prohibited; the output-file enumeration needed to download final artifacts is allowed.
- Preserve partial downloads, prior errors and receipts. Resume missing files where possible, verify source identity and content hashes, and record attempts. Do not resubmit a GPU job just because artifact transfer failed.
- Keep retries bounded and space repeated failures apart to minimize Kaggle interactions. Report a persistent blocker clearly rather than retrying indefinitely.

## Authorized delivery change — 2026-10-22

- The user authorizes GitHub/HF delivery instead of large Kaggle datasets or mounted notebook outputs, and direct pip installation instead of shipping wheel archives.
- Preserve the frozen source and exact core pins. Use immutable commit URLs and content hashes; never fetch upstream HEAD or install unpinned libraries. Retain Kaggle's native Torch/CUDA/NumPy family.
- Prefer zero dataset/kernel-output mounts. Install only version differences into an isolated overlay with exact approved wheel hashes; retain matching native packages.
- Captured activations stay private on HF. Authenticate on Kaggle through HF_TOKEN (or the historical HF_token) Secrets; never include tokens in notebooks, archives, receipts or logs.
- Both earlier GPU instances were stopped by the user. Qualify the new direct-pip/source bootstrap on CPU before GPU replacements. The separately authorized long E2E retry remains in scope.
- For the current operator retry only, the user explicitly permits their temporary HF token in the private Kaggle submission instead of Secrets. Keep it out of public distributions, logs and receipts; do not broaden this exception to other credentials or jobs.

## Authorized Temporal VAE benchmark exception — 2026-10-26

- The user separately authorizes the supplied HF token inside this private Query-versus-Temporal Video VAE benchmark notebook only, including its CPU input-authentication preflight. Keep it out of public source archives, logs and receipts. Do not extend this exception to another job or credential.
- Benchmark only the Video decoder with the saved real H3 latents; no TE/DiT sampling or Audio VAE parallelism. Preserve the default Query mode while Temporal is qualified.

## Validated Temporal promotion — 2026-10-27

- After the physical two-T4 comparison, the user explicitly requests Temporal as the Video VAE default. `auto` follows Temporal; explicit Query and Native remain available. Audio stays Native single-GPU.
- The new TE attention/communication experiment is operator/block level, without DiT sampling or full H3 generation. Its public trained shards need no HF credential; the earlier private-token exception is not extended.
