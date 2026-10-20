# L1 offline LoRA and detailed GPU memory audit

Status: submitted to Kaggle as **H3 L1 Offline LoRA Memory**, version1; awaiting a physical dual-T4 allocation. No new T4 performance or memory result is claimed yet.

Notebook: https://www.kaggle.com/code/ocnsai/h3-l1-offline-lora-memory

Only L1 runs. S1/S2 and their downloaded outputs are preserved. The request retains864×480,362 frames, seed20261020,4 LCM/simple steps, CFG1, video/audio shifts12/2, both supplied references and the original prompt. The actual sequence is expected to match the previously observed46,447 tokens and is checked again at the attention consumer.

## Offline loading and gates

All checkpoint downloads and conversion occur on Kaggle. The base remains unchanged. The converter writes a new full checkpoint using Native FP32 sequential LoRA merging, strength/alpha normalization, full-input-width ConvRot INT8 requantization and the original storage dtype for dense weights. Quantization precedes TP sharding. Existing base rank-local checkpoint identities are not reused.

Offline merging/requantization is a different numerical model from online bypass LoRA. The reference is the Native offline merge, not byte equality to the previous online path.

Local gates passed on the5060 Ti: independent merged bytes for real QKV, attention-output, FC1, FC2 and BF16 refiner projections; a CUDA allocator/shape observer test with byte-identical observed versus unobserved outputs. Local PyTorch is2.12.1+cu132; the physical job retains the previously used2.11.0+cu130 stack. These local gates are correctness checks, not T4 performance evidence.

Before the full request, Kaggle will:

1. Verify every active merged target against an independent ordinary A/B merge-and-requantization formula, saving per-tensor hashes and a conversion receipt.
2. Check merged blocks0/49 at S33/S513 on physical TP2 against Native merged-block execution using Native attention/row arithmetic.
3. Check that the memory observer preserves those TP block outputs byte-for-byte and that no online LoRA adapters are attached.

The Native block gate does not qualify sparse attention as dense Native-equivalent. Existing accepted scalar attention remains selected for the actual L1 request. Row Partial/Selective B have no online adapter work to perform after a complete offline merge.

## Memory evidence

The production operators are unchanged. Memory observers are confined to benchmark entrypoints and enabled only for this job.

- Separate rank0, rank1 and parent-GPU0 records with process IDs and common wall-clock/monotonic timestamps.
- Bounded CUDA allocator histories: allocation address/size/stream/source stack, host free request, allocator-observed CUDA-completed free, segment allocation/release and OOM events.
- Complete per-command histories for TE, DiT evaluations and model handoffs; representative raw VAE invocations plus ordinary per-command peaks/NVML sampling for every invocation.
- Storage-deduplicated resident weights, retired weights awaiting safe release, model auxiliary buffers, parent VAE weights and block-input descriptors.
- Per-block and projection/attention/RoPE/quantizer phase markers. Tensor shapes/operators are observed for representative blocks0/24/49; allocator events cover all DiT blocks.
- Allocated, active, reserved, inactive-split bytes, allocation retries, driver free/total memory, sampled per-process NVML usage, RSS and cgroup pressure.
- Initial/final allocator segment snapshots for each chunk. Histories are capped at50,000 events per chunk and any truncation is reported explicitly. Tensor observations are bounded and never retain payload references.

`free_requested` does not mean CUDA reuse is already safe. `free_completed` does not mean memory has been returned to the driver: inactive segments can remain cached. Unattributed/library memory and pre-existing allocations are reported separately from known model weights; independent maxima are not added together.

Profiled generation time includes observer/export overhead. Conversion and correctness gates are outside the request. This run is not an uninstrumented latency comparison against the earlier online attempt, which failed.

## Artifacts and safety

The6 GiB/rank weight cap,9 GiB activation reserve,96 MiB scalar-score workspace cap, owned-FP32 FC2 quantizer and14.5 GiB denoising allocated-peak acceptance bound remain in place. These are admission/acceptance controls, not a guarantee against fragmentation or non-PyTorch allocations. Failures remain failures.

Raw data is under `ref2va_e2e_results/L1/gpu_memory/{rank0,rank1,parent_gpu0}/`. Generated reports contain reconstructed peaks, static/input/temporary categories, largest temporary lifetimes and per-phase breakdowns. A readable sampled memory timeline is produced during local postprocessing. Full raw tensor dumps are not enabled; compact decode inputs are saved only if recovery is needed after an inference/export failure.

The independent local collector downloads the completed MP4/WAV, receipts and bounded memory artifacts to `semantic_results/l1_offline_memory_v1/`. It launches no GPU jobs. The uploaded source binding is preserved in `kaggle_l1_offline_memory/source_binding.json`.
