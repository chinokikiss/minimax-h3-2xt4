# H3 quality-first attention experiments — 2026-10-14

Primary objective: preserve generation quality while improving practical throughput on physical T4 SM75. Operator differences, omitted mass and denoising divergence are diagnostic measurements, never standalone quality rejection criteria. All conclusions are scoped to completed stages.

## Audit and reusable evidence

`D:/attention_eval/t4_redo_20261013/DIAGNOSTIC_REPORT.md` contains actual T4 evidence for four selected heads: TE10/head40, DiT49/step9/head10, DiT25/step9/head14 and VAE0/head5. Exact activations, full sequence lengths, source hashes and transferred-file hashes are retained. These biased diagnostic heads do not estimate population accuracy or generation failure rates. Existing 170 native captures span five layers in each component, three DiT calls and all 30 VAE decode passes. No new full-tensor capture is needed for operator exploration.

Reusable implementations: baseline INT8 QK/PV, dense preparation, Group32 and scalar pooling. Reusable profiler evidence: Systems CUDA/NVTX/OSRT, launch resources and telemetry. The scalar-pool change preserves the approximation while accelerating the selected single-head kernel. High numerical difference on that head establishes an approximation difference, not downstream quality failure. Existing local SM120 videos may help prepare the review workflow but do not qualify a T4 candidate.

Nsight Compute returned ERR_NVGPUCTRPERM. Achieved occupancy, stalls and counter-based bandwidth remain unavailable. Shared-memory launch attributes are resource bounds only. Fused-kernel QK/softmax/PV instruction-time attribution is not available from Systems; phase markers can separate preparation/routing/fused execution, not split fused instructions.

## Hypotheses and controlled matrix

| Hypothesis | Variables changed | Controls / measurements |
|---|---|---|
| H1: K centering and outlier scale granularity explain difficult QK heads | Sol centroid mean versus token mean versus no centering; Q32/K64, Q128/K64, token/token and Group32 | Same captures, mask, routing, V and FP16 PV. Separate FP16-source and FP32-source quantization. Logit/probability/outlier diagnostics; independent implementation reference; actual T4 timings |
| H2: approximate correction carries useful conditional information beyond dropping | Drop, original pooling, scalar pooling; later two sub-block summaries only if justified | Exactly the same routes; inspect block LSE, omitted/retained mass, conditional weighted V and oracle ablations. Independent online normalization. Quality, not L2, decides utility |
| H3: sparse work savings are hidden by pooled arithmetic and preparation/launches | Scalar correction, tau 0/0.5/1, top-k budget 0.25/0.5/0.75 | Measure actual density after mandatory conditioning leaves. API, raw kernel, preparation and routing separately; Systems/telemetry; include costs in candidate decisions |
| H4: a simple layer/stage policy preserves quality more efficiently | All DiT layers; first/last 10 layers dense; first/last 4 of 20 calls dense | Compare one policy change at a time. No layer declared sensitive from operator L2 alone. Policy advancement requires paired quality evidence |
| H5: component choices have different quality/performance tradeoffs | TE dense INT8, DiT sparse INT8, VAE dense INT8/PV8 | First change one component with other components at native attention; then combine validated candidates. Include native FP32 attention and FP16 SDPA references |
| H6: operator gains survive deployment overhead | Two-rank TP, 1/2/4 simultaneous arrivals, serialized admission versus overlap where supported | Real 2×T4 latency, completion throughput, p50/p95, queue delay, peak allocated/reserved/NVML memory, CPU/GPU timelines, H2D/D2H/P2P and NCCL. Do not call queueing parallel GPU execution |

Sage comparison is an SM75 TileLang implementation of Sage-inspired mathematics, not upstream Sage performance. Upstream smoothing is token mean; upstream block scales reduce over both tokens and channels, Q block128 or warp32/16 and K block64. Our dense kernel consumes repeated token scales and applies sqrt(D)/log-base conversion itself. FP32 centering and half-away rounding are explicit. FP16 subtraction rounding versus fused FP32 subtraction must be labeled, not conflated. Preserve upstream source digests and the selected API contract.

## Workloads

Anchor: exact `bench_assets/tokyo_ref2va.json`, reference JPEG byte identity preserved, 640×352, 90 frames, 24fps, 20 steps, CFG1, res_multistep/simple, seed20261013, no LoRA/negative/audio/video reference. Native and candidate runs must independently encode the reference and prompt when TE changes; freezing baseline conditioning would hide TE degradation. Reuse identical saved noise, sigmas and initial reference latents where they are unaffected by the tested component.

Small screening: anchor plus seed20261014; native, FP16 attention, Sage32 dense DiT, scalar-pool DiT tau0, and drop-only with the same routes. TE/VAE stay native initially to isolate DiT. Then separate TE and VAE screens. Run full 20 steps and full 90-frame decode; four-step previews cannot establish production quality.

Expanded validation: three prompt families (forward street movement, locked-camera street activity, slow lateral street movement), seeds20261013–20261017 and both 640×352 and 512×512 at 90 frames. Preserve settings within each paired comparison. These prompts initially share the supplied Tokyo reference; report this limitation. Add at least three distinct reference assets before broad claims about identity/appearance robustness. Five seeds are exploratory, not proof of rare-failure absence. Use baseline native quality as a control; an optimization cannot be blamed for a defect already present in both paired outputs.

## Quality metrics and acceptance

Pre-register metrics before seeing candidate results. Video embeddings versus reference and text (pinned DINOv2/CLIP checkpoints, spatial crop/resize protocol recorded) measure appearance/semantic consistency, not identity guarantees. Use optical-flow-warp residual and flow discontinuities for temporal diagnostics, with motion magnitude reported to detect frozen-video gaming. Measure frame embedding drift and prompt/reference similarity separately before/after 2 seconds. Record video/audio duration, clipping, loudness, audibility and blind review of traffic/motion sound consistency. Automatic AV metrics require a separately validated semantic model for non-speech scenes; lip-sync metrics are unsuitable here. Diversity across seeds is reported independently of reference fidelity.

Blinded paired review: anonymized randomized A/B clips with sound and reference/prompt visible, order balanced, reviewers unaware of attention mode. Score reference preservation, prompt adherence, temporal/motion quality, artifacts, AV consistency and overall quality 1–5; mark severe failures. At least two independent reviewers where feasible, retain disagreements. Never describe automated or assistant assessment as human review.

Provisional screening margins, fixed before ratings: overall/reference/temporal paired score decrease no worse than 0.25 on the 1–5 scale, no reproducible new severe failure in screening. Final noninferiority requires paired 95% intervals within these margins, and a severe-failure increase upper interval ≤10 percentage points. Small samples may be inconclusive; report that rather than passing them. Show sensitivity to margins and every metric, not an opaque weighted score. Automatic metrics are supporting evidence; no universal embedding threshold is assumed. Report means/medians/paired deltas, bootstrap intervals resampling at prompt/reference/seed units, failure counts and Wilson intervals. Do not treat frames as independent samples.

Pareto frontier uses only candidates with actual generation quality evidence and full-pipeline unprofiled timings on the same hardware/workloads. Screening and profiling results are separate. No current candidate has a validated quality/throughput Pareto position.

## Execution and reproducibility

1. CPU contract sanity and T4 independent reference checks validate implementations, including causal masking, GQA mapping and ragged tokens. A kernel-reference tolerance can block an implementation bug; candidate-native error cannot reject its quality.
2. First T4 jobs reuse the two tiny private datasets, full N353/N6814/N1797. At least one-second/five-call warmup, 15 samples per operation, separate compile/first-call/load, raw timing arrays and telemetry. Use fresh repeated processes for finalists and randomized configuration order to reduce clock/order bias. Four selected heads are labeled diagnostic scope. Full-head and TP rank-head measurements follow dataset availability.
3. Qualify a native T4 generation baseline and candidate integration before spending budget on quality sweeps. Run small paired screens.
4. Advance candidates by generation evidence, expand prompts/seeds/references and prepare blinded review packets.
5. Warm full 2×T4 pipeline (loading/compilation separately), repeat requests, compare sequential/concurrent arrivals, trace finalists in separate profiled runs. Report profiler overhead rather than subtracting guessed corrections.

Private jobs: `chinokiki9/h3-quality-first-t4-quantization` and `chinokiki9/h3-quality-first-t4-pooling`. Builders: `build_quality_first_t4_job.py [--sparse]`; push with `kaggle kernels push -p <folder> --accelerator NvidiaTeslaT4 -t 7200`. Use pinned TileLang0.1.15 and Torch2.11+cu128 on T4. Do not blindly reuse a cu130 bootstrap with a compiler that drops SM75 support.

Record hardware/driver, packages, checkpoint SHA256 and pinned HF revision, ComfyUI source digest or real commit, workspace base commit plus dirty-file digests, exact commands and settings. User's ComfyUI-master is a source directory, not a verified Git commit; use source hashes. Never silently substitute the older pinned checkout.

## Missing prerequisites and bounded data

T4 allocation and Systems are available; Compute counters are provider-blocked. The authorized 16.27GB upload stopped with resumable state, so full-head remote replay is not yet available. Reuse completed private diagnostic slices immediately. Preserve original local captures; resume transfer without new copies. Do not let an obsolete numerical-only suite make acceptance decisions.

Candidate full-pipeline TP integration is not yet qualified. Remote models total 51.53GB; verify pinned public-download availability, disk/RAM headroom, dependencies and exact ComfyUI-master portability in a prerequisite job. Actual T4 runs must expose any download/OOM/source incompatibility explicitly. Additional reference assets, validated metric models and independent human ratings are not currently available.

No further full Q/K/V or per-layer dumps. Retain per-head summary JSON/CSV, exact configs, low-volume trajectory summaries, compressed final video/audio and a few contact sheets. At most two finalist Systems traces per stage/workload until a specific unresolved bottleneck justifies more. Preserve existing raw measurements; avoid duplicate exports/checkpoint copies. Stage status and final limitations distinguish completed operator work from pending generation validation.
