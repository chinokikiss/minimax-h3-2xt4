# Generation quality: existing V4 screen

Measured: eight complete90-frame640×352 clips and3.75-second stereo audio on physical T4. One Tokyo reference/prompt, two paired seeds. Native, fused Sage32 DiT, scalar tau0 DiT and dropping; TE/VAEs native. All outputs decode; exact checkpoint/source manifests are preserved.
Automated CLIP/DINO/flow diagnostics completed for all eight runs. User review feedback: "Not obvious / hard to tell the difference." This is informal qualitative evidence; per-pair coverage, numeric ratings and listening conditions were not specified. No statistical quality noninferiority, identity guarantee or absence of severe failures is inferred.

| Run | Reference DINO | Prompt CLIP | Consecutive DINO | Post2s prompt CLIP | Post2s flow residual |
|---|---:|---:|---:|---:|---:|
| native_seed20261013 | 0.9598 | 0.3167 | 0.9985 | 0.3148 | 0.01076 |
| sage32_dit_seed20261013 | 0.9608 | 0.3147 | 0.9980 | 0.3158 | 0.01176 |
| scalar_tau0_dit_seed20261013 | 0.9439 | 0.3224 | 0.9952 | 0.3231 | 0.01355 |
| drop_tau0_dit_seed20261013 | 0.9469 | 0.3224 | 0.9959 | 0.3245 | 0.01384 |
| native_seed20261014 | 0.9389 | 0.3089 | 0.9982 | 0.3106 | 0.01085 |
| sage32_dit_seed20261014 | 0.9377 | 0.3086 | 0.9983 | 0.3133 | 0.01074 |
| scalar_tau0_dit_seed20261014 | 0.9360 | 0.3182 | 0.9966 | 0.3187 | 0.01337 |
| drop_tau0_dit_seed20261014 | 0.9324 | 0.3180 | 0.9966 | 0.3202 | 0.01349 |

These are descriptive cosine/flow proxies, not calibrated perceptual quality measures. CLIP truncates the long prompt; processor crops can omit reference details. High consecutive similarity can reflect low motion. Two seeds are two paired samples; frames do not create independent replication. No definitive scalar-pooling benefit or severe-failure rate can be established from these proxies.
Blinded review packet: D:\attention_eval\quality_first_20261014\generation_v4\blinded_review\review.html. All six pairs retain blank numeric ratings; the user's qualitative response is preserved separately, without invented scores.
P3 next screen submitted as chinokiki9/h3-attention-quality-challenge version1 on actual2×T4. Full-head conformance/performance selected fused Group32 preparation with the unchanged legacy Group32 MMA. Two fixed-workload Group32 clips reuse V4 native conditioning/noise/sigmas. Eight new512×512 challenging motion/reference clips compare native, unchanged fused Sage32, selected Group32 and Sol tau0 dropping across two paired seeds. Candidate order is randomized within each challenging seed. TE/reference inputs are prepared once per seed; all four candidates share their exact hashes. These cached-input DiT comparisons are not full-pipeline TE/request latency measurements.
The new screen requests per-layer/step attention events, input strides, loading/JIT/qualification timings, step9 CPU profiles and Nsight CUDA/NVTX/OSRT timelines. First-call initialization and profiled-step observer overhead must be excluded from steady-state interpretations. No new large QKV dumps are requested. Job submission alone is not evidence of successful generation; verify the collected results below when available. Independent quality review remains required.
Audio export integrity is checked for V4; there is no validated audio-consistency score or completed listening review. Selective sensitivity is not defined using operator L2 and selective quantization remains deferred.
Latest saved job state: quality: evidence_collected, KernelWorkerStatus.COMPLETE; sol: evidence_collected, KernelWorkerStatus.COMPLETE.

## Collected next-generation screen

Provider artifact status: screen_generated; 10 completed run records. Evidence: D:\attention_eval\quality_constrained_20261015\quality_challenge_v1_snapshot\generation_screen.
