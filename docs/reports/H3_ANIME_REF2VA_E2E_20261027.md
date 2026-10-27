# Eight-step anime REF2VA E2E — 2026-10-27

Status: **generation completed on two physical SM75 Tesla T4 GPUs** in
[GPU E2E version 1](https://www.kaggle.com/code/pamcks/h3-anime-ref2va-8step-t4-20261027).
All eight sampling steps, both decoders and media export completed with finite
outputs. The original notebook then failed a post-export observer assertion:
Temporal VAE attention was mislabelled as denoising. The original failed result
is preserved; a separate local validation reconciles the counter and passes all
saved-output checks. No GPU rerun was required. Full visual/audio quality review
and baseline equivalence remain open.

**CPU preflight version 3 passed** and its final outputs were collected once.
Kaggle then acknowledged **GPU E2E version 1** at
[H3 Anime REF2VA 8Step T4](https://www.kaggle.com/code/pamcks/h3-anime-ref2va-8step-t4-20261027).
Its final outputs were collected once, with no notebook execution status/list/
live-log polling. Video, audio and latent payloads remain private in the local
evidence directory; public evidence contains metrics, hashes and receipts.

The user's explicit request authorizes one full E2E generation for this workload,
superseding the operator-only restriction for this run. Other experiments and
historical environments remain untouched. No status/list/live-log polling or
automatic duplicate/retry is configured.

## Exact workload

| Parameter | Value |
|---|---|
| Model/base | MiniMax-H3 Ref2VA; `cnmds/minimax-h3-tp2-int8-561a6211` |
| Base revision | `d3e6a4b131056d2c626a01acf6ba48123a885783` |
| Adapter | `minimax_h3_ref2v_turbo_8step_v1.0_768p_comfyui_bf16.safetensors` |
| Adapter revision | `lightx2v/Minimax-h3-Turbo`, `30f485ed874aa44c8183348ad361c5eabad318da` |
| Adapter SHA256 | `6a56f41ab4229c9dd845b9501bbd475ee57e112d846cf2e819d534a1ae928c5a` |
| LoRA | Strength1.0; dual-GPU offline baking from the TP2 bundle; persisted to job disk |
| Video | 1152×672; 243 frames; 24fps; 10.125 seconds |
| Sampling | Eight Euler steps, Simple scheduler, CFG1.0 |
| Flow shifts | Video6, Audio3 |
| Seed | 20261020 |
| Reference images | Two, in the supplied Picture1/Picture2 order; Native `ref_image_size=match` |
| Negative prompt | Empty; no additional CFG branch |

The full supplied generation prompt is retained, including subject definitions,
shot timing, reference retention, soundscape and music instructions. The
contract rejects changed dimensions, sampler, step count, prompt checksum,
adapter identity or reference order. It does not silently reduce resolution or
replace Euler with the previous LCM setup.

243 is already on Native H3's `17k+5` frame grid. The empty joint latent uses
72 video time positions and 405 audio time positions. Actual TE/DiT token
counts are recorded from the executed model; no old long-sequence token count
is substituted for this new workload.

## Reference provenance

1. Character: [MediBang, 花未眠 by Shiori](https://medibang.com/picture/1h2106242105027740014340778/).
   The page's displayed artwork resolves to a 1280×905 JPEG, 317,018 bytes;
   SHA256 `e5b1710572f2587f7927e65b9e5f8f742bc435ca388ae23a16880fa6bf542383`.
2. Environment: [Wikimedia shrine artwork](https://commons.wikimedia.org/wiki/File:Algorithmically-generated_landscape_artwork_of_forest_with_Shinto_shrine.png).
   The original PNG is 4096×2048, 7,703,256 bytes;
   SHA256 `cd4cbfd7137b80e85e64ada86ffd93e512b874de84cb4291fc0278b747461e03`.

Both images were downloaded locally and verified. The character page initially
returned an empty direct response; its rendered browser page exposed the image
URL. The shrine's original file was downloaded through the browser after the
direct local request was rejected. The Linux CPU gate must separately prove
that its direct public downloads return these exact bytes before a GPU launch.
No replacement illustration or generated reference is used.

The new reference image bytes are excluded from the public source archive.
Only their public URLs, dimensions and checksums are included. No private HF
token is needed for this experiment's public inputs and weights.

## Frozen implementation and qualification

The source distribution is content-addressed:
`1973087713b2627b03135908f0d118485fb12bfa61918238cfe99139e0354de3.tar.gz`,
18,501,310 bytes. Capsule SHA256:
`6e4e1212cecc2cc838ec79c91bcf8c9d95a3bb11383771aab38682f001a8c350`.
The immutable source publication is commit
`68510431656fb806d30b3528a1d5844cc2510ea9` of `chinokikiss/minimax-h3-2xt4`.

It preserves the frozen legacy ComfyUI `d49e888586dd8ae012c0667b33466b815fee07f7`
archive and the approved exact core lock. Kitchen0.2.37 and TileLang0.1.13 stay
pinned. Kaggle retains and records its matching native Torch/CUDA/NumPy family;
the local conda environment and original snapshot are not modified. Source,
dependencies and weights use direct delivery without model/dependency Dataset
or notebook-output mounts. The latest replacement retains zero Dataset mounts.
There are no wheel archives, upstream HEAD fetches or credentials in the
published distribution.

The current H3 source is captured with per-file hashes, including:

- FP16-fix flow with the existing FP32 SwiGLU/quantization boundaries;
- fused-stored FC2 preparation and Scaled FP16 row SUM;
- the physically qualified SM75 TE transpose-view column route;
- current Kitchen TE/DiT attention and the shared dtype-preserving adapter;
- Temporal Video VAE default, component-only VAE loading and Native single-GPU
  FP32 Audio VAE decoding;
- bounded weight residency and disk-backed offline LoRA baking.

The eight-step adapter's actual safetensors header was read by HTTP ranges and
checked against all 208 Native target matrices. Local runtime contracts passed
179 tests. The corrected extracted snapshot separately passed 12 workload/layout
tests, its adapter API and every vendor-file checksum. Two additional local
tests cover binary reference delivery and reject changed bytes. These checks
are not physical T4 qualification.

### CPU execution and delivery history

The first accepted CPU notebook completed its setup in 153.884 seconds, including
179 runtime tests and the disk-bake contract. Its actual environment was
Python 3.13.15 and native CPU Torch 2.11.0+cpu, with the approved core overlay.
Its extra phase passed 11 tests but failed the native adapter's vendor-file
integrity check: the archive builder had omitted `LICENSE` and `NOTICE`.
Reference verification had not begun. This was an archive packaging failure;
it is not an E2E or GPU result.

The corrected builder includes both original files and checks the full vendor
manifest before publishing. The amendment records that existing `patches/`
executable bytes were unchanged. A fresh CPU gate still binds the corrected
archive rather than reusing the first receipt.

Two attempts to stage the original reference bytes inside a private notebook
failed to return a valid submission acknowledgement: a connection reset for the
10,706,594-byte Base64 notebook, then a non-JSON response for the 10,038,237-byte
Base85 notebook. One saved-source retrieval per ambiguous attempt matched the
first submission, not either retry. Neither is recorded as an accepted version.
No status, list or live-log retrieval was used. The failures correlate with the
large payload; the server did not establish a payload-size limit or root cause.

The replacement is an 11,607-byte notebook using the original public URLs and
mandatory SHA256, byte-length and image-dimension checks. Kaggle acknowledged
**version 2** at `2026-10-27T13:02:43.718714+00:00`. It repeats the Linux setup,
179 runtime tests, 12 workload/layout tests, vendor integrity and reference
authentication, without a GPU. Its collected results confirm that all 179+12
tests, vendor integrity and disk-bake CPU qualification passed. The character
download returned the original bytes; the shrine request returned HTTP 429
`Too Many Requests (015d12c)`. There was no sampling, decoding or GPU allocation.
A local negative check confirms that this failed input receipt blocks GPU job
construction before any submission call.

The accepted CPU attempt is preserved under
`D:/attention_eval/h3_anime_e2e_20261027/retry_cpu_v4/`; earlier attempts remain
in their own directories. The directory name counts local delivery attempts;
the accepted Kaggle version is 2. Public metadata and diagnostic logs are
archived in [the evidence directory](../data/h3_anime_ref2va_e2e_20261027/).

### Reference-only Dataset recovery

A selective private notebook carrying only the original shrine PNG was locally
verified byte-for-byte (7,703,256 image bytes, 9,641,569 notebook bytes), but its
submission also returned a connection reset without acknowledgement. One saved
source retrieval matched the already-collected failed CPU version 2, not that
retry. The image-bearing retry is not recorded as accepted.

An alternate private Dataset upload containing only the two original images
(8,020,274 bytes total) was attempted. The first creation failed with a TLS EOF
at the CLI's built-in existence check, before upload began. One separately
preserved retry timed out after 300 seconds without acknowledgement. Its
creation remains uncertain; no Dataset readiness query or further retry was
made. The current replacement does not depend on this Dataset.

### Immutable GitHub input recovery and remaining blocker

The publicly supplied shrine PNG is now mirrored separately from the frozen
runtime capsule at commit `225da31d025d510bb152683f53cf6f0a9843e0d2`.
[Attribution and original-file provenance](../../bench_assets/public_reference_images/README.md)
are retained. A local download from the immutable raw GitHub URL returned all
7,703,256 original bytes and the original SHA256; there was no re-encoding,
resizing or pixel change. The character image continues to use its already
successful original public URL. Captured activations and credentials are not
part of this mirror.

The replacement notebook fetches the mirror and verifies its byte count/hash
before runtime setup. The regular input gate then authenticates both images
against the original contract. GPU notebook construction inherits the exact
mirror URLs from its successful CPU qualification. A changed response fails
before any model execution. Three local reference-delivery tests pass, covering
exact binary transport, corruption rejection and immutable-URL enforcement.

The small mirror retry is stored under
`D:/attention_eval/h3_anime_e2e_20261027/retry_cpu_v6/`. It has zero Dataset or
notebook-output mounts, no embedded image payload and no credentials. Its
`SaveKernel` call returned `SSL: UNEXPECTED_EOF_WHILE_READING` before a version
acknowledgement. **Acceptance is unconfirmed; no GPU job exists in our submission
records.** The transport error also affects a small notebook, so large payloads
alone do not explain all failed submissions. Further requests were held at that
point; the later user-authorized saved-source check is recorded below.

### User-authorized local-input retry: accepted version 3

The user supplied the character JPEG and a PNG named
`屏幕截图 2026-10-27 211852.png`. Inspection confirms that the latter is the full
4096×2048, 7,703,256-byte original PNG with the original SHA256; its filename
does not indicate any pixel or encoding difference. Both supplied files were
verified and copied into a new evidence directory without replacing historical
attempts or changing the prompt/input contract.

One saved-source retrieval resolved the previous mirror retry's uncertainty:
the saved code still matched the already-collected failed version 2. That retry
was not recorded as accepted. The new submission was then acknowledged as
**version 3**. It combines private inline delivery of the small JPEG with the
immutable GitHub PNG mirror, keeping zero Dataset/output mounts and avoiding
both original image sites. Each file still undergoes the original byte count,
dimension and SHA256 checks.

The active evidence directory is
`D:/attention_eval/h3_anime_e2e_20261027/retry_cpu_v7/`. Its notebook includes
reference image bytes and remains private; only metadata, hashes and diagnostic
receipts are archived publicly. Local syntax, content-hash, delivery-binding and
no-GPU/no-Dataset checks passed, alongside three reference-delivery tests.
The frozen runtime capsule, package pins and generation settings are unchanged.
GPU construction inherits both delivery paths from a successful CPU receipt.
No notebook status/list/live-log polling is configured.

## GPU execution and acceptance

The fresh CPU version 3 gate passed 179 runtime regressions, 12 workload/layout
tests, all vendor-file checks, the disk-bake CPU contract, both original image
hashes/dimensions and the eight-step adapter's 208 target mappings. Its result
is `cpu_input_ready` and its matching frozen setup receipt is `cpu_ready`.
This qualifies the CPU setup/input path; it is not physical GPU validation.

The single bounded 2×T4 job completed these stages:

1. Verify both distinct SM75 T4 owners and the retained environment/compiler.
2. Fetch the immutable DiT shards and eight-step adapter, then bake offline on
   both GPUs. Verify all 208 Native targets and persist the resulting TP2 weights
   to job disk. Release only this job's verified original DiT downloads.
3. Fetch TE shards, auxiliary ranges and VAE checkpoints; load component weights
   only when needed. Image-only references must not load the Audio encoder.
4. Run the exact two-reference prompt and eight-step Euler trajectory. Record
   per-step finite summaries, actual layouts, backend counters and memory data.
5. Save final decode latents before decoding, use the default Temporal Video
   strategy and Native Audio decoding, then export MP4 with audio and a WAV.

Kaggle acknowledged GPU notebook **version 1**. The notebook uses the identical
CPU-qualified capsule and inherits both verified reference delivery paths. Local
checks confirm syntax, private visibility, T4 selection, zero Dataset/output
mounts, exact embedded JPEG bytes and absence of credentials. The reference
JPEG is inside the private notebook; that notebook is not copied to the public
evidence archive. The GPU submission receipt, qualification receipts and public
metadata are archived separately from reference payloads.

### Physical execution results

The worker verified two distinct Tesla T4 UUIDs, both compute capability 7.5.
The actual environment was native Torch **2.11.0+cu128 / CUDA 12.8**, Kitchen
**0.2.37** and TileLang **0.1.13**. The frozen ComfyUI source and legacy attention
API were retained. The actual DiT sequence was **59,134 tokens** per rank, with
Q shape `[1,28,59134,128]`: 2,392 text, 759+741 reference-image, 810 audio and
54,432 generated-video tokens.

| Recorded stage | Seconds | Scope |
|---|---:|---|
| Offline dual-GPU LoRA bake | 434.87 | All 208 targets; 21,237,118,728 bytes persisted to job disk |
| Reference encoding and TE | 154.89 | Both exact reference images and the full prompt |
| Eight-step sampler | 1,338.75 | All eight denoiser outputs and callbacks finite |
| Temporal Video VAE | 232.98 | `native_temporal_pairs`, no fallback |
| Native Audio VAE | 3.28 | Single-GPU Native, no fallback |
| File export | 17.40 | MP4 with audio plus separate stereo WAV |
| Recorded foreground interval | 1,748.46 | Reference/TE through export; excludes offline bake/downloads |
| Whole execution worker | 2,886.64 | Includes downloads, bake, process setup/cleanup and final assertion failure |

These are one-request observations with cold loading/compilation, not repeated
steady-state benchmarks or a measured speedup. Denoiser command time on rank 0
was 188.83 seconds on the first call and approximately 146.68–157.28 seconds on
later calls. The recorded foreground interval was about **29 minutes 8 seconds**.

PyTorch denoiser peak allocation was **12.57 GiB / 11.05 GiB** on ranks 0/1;
peak reservation was **13.83 GiB / 14.31 GiB**. Sampled full-device usage reached
approximately **15.00 GiB / 14.98 GiB**, including other CUDA allocations in the
instance. This workload completed without an observed GPU OOM or cgroup OOM
kill; the device figures show little spare memory for a larger workload.

Each rank recorded 250 optimized TE column calls, 50 TE causal-attention calls,
900 Scaled FP16 epilogues and 450 fused SwiGLU/quantization preparations. The
saved rank state confirms disk-baked DiT weights, zero inference merges and no
online LoRA attachment. Image-only reference preparation loaded no Audio encoder.

### Post-export observer failure and reconciliation

The original failure is `assert kitchen_denoiser_calls(bench) == 50*steps`,
after successful file export. The observer updated its context for Query VAE
`video_pixels` commands, but not Temporal `video_temporal_pair`. Consequently,
the last `denoise/nfe8` label remained active during Temporal decoding. On
**each rank**, the old filter counted **2,416** calls: the expected 400 DiT calls
plus **2,016** later Video VAE calls. Including TE gives 2,466 total Kitchen calls.

The saved event list contains a `command_gpu_span` completion marker for every
denoiser command, appended after that command's operators. Restricting calls to
their matching completed command intervals recovers exactly **50 calls in each
of eight steps**, on both ranks. All 2,016 excess calls occur after the final
denoiser completion marker. The Temporal RPC path in the frozen source explains
the stale labels; this is an observer attribution defect.

The benchmark observer now assigns context for every RPC and profiles Temporal
decoder commands explicitly. The verifier uses completed command boundaries;
it still counts genuine extra calls inside denoising and rejects missing or
duplicate markers. Four targeted regressions and the complete **183-test CPU
suite pass**. These changes affect benchmark instrumentation/validation only.
The submitted frozen archive and operator implementations are preserved.

`benchmarks/validate_anime_e2e_outputs.py` independently checked the unchanged
saved T4 result, both ranks, all eight finite trajectories, every remaining
integration assertion, exact output shapes, offline LoRA state and local media
checks. Its separate result is
`saved_e2e_checks_passed_observer_assertion_reconciled`. The original remote
`status=failed` and traceback remain archived; no result JSON was rewritten and
no new GPU job was submitted.

### Exported media and review

The local MP4 fully decodes through FFmpeg: **243 frames, 1152×672, 24 fps,
10.125 seconds**, with an audio stream. The WAV contains **324,000 stereo samples
at 32 kHz**, also 10.125 seconds. Its measured peak is 0.1483 full scale, RMS
0.02318, and no exact PCM clipping was detected. These checks establish a valid,
non-silent export, not audio semantics or baseline fidelity.

The 12-frame contact sheet samples 0–10.08 seconds. The girl's purple hair,
kimono and umbrella remain identifiable in those samples; the shrine is present
and the white fox appears during the sequence. This is a sparse visual review;
smooth motion, identity details, sound/music and prompt adherence still require
full playback. No dense Native control generation was run.

Private local media:
`D:/attention_eval/h3_anime_e2e_20261027/retry_cpu_v7/gpu_outputs/turbo_e2e/generation/`.
The files are `video.mp4`, `audio.wav` and recovery latents `decode_inputs.pt`.
The preview is under `retry_cpu_v7/local_review/contact_sheet.jpg`.

Acceptance requires eight completed denoiser outputs and sampling steps,
finite sampled/decoded tensors, exact requested video dimensions/frame count,
250 optimized TE column calls per rank, no online LoRA calls and a disk-baked
checkpoint that is not merged again during inference. Video strategy/admission
and any Native fallback reason are recorded rather than hidden.

Model preparation and compilation remain visible as cold stages. A single
request cannot establish steady-state throughput or a baseline speedup.
Successful execution also does not establish visual/audio quality or generation
equivalence; the exported media still need review for identity, continuity,
fox interaction, late motion and audio/music consistency.

The proposed GPU execution cap is 4,800 seconds plus a 300-second setup stage,
with a 5,400-second notebook limit. These are bounds, not an ETA. Final artifact
collection is once per phase; the user may report completion rather than relying
on status polling.

Evidence directory: `D:/attention_eval/h3_anime_e2e_20261027/`.
Machine-readable [workload and delivery status](../data/h3_anime_ref2va_e2e_20261027.json).
