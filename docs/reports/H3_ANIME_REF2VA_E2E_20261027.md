# Eight-step anime REF2VA E2E — 2026-10-27

Status: the corrected Kaggle CPU preflight **version 2** at
[H3 Anime REF2VA CPU Preflight](https://www.kaggle.com/code/pamcks/h3-anime-ref2va-cpu-preflight-20261027)
passed environment, runtime and vendor checks, then failed reference
authentication because Wikimedia returned **HTTP 429** for the original shrine
image. Its final outputs were collected once. An exact immutable GitHub mirror
now resolves that input-origin dependency. The replacement small notebook's
submission hit a Kaggle API TLS EOF; acceptance is **unconfirmed**. Further
Kaggle requests have stopped to avoid duplicate execution.
**No GPU job has been submitted and no T4 generation result is claimed.**
The GPU job requires the matching Linux setup, runtime checks and reference-image
authentication to pass first. No notebook execution status/list/live-log endpoint
is queried.

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
alone do not explain all failed submissions. No additional source/status/output
request has been made for this latest uncertain attempt.

## GPU execution and acceptance

After the fresh CPU gate passes, one bounded physical 2×T4 job will:

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
