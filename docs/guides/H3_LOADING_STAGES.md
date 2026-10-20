# H3 LoRA baking and VAE stage loading

Deployment uses the [pinned local ComfyUI runtime](H3_LEGACY_RUNTIME.md), including
its original Video VAE attention API. The newer sibling is not the default.

Implemented 2026-10-19 in the linked TP2 ComfyUI node. Restart ComfyUI to load
the changes. The operators workflow has been regenerated and copied to
ComfyUI's workflows directory; original and Native fallback workflows remain.

## Offline LoRA

Set **`MiniMaxH3TP2Loader.lora_mode=offline`**, connect the existing LoRA stack,
and select the original diffusion checkpoint. `online` remains the default.

On the first use, the loader releases the previous TP service before baking.
Whole DiT/refiner layers are balanced across `cuda:0` and `cuda:1` by matrix
size. Workers are spawned processes; each processes one complete matrix at a
time. A matrix is never divided between devices: Native FP32 dequantization,
ordered adapter application, alpha/strength scaling and deterministic
full-input-width ConvRot requantization are unchanged. A single affected layer
uses one worker. One available GPU remains supported by the converter.

The base file is copied once. Workers write disjoint weight/scale ranges through
independent descriptors; no merged tensors travel through NCCL or CPU pipes.
Publication happens after all workers finish and fsync. The parent records
checksums, source identity, layer assignment and verification status. Cancellation
is checked during copying and between matrices; failed/cancelled worker work
removes only its unpublished scratch file. Original checkpoints are never edited.

The loader caches output beside the base:

```
D:/minimaxH3/h3_lora_baked/<source-and-stack-identity>.safetensors
```

Its receipt is `<checkpoint>.json`. Warm reuse validates base/LoRA identities
and the output size/mtime. A changed source or strength gets another cache key;
missing or inconsistent receipts are rejected. A new bake requires disk space
for another full diffusion checkpoint. After choosing the baked file, the loader
passes an empty online LoRA stack to TP workers, preventing double application.
Offline merging retains its existing numerical contract: it is not byte-exact
with an online floating-point LoRA bypass over the original INT8 model.

The original CLI is preserved and now defaults to up to two devices:

```powershell
python benchmarks/merge_h3_lora.py --base D:/minimaxH3/MiniMax_H3_Ref2VA_pruned_int8_convrot.safetensors --lora D:/minimaxH3/loras/example.safetensors 1.0 --output D:/minimaxH3/h3_baked.safetensors --devices cuda:0 cuda:1 --verify-native
```

`--prefix` still exports a bounded fixture. `--verify-native` independently
checks each merged matrix/scale against the ordinary ordered A/B formula.
Compilation/loading/copy/baking cost is one-time work, not inference latency.

## VAE weights by stage

`MiniMaxH3ParallelVAELoader` now recognizes Native H3 video/audio checkpoints
and returns a staged VAE. Unsupported VAE types retain the ordinary loader.
No additional stage switch is needed.

At node load, only encoder weights and shared latent normalization are read.
Native metadata and public encode/decode/tiling planners are retained through a
meta-device shell. The inactive half has no CPU/GPU parameter storage. At first
decode, the encoder patcher/storage/host lease is released before decoder reads
and allocations. Before denoising, registered encoders are explicitly released.
Another encode automatically reloads the encoder when necessary. Component
switches are serialized by the TP session lock and VAE lock.

| Current checkpoint | Encoder stage bytes read | Decoder stage bytes read |
|---|---:|---:|
| Video INT8 ConvRot | 360,747,328 | 2,450,218,528 |
| Audio FP32 clean | 345,477,248 | 259,672,288 |

Shared latent buffers are included in each stage. Video encode retains
`encoder`/`quant_conv`; decode retains `decoder`/`post_quant_conv`. Audio encode
retains encoder/posterior projections; decode retains `decoder`/`dec_in_proj`.
Video pixel statistics and the nonpersistent decoder RoPE frequencies are
rebuilt with Native values. Quantized weight wrappers are preserved.
The legacy fused-attention API's nonpersistent `qk_norm_scale` buffers are also
restored to ones, including when query-parallel admission falls back to Native.

CPU component construction/backing uses the TP service's shared host budget.
Stage teardown, service close/abort and VAE node eviction return that lease.
Normal Native model management owns parent GPU transfers/offload. Worker VAE
decode remains decoder-only and retains its existing collective admission.

The current local `ComfyUI-master` has changed Video VAE attention to a different
API from the qualified query-partition implementation. `auto` therefore falls
back to Native public video decode with a recorded reason on that source,
instead of raising an import/signature error. The pinned, qualified attention
API retains query-parallel decode. This change does not claim qualification of
a new attention/partitioning algorithm. Audio channel-parallel selection remains.
The follow-up [query fallback check](../reports/VIDEO_VAE_QUERY_FALLBACK_CHECK.md) records
the exact compatibility gates and observed Native attention calls.

## Idle residency

With **`idle_residency=True`**, successful queue-idle completion now preloads
the registered VAE encoders, then starts the existing partial TE preload on both
workers. Encoders execute in the parent on the ordinary Native VAE device,
normally GPU 0; they are not duplicated on both TP ranks. Their allocation is
visible to the workers' driver-memory admission guards.

Encoder reads are cancellable between tensors. GPU preload requires weights
plus `idle_reserve_bytes` headroom, and uses Native model management. If GPU
headroom is insufficient, a prepared CPU encoder can still be reused. If VAE
speculation fails, it records the reason and still permits TE prefetch. The
background task does not run image/audio encoding or prepare a decoder.

Prompt submission cancels speculation; compatible completed encoder/TE entries
can be reused. The active graph filters registered VAE names, so stale cached
VAE nodes are released. Memory pressure or an incompatible request releases
encoder and TE speculation. Active Native VAE calls protect their own staged
component from that pressure callback. TE parent-model reclamation preserves
eligible preloaded encoder patchers, and sampling explicitly releases them.

## Validation and limits

Evidence directory: `D:/attention_eval/h3_loading_stages_20261019/`.

- `test_h3_loading_stages.py`: 13 fixture tests cover whole-layer ownership,
  spawned vs serial checkpoint bytes, ordered multi-LoRA/alpha handling,
  base preservation, cache reuse/corruption rejection, scratch cleanup,
  no double application, component separation, cancellation, graph filtering,
  memory pressure and worker-abort parent leases.
- `qualify_h3_layer_bake.py`: five INT8/BF16 targets, two mixed-rank adapters,
  independently verified Native weight/scale bytes. Two spawned CUDA workers
  sharing the local GPU produce the same checkpoint bytes as serial baking.
- `qualify_h3_staged_vae.py`: actual trained video encoder and all 36 decoder
  layers on a 32×32 single frame; actual trained audio encoder/decoder on a
  1,600-sample stereo input. Both public encode and decode are byte-exact with
  ordinary full Native VAE loading. GPU encoder idle preload, exact component
  read counts and encoder destruction before decoder loading pass. The current
  Video VAE API's auto-to-Native compatibility path is exercised.
- Existing operator integration, tiny ComfyUI wiring and loading/residency
  regression suites are rerun; their logs and source hashes accompany the results.

These are local correctness checks on RTX 5060 Ti SM120 and CPU fixtures.
The two-worker CUDA test shares one GPU and is not physical dual-GPU evidence.
No complete H3 generation, full trained diffusion checkpoint bake, new T4 job,
profiling run or dual-GPU speedup measurement was performed.
