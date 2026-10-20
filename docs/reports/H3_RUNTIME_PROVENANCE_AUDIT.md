# ComfyUI source provenance correction — 2026-10-20

There is no verified evidence that the local `ComfyUI-master` was synchronized
with upstream during the recorded work. It has no Git metadata; the saved
source identity also records its commit as `null`. Calling it "newer upstream"
was unsupported. It is an API-different source tree; its chronological ordering
relative to the separate pinned checkout is not established.

All **642 recorded source/configuration files** in
`D:/attention_eval/quality_first_20261014/assets/source_identity.json` still match
the current `ComfyUI-master`, including `vae.py`, `model.py`, `ops.py` and `sd.py`.
This covers those recorded files, not every unrecorded file or earlier history.
The comparison is saved in
`D:/attention_eval/h3_conda_runtime_20261019/master_provenance_audit.json`.

## What was restored

The project launcher and benchmark defaults were redirected to the separate
`2xT4 minimax h3/ComfyUI` checkout at `d49e888586dd8ae012c0667b33466b815fee07f7`.
The default interpreter was returned to the existing conda `comfyui` environment.
The experimental Torch 2.11 venv was retained as historical evidence. The conda
installation was not downgraded, and `ComfyUI-master` was not rolled back in place.

The recorded additions to Master were the TP2 custom-node junction and a
separate TP2 operators workflow. The original workflow was used as input to the
workflow builder; no overwrite of it is recorded.

## Functional scope

When the legacy query-parallel extension runs against Master's different Video
VAE attention signature, it selects Native fallback. In the measured fresh
process, regular and tiled decode each used 36 SDPA calls and zero Kitchen
attention calls; QKV weights were still quantized. This is a compatibility
difference, not evidence of a new upstream update or a loss of INT8 weights.
The selected pinned tree supports the original query wrapper and its Kitchen
path. Local trained decoder checks are byte-exact with Native controls.

Master also lacks `_embed_and_pack`. A projection-boundary bridge was added to
the project's residency patch for that API; the pinned path retains its original
packed broadcast. Native model source was not edited for the bridge.

At the time of this audit, the controlled launcher restricted custom nodes to `minimax_h3_tp2`,
disables partner/API nodes, disables dynamic VRAM and disables automatic browser
launch. These are launcher-imposed restrictions and differ from an ordinary
ComfyUI launch. It also uses the pinned checkout's own user/custom-node folders;
it does not mirror all of Master's plugins or settings. Master's normal launch
is separate and is not changed by those flags.

The subsequent pipeline review made those node/API restrictions opt-in through
`--isolated-ui`. Dynamic VRAM remains disabled for TP2 window ownership; the
selected source tree still has its own plugins and settings.

Requested operator integration, offline LoRA baking, staged VAE loading and idle
preloading remain. The Token Refiner SDPA change was separately requested; it
was not a side effect of restoring source selection.

Local tests do not establish that all original workflows, plugins, generation
outputs or performance are identical across these two source trees. No full H3
generation or new physical TP2 run was performed for this correction.
