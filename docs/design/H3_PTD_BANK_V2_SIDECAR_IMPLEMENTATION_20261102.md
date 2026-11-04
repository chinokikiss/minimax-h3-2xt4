# H3 PTD Bank v2 Sidecar Implementation — 2026-11-02

Status: **Local CPU integration complete; physical dual-T4 bake and E2E are unqualified.**
Scope: opt-in, base-free PTD inference and bounded hop-major offline baking. The existing TP2 Loader, native operators, v1 PTD Bank and regular Turbo LoRA route remain the defaults.

## Code entry points

| Responsibility | Opt-in source |
| --- | --- |
| Self-contained Bank admission and Native-facing reader | `patches/ptd/bank_v2.py` |
| Independent PTD + ordinary LoRA terms | `patches/ptd/composite_lora.py` |
| One-hop bounded Baker, shared static extraction, final assembly | `patches/ptd/bake_v2.py` |
| Full v1 Bank reindexing, no expert-payload copy | `patches/ptd/promote_v2.py` |
| Part SHA256 admission, remote readback proofs, private Kaggle staging | `patches/ptd/publish_v2.py` |
| Experimental v2 rank-local CPU parallel reads and post-read cache advisory | `patches/ptd/bank_v2.py::HopReaderV2.load_rank` |
| Offline command line | `tools/h3_ptd_bank_v2.py` |
| ComfyUI v2 Loader | `MiniMaxH3TP2PTDBankLoader` in `patches/ptd/nodes.py` |
| Reader format dispatch | `patches/tp2_bundle.py::open_checkpoint` |
| Worker v2 branch | `patches/ptd/runtime.py::_open_bank` |

The new Reader uses the original Native module shells, `ExpertQueue`, pageable `HostWeightCache` and GPU weight-window. It never loads the original 19.78 GiB DiT descriptor after Bank publication.

## Offline use

Run these commands only in an appropriately provisioned **Baker** environment. Every production hop requires two physical Tesla T4 GPUs and the existing pinned Native/Kitchen/ComfyUI source. Use the same `--lora FILE STRENGTH` arguments for shared and all eight hop invocations. A Bake result is a **local part**, not a published remote dataset.

`python tools/h3_ptd_bank_v2.py shared --base /input/dit_reader.json --checkpoint /input/ptd.safetensors --output /tmp/shared [--lora /input/ordinary.safetensors 1.0] [--allow-pruned-adaln]`

`python tools/h3_ptd_bank_v2.py hop --base /input/dit_reader.json --checkpoint /input/ptd.safetensors --hop 0 --output /tmp/hop_00 [--lora /input/ordinary.safetensors 1.0] [--allow-pruned-adaln]`

Repeat the `hop` command explicitly for hops 0–7, **publishing and verifying each completed part remotely before reclaiming its local scratch**. Do not keep eight hop parts in `/tmp`. Output destinations are non-overwriting, and aborted `.partial` directories are retained for diagnosis. No unsafe automatic scratch deletion, Kaggle upload or remote mount operation is embedded in the current implementation.

### Private Kaggle part publication (explicit; never automatic)

1. Run `prepare-kaggle` only after each completed local part has passed source-file SHA256 verification:

   `python tools/h3_ptd_bank_v2.py prepare-kaggle --root /tmp/hop_00 --part-id hop_00 --staging /tmp/hop00-kaggle-stage --dataset-id OWNER/ptd-hop00-RECIPEID --license AUTHORIZED_KAGGLE_LICENSE_ID`

   The new upload directory is a **hardlink tree on the same filesystem**, not an extra 21 GiB payload copy. Its root contains only `dataset-metadata.json` and a `payload/` subdirectory with the verified Bake files, keeping top-level entries below Kaggle's **50 top-level-file** limit. The original immutable part directory is left untouched. The required license is a conscious operator input, never guessed by the code. Do not write to linked payloads in either directory. The command prints the safe upload argv without executing it.

2. From the authenticated, quota-approved **Baker** session, explicitly run `kaggle datasets create -p /tmp/hop00-kaggle-stage -q -t -r tar`. The `-r tar` is mandatory: Kaggle CLI's default directory mode is `skip`, which would silently omit `payload/`; `tar` packages that directory for upload. Archive packaging may require an additional **part-sized temporary disk allocation** (~21.3 GiB for one 50-layer hop), so re-check `/tmp` capacity before issuing the actual create. The actual remotely exposed unpacked path must be inspected, never assumed. **Never append `-u`: that would request a public Dataset.** This is a new private Dataset create, not `dataset_upload` or a replacement Dataset version. The CLI return code is an upload-request acknowledgment, **not** proof that Kaggle has processed or exposed the bytes. Do not delete source or staging scratch on upload acknowledgment.

3. In a fresh environment where the now-processed, **version-pinned** private Dataset has actually been mounted read-only (Kaggle `/kaggle/input` mounts cannot be hot-attached mid-run), run the readback verifier on that mount. Supply the expected SHA256 of the **original local receipt**, recorded before upload, and the pinned Dataset version reported by Kaggle:

   `python tools/h3_ptd_bank_v2.py verify-part --root /kaggle/input/hop00/payload --part-id hop_00 --expected-receipt-sha256 ORIGINAL_RECEIPT_SHA256 --published-identity OWNER/ptd-hop00-RECIPEID/versions/1 --output-proof /tmp/hop00.readback.json`

   The sample `payload/` path above is **conditional** on Kaggle actually exposing that unpacked subdirectory; inspect the mount and point `--root` to the directory containing `receipt.json`, rather than guessing the server-side archive layout. Repeat this for `shared` and all eight hops. `verify-part` streams full SHA256 for receipt-pinned rank payloads, per-rank sidecars, overlay, static auxiliary, Native header, block plan and recipe. It validates safetensors metadata and rank coverage without materializing large tensors or running inference. The claim `--published-identity` is **operator-supplied**, not independently authenticated against Kaggle account metadata; independently inspect the attached Dataset version. Preserve the nine proof files outside disposable part scratch.

4. Only after all nine parts are mounted and verified, assemble the final logical index:

   `python tools/h3_ptd_bank_v2.py assemble --shared-root /kaggle/input/shared/payload --hop-roots /kaggle/input/hop00/payload /kaggle/input/hop01/payload /kaggle/input/hop02/payload /kaggle/input/hop03/payload /kaggle/input/hop04/payload /kaggle/input/hop05/payload /kaggle/input/hop06/payload /kaggle/input/hop07/payload --readback-proofs /tmp/shared.readback.json /tmp/hop00.readback.json /tmp/hop01.readback.json /tmp/hop02.readback.json /tmp/hop03.readback.json /tmp/hop04.readback.json /tmp/hop05.readback.json /tmp/hop06.readback.json /tmp/hop07.readback.json --manifest /tmp/bank_v2.json`

   Production assembly checks that all nine readback receipts are independently anchored to the pre-upload source receipt SHA256, and that shared recipe hashes, layer counts, pinned identities and mounted roots agree. It rejects aliased part mounts and incomplete shard/rank coverage. It validates the full v2 Bank against a temporary index before atomic publication. The final Bank manifest itself still needs explicit durable publication; the CLI does not upload it. Source scratch reclamation is an **external, authorized operation** after successful independent remote verification.

Never confuse a local source `verify-part` proof with a verified remotely mounted copy. The proof is mount-path-bound but not signed, and neither this CLI nor the current project can attest to Kaggle account privacy, quota, Dataset processing state, or a particular version through Kaggle's API. This phase does not implement network-streamed remote readback or full-duplex publishing.

For a preexisting **complete** v1 Bank, a sidecar converter is also available:

`python tools/h3_ptd_bank_v2.py promote-v1 --v1 /path/complete_v1/bank.json`

It creates a separate `bank_v2.json` and static auxiliary in the v1 directory without modifying the original v1 receipt, Bank index or expert payloads. Do not assume this converter reduces the disk requirement of the original all-eight-hop Baker.

## Inference wiring

- Choose the **MiniMaxH3TP2PTDBankLoader** node, not the ordinary `MiniMaxH3TP2Loader`.
- Supply an existing TE checkpoint and the completed v2 manifest. If the eight parts live outside the manifest directory, provide explicit part roots using `part_roots_json`, e.g. `{"hop_00":"/kaggle/input/h3-hop00","shared":"/kaggle/input/h3-shared"}`.
- Connect MODEL, SESSION and H3_PTD_BANK to the existing PTD Adapter; connect its controller to the PTD Sampler.
- Keep per-hop refiner and PDD heads enabled: they cannot be disabled with the original Base absent.
- The ordinary LoRA stack is baked offline; no online LoRA is accepted.
- The existing regular H3 workflow does not need a PTD Bank, and its dispatch remains unchanged.

## Experimental direct-L2 host I/O (strictly opt-in)

The v2 Reader already opens each immutable part's mounted path directly via `CheckpointReader`; it does **not** create an L1 `/tmp` rolling copy. The new optional v2 Loader inputs control read concurrency and advisory page-cache retirement without changing any GPU operator or original Loader default:

| v2 Loader option | Default | Local behavior |
| --- | --- | --- |
| `ptd_io_workers` | `1` | At `2..8`, only the existing rank's tensor reads run in a bounded `ThreadPoolExecutor`. Each reader performs its own positional `pread` into final CPU tensor memory; no CUDA/NCCL work is dispatched from these threads. |
| `ptd_retire_page_cache` | `false` | After all tensor reads and rank admission complete, try Linux `posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED)` on the now-consumed immutable shard. The hint is advisory; unsupported platforms record `unavailable` and never fail inference. |

Start with **4 threads per rank** (=8 readers total) on a separately approved CPU/NFS qualification, and compare against 1 and 8/rank (=16 total). The previous Kaggle CPU sweep measured 8–16 global readers as the best **synthetic-file** range, but did not benchmark PTD shard preparation or host memory contention. `HopReaderV2.workspace_bytes` reserves 8 MiB per extra I/O worker within the existing `WeightSpec.prepare_peak_bytes` accounting. The original `HostWeightCache` still schedules one rank preparation at a time; its capacity, lock/hold and memory budget contracts are unchanged.

No assertions are made about GPU E2E latency, NFS throughput, page-cache eviction rate, or absence of memory pressure until a physical dual-T4 / actual-mounted-Dataset study completes. On Windows the fadvise path is expected to be unavailable. V1 PTD, ordinary Loader, generic `CheckpointReader`, runtime LoRA, and all operator backends are unaffected.

## Mounted Dataset CPU I/O qualification gate (no GPU submission)

Run only **after** nine immutable Dataset parts have been independently verified and the v2 manifest has passed complete Bank admission. This probe loads representative rank shards directly from the mounted Dataset; it never modifies source files, uploads, downloads, remounts, triggers GPU compute, or drops the global kernel page cache. Supply a local UTF-8 `mounts.json` with all nine `part_id: /kaggle/input/.../payload` mappings.

```bash
python benchmarks/run_ptd_v2_io_cpu_probe.py \
  --manifest /tmp/bank_v2.json \
  --part-roots-file /tmp/mounts.json \
  --hop 0 --layers 0 25 49 --ranks 0 1 \
  --repeats 2 --max-seconds 900 \
  --output /tmp/ptd_v2_io_cpu_probe.json
```

The standard four profiles are `1-off` (baseline), `4-off`, `4-on` and `8-on` **per rank**, with a maximum of 96 rank-load cases per run. **This standalone probe measures rank 0 and rank 1 sequentially**, so it does not recreate simultaneous dual-rank NFS pressure, GPU copy contention or the global 8–16-reader topology; those require the physical dual-T4 study. The probe first checks the v2 manifest/rank coverage with the production metadata-only Bank reader, then times each CPU `load_rank` and its tensor-reading segment separately. **After** each timed rank load it hashes every tensor and requires complete bitwise agreement with the Bake state proof. It samples process RSS and, where available, cgroup-v2 current/file/anon memory, file refaults, reclaim scans and OOM events. Intermediate JSON receipts are atomically checkpointed; an exception is retained as `status=failed` with the completed-case inventory. The report binds its results to the manifest SHA256, recipe SHA256, claimed part identities, mount paths and a SHA256 fingerprint of each rank state.

Interpretation:

- `status=passed` means **CPU rank tensor exactness and completed probes only**, not PTD visual quality, exact Native full-model parity, or GPU qualification.
- `page_retire_status=hint_issued` means the OS accepted advisory calls; it does **not** establish actual page eviction. `unavailable` is expected on Windows.
- `summary.*.aggregate_read_mib_per_sec` and `median_io_speedup_vs_1_off` are observations potentially biased by NFS cache warmup and sequential profile order. Repeat in a fresh controlled environment and use per-rank/per-layer rows before making any ROI decision.
- Check `memory.cgroup_file_peak_bytes`, `memory.cgroup_pgscan_kswapd_delta`, `memory.cgroup_refault_file_delta` and `memory.cgroup_oom_delta` separately from process RSS. Do not enable `ptd_retire_page_cache` based solely on a successful fadvise return.
- Keep ordinary Loader and PTD v2 I/O defaults unchanged until **actual dual-T4 PTD E2E** independently proves no unacceptable performance regression, precision/quality difference, or host/VRAM pressure. No automatic Kaggle task submission or polling is part of this gate.

## Full-source single-hop preflight (read-only; no GPU job)

The dedicated `benchmarks/preflight_ptd_v2_bake.py` verifies a complete Native DiT descriptor, the released PTD 8-hop/32-interval metadata and adapted 50-layer coverage, source-size/SHA256 identity and explicit pruned-AdaLN consent. It estimates disk headroom with `patches/ptd/bake_v2.py::estimate_hop_scratch`, **the exact function called immediately before a hop Bake**, so the two paths cannot silently drift. Ordinary LoRA file presence/finite strengths and Native compatibility are checked if `--lora` is supplied. This is metadata / source qualification, not Native full-weight merge or final inference quality verification.

```bash
python benchmarks/preflight_ptd_v2_bake.py \
  --base /input/dit_reader.json \
  --checkpoint /input/MiniMax-H3-Ref2VA-PTD-RefAsymOPD-8Step.safetensors \
  --scratch-dir /tmp \
  --allow-pruned-adaln --verify-ptd-sha256 --check-t4 \
  --output /tmp/ptd_v2_preflight.json
```

Use the **same** `--lora FILE STRENGTH` arguments in preflight, shared Bake and every hop Bake. Omit `--allow-pruned-adaln` for an unpruned base. `--verify-ptd-sha256` streams the full fixed release before qualifying it; a size match or the old two-block trained subset is not sufficient. The default `--check-t4` **off** leaves `cuda_gate=not_checked`; an otherwise fully source-qualified report is `status=source_qualified`, **not** `passed`. On the physical Baker, enable the explicit T4 check to obtain `physical_baker_ready=true`. Neither mode launches kernels, downloads weights, submits a Kaggle job or changes any original production Loader path. Set `--scratch-dir` to the **same existing filesystem/directory** that will hold the hop Bake `--output` parent. The output report is optional, non-overwriting if requested, and contains `issues` and `pending` fields. A non-ready report exits nonzero.

This preflight does not provision missing sources, prove remote Dataset identity/privacy or reserve the additional `kaggle -r tar` packaging space. The worktree currently contains no complete real PTD checkpoint, no complete nine-part remote readback proofs and no production `bank_v2.json`. A real CPU/NFS probe remains blocked until independently verified version-pinned mounts exist; the old two-block T4 subset and synthetic fixtures cannot fill that evidence gap.

## Proof and limitations

The local conda `comfyui` tests passed the original 77/77 selected old+new CPU regressions. The new publication extension also passed 17/17 publication-targeted tests, 80/80 combined PTD + I/O + mounted-probe + source-preflight regression tests, and the Node Loader migration check, with source-bound coverage for:

- Separate v1 and v2 format admission and unchanged production Loader defaults.
- Base descriptor removal after v2 publication.
- All eight hop manifests and TP2 rank-state coverage, split-part root lookup, missing part rejection.
- Metadata/schema-only production validation without runtime payload SHA256.
- PTD+ordinary LoRA term order, a single Native FP32 full-width quantization, and independent tensor reference comparison.
- One-hop-at-a-time synthetic CPU Bake and staged final publication.
- Full readback SHA256 for all rank payloads, sidecars, static header/block-plan and auxiliary; independently supplied source-receipt/recipe/layer-count pinning; operator-supplied version/commit provenance gate; same-filesystem hardlink private-upload staging without touching the source Baker directory.
- Bitwise CPU equivalence of serial and parallel v2 rank loading, rank-local memory admission reserve, strict I/O option validation, and post-read advisory call ordering.
- A CPU-only mounted Bank probe that separates rank read/load/hash times, checks every tensor against the Bake SHA256 state proof, samples cgroup-v2 file cache/PSI counters, and preserves partial failures without running CUDA kernels.
- Source preflight with full release-size / optional full-SHA gating, missing-input rejection, fixture-vs-production separation, explicit physical-T4 gate, and the shared Baker scratch estimate.

**Not yet proven:** full-weight joint Bake on two actual T4s, actual Kaggle private publication/readback or version identity attestation, 170 GiB upload/readback throughput, Kaggle private-dataset quota suitability, page-cache retirement effectiveness, parallel I/O throughput/latency ROI, optional Layer-band Full-Duplex, complete 8-hop trained PTD accuracy, runtime fallback counters and 1 MP E2E performance/VRAM. Avoid conflating the preceding ordinary Turbo 1 MP E2E with PTD validation.

Next physical gate: frozen CPU/Linux environment qualification, then one actual trained PTD hop on two T4s with the original Base unavailable to inference. Only after that should eight-hop/1 MP and network/cache profiling be scheduled. Never automatically submit, retry or poll a Kaggle GPU job without a separate explicit instruction.
