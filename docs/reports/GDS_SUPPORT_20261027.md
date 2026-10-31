# GPUDirect Storage support check — 2026-10-27

Local Windows execution is **unsupported**. Direct GDS is also **unavailable
on the tested Kaggle two-T4 allocation**: both strict device checks fail because
the host `nvidia-fs` driver is not loaded. cuFile additionally reports PCI
P2PDMA as unsupported. The verdict applies to this environment and scratch path;
it does not imply that T4 hardware cannot use GDS on a correctly configured host.

The corrected probe completed in **14.28 seconds**; its supervisor exited
successfully after **14.86 seconds**. No package was installed and no H3
component was loaded. All eight returned files were collected once and their
hashes recorded. The returned probe matches the submitted source byte for byte.
See [summary.json](../data/gds_support_20261027/summary.json) and the complete
[GPU result](../data/gds_support_20261027/gpu_result.json).

The first submission failed before GPU I/O because `/kaggle/temp` was absent.
Its complete outputs are preserved. The same probe was submitted as version 2
using `/tmp`.

## Executed Kaggle result

| Item | Observed |
|---|---|
| GPUs | Two Tesla T4s at PCI 00:04.0 and 00:05.0 |
| Driver / kernel | 580.178.04 / Linux 6.18.55+ |
| Native Torch / CUDA build | 2.11.0+cu128 / 12.8 |
| Native cuFile package | nvidia-cufile-cu12 1.13.1.3; libcufile.so.0 found |
| Native Torch GDS bindings | Present |
| nvidia_fs module / device / stats | All absent |
| Requested scratch | /tmp, on an overlay filesystem |
| PCI P2PDMA | cuFile reports CUDA_ERROR_NOT_SUPPORTED |
| Device 0 and device 1 | cuFileHandleRegister fails: nvidia-fs driver is not loaded |
| CPU compatibility fallback | Explicitly disabled; not accepted as GDS support |

Each child allocated its small synthetic CUDA payload, then failed while
registering the file handle, before its intended 4-MiB read/write. This is a
host-prerequisite failure, not a byte-comparison failure or throughput result.
Enabling compatibility mode would use the CPU path and would not validate GDS.
Installing a Python package cannot by itself supply the missing host module.
No driver or environment change was attempted.

Ordinary bounded disk I/O remains the portable candidate for history storage.
Disk bandwidth, memory headroom and the history/head streaming integration have
not been benchmarked by this support probe.

## Executed local check

The probe ran with the existing
`C:\ProgramData\miniconda3\envs\comfyui\python.exe`, without installing packages,
loading H3, opening a probe data file, or allocating a CUDA tensor.

- Windows 11, build 22631; NVIDIA GeForce RTX 5060 Ti, driver 616.56.
- Torch 2.12.1+cu132, CUDA build 13.2.
- `torch.cuda.gds` imports, but its native operations are Python error stubs.
  Merely testing `hasattr(torch._C, '_gds_load_storage')` is a false positive.
- Executing the `GdsFile` initializer rejects the platform with
  `RuntimeError: GdsFile is not supported on this platform.`
- No `gdscheck.py`, `nvidia-cufile-cu12`, or `nvidia-cufile-cu13` was found.
- The local disk is a WD_BLACK SN7100 NVMe SSD and the volumes are NTFS. Having
  an NVMe disk does not remove the operating-system/API restriction.

The exact result is in [local_probe.json](../data/gds_support_20261027/local_probe.json).
NVIDIA explicitly lists Windows as unsupported on its
[GDS product page](https://developer.nvidia.com/gpudirect-storage/).

## Existing Kaggle evidence

Six saved native-runtime receipts from the October 26–27 VAE, TE and Kitchen
runs contain `nvidia-cufile-cu12==1.13.1.3` alongside native Torch
`2.11.0+cu128`. The sanitized receipt summary is in
[kaggle_saved_packages.json](../data/gds_support_20261027/kaggle_saved_packages.json).
This establishes an installed user-space package on those historical instances;
it does not establish a loaded host driver, native Torch GDS bindings, a supported
filesystem, or a successful direct transfer on the intended scratch path.

The preserved T4 results establish Tesla T4 / SM75 hardware. NVIDIA lists T4 as
GDS-capable, while its newer NVMe PCI P2PDMA path requires Ampere or newer.
See the [device and P2PDMA requirements](https://docs.nvidia.com/gpudirect-storage/troubleshooting-guide/index.html).
The saved receipts do not contain `gdscheck`, host `nvidia_fs` state, or a strict
direct-I/O roundtrip. The previous CPU storage probe is not evidence for these
GPU-host prerequisites. No claim of Kaggle support or lack of support follows.

## Probe and submission provenance

[probe_gds_support.py](../../benchmarks/probe_gds_support.py) defaults to metadata
inspection. Explicit `--gpu-io` launches separate bounded child processes for
the requested devices. Each child writes and reads one 4-MiB synthetic CUDA
storage through `GdsFile` on the selected scratch filesystem, then compares bytes.

The child disables compatibility fallback through a private cuFile configuration
and the documented `CUFILE_ALLOW_COMPAT_MODE=false` /
`CUFILE_FORCE_COMPAT_MODE=false` environment overrides. It records `gdscheck -p`
and any cuFile diagnostics. A successful roundtrip is only labeled verified if
the checker confirms compatibility is disabled and no POSIX-mode receipt appears.
A working roundtrip without that configuration evidence remains unverified.
NVIDIA documents these controls in its
[installation and troubleshooting guide](https://docs.nvidia.com/gpudirect-storage/troubleshooting-guide/index.html).

[build_gds_support_job.py](../../benchmarks/build_gds_support_job.py) prepares a
private, offline, zero-mount Kaggle job. The prepared directory is
`tmp/gds_support_20261027`. Version 1 was accepted at 07:38:04 UTC
(15:38:04 Asia/Shanghai) on October 27 at
[h3-gds-support-check-2026-10-27](https://www.kaggle.com/code/pamcks/h3-gds-support-check-2026-10-27).
Its 0.268-second supervised worker failed before importing Torch or probing
any GPU. The first final collection returned all seven output files;
[attempt1_manifest.json](../data/gds_support_20261027/attempt1_manifest.json)
records their hashes. Version 2 was accepted at 07:49:01 UTC (15:49:01
Asia/Shanghai), with only the requested scratch path changed to `/tmp`.
The probe payload hash is identical. The replacement job directory is
`tmp/gds_support_20261027_retry_v2`.

The probe uses the native
environment, installs nothing, loads no H3 component or model, and uses no
credential. Version 2 checks `/tmp` on devices 0 and 1 with 40-second child
deadlines and a 135-second outer execution budget. Provider startup is outside
that budget. Collection is once per batch, without status or live-log polling.

The user explicitly authorized this one bounded batch after being shown the
[AGENTS.md](../../AGENTS.md) restriction on GPU setup checks. The authorization
does not extend to model generation, dependency installation or another job.
The platform runtime cap is 180 seconds, with the 135-second supervisor budget
inside that cap. No status/list endpoint or live log is polled. One final
collection per version is used to allow provider startup without polling. The
version-2 final collection completed at 07:54:36 UTC (15:54:36 Asia/Shanghai),
once after the startup allowance. No status/list endpoint or live log was polled.
Receipts are [version 1](../data/gds_support_20261027/gpu_submission.json),
[first collection](../data/gds_support_20261027/final_collection.json), and
[version 2](../data/gds_support_20261027/attempt2_gpu_submission.json), with its
[collection receipt](../data/gds_support_20261027/attempt2_final_collection.json)
and [artifact manifest](../data/gds_support_20261027/attempt2_manifest.json).
