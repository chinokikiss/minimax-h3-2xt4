"""Read-only source/space/T4 admission for a single full PTD Bank v2 hop.

Never downloads model weights, bakes shards, uploads Datasets or runs GPU kernels.
A two-block historical T4 subset is not a production checkpoint. A successful
metadata-only check is never described as release-hash-verified.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from patches.ptd.checkpoint import PTDCheckpoint, RELEASE
from patches.ptd.composite_lora import CompositeLoRAStore
from patches.ptd.bake_v2 import _devices_ok, _verify_inputs, estimate_hop_scratch
from patches.tp2_bundle import BundleReader


FORMAT = 'h3-ptd-v2-single-hop-preflight-v1'


def _file_hash(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def inspect_inputs(base_descriptor, checkpoint_path, scratch_dir, *,
                   ordinary_loras=(), layers=50, allow_pruned_adaln=False,
                   verify_ptd_sha256=False, check_t4=False, allow_fixture=False,
                   preverified_checkpoint=None):
    """Inspect source identities and the exact Baker scratch gate, without writes.

    Only the production 50-layer, fixed-release, full-SHA case can be
    source_ready. GPU checking is a separate physical-device gate.
    """
    base_path = Path(base_descriptor).resolve()
    checkpoint_path = Path(checkpoint_path).resolve()
    scratch_path = Path(scratch_dir).resolve()
    issues = []
    report = dict(format=FORMAT, mode='read_only', layers=layers, fixture=allow_fixture,
                  base_descriptor=str(base_path), checkpoint=str(checkpoint_path),
                  scratch_dir=str(scratch_path),
                  pinned_ptd_release=dict(RELEASE),
                  release_sha256_checked=False,
                  cuda_gate='not_checked', source_ready=False,
                  physical_baker_ready=False, scratch=None,
                  ordinary_loras=[dict(path=str(Path(p).resolve()), strength=str(s))
                                  for p, s in ordinary_loras])
    if type(layers) is not int or not 1 <= layers <= 50:
        issues.append('Layer count must be in 1..50')
    if not allow_fixture and layers != 50:
        issues.append('Production preflight requires all 50 DiT layers')
    if not base_path.is_file():
        issues.append('Full Native DiT base descriptor is missing')
    if not checkpoint_path.is_file():
        issues.append('Full trained PTD safetensors checkpoint is missing')
    if not scratch_path.is_dir():
        issues.append('Scratch directory does not exist; no directories are created')
    normalized_loras = []
    for path, strength in ordinary_loras:
        file_path = Path(path).resolve()
        try:
            scale = float(strength)
        except (TypeError, ValueError, OverflowError):
            issues.append('Ordinary LoRA strength is not a finite number: ' + str(file_path))
            continue
        if not math.isfinite(scale):
            issues.append('Ordinary LoRA strength must be finite: ' + str(file_path))
        if not file_path.is_file():
            issues.append('Ordinary LoRA file is missing: ' + str(file_path))
        normalized_loras.append((str(file_path), scale))

    if checkpoint_path.is_file():
        size = checkpoint_path.stat().st_size
        report['checkpoint_bytes'] = size
        if not allow_fixture and size != RELEASE['bytes']:
            issues.append('PTD file size differs from pinned complete 8-hop release')
        if verify_ptd_sha256:
            if preverified_checkpoint is not None:
                # download_on_t4 already verified all bytes. Reuse its SHA only
                # while the full inode/ctime/size identity remains unchanged.
                st = checkpoint_path.stat()
                fingerprint = (st.st_dev, st.st_ino, st.st_size,
                               st.st_mtime_ns, st.st_ctime_ns)
                if (not isinstance(preverified_checkpoint, dict)
                        or preverified_checkpoint.get('sha256') != RELEASE['sha256']
                        or tuple(preverified_checkpoint.get('fingerprint', ())) != fingerprint
                        or preverified_checkpoint.get('verifier') != 'download_on_t4'):
                    issues.append('Preverified PTD release source identity is invalid')
                    digest = None
                else:
                    digest = RELEASE['sha256']
                    report['checkpoint_digest_source'] = 'download_on_t4_same_inode'
            else:
                digest = _file_hash(checkpoint_path)
                report['checkpoint_digest_source'] = 'preflight_full_file_sha256'
            if digest is not None:
                report['checkpoint_sha256'] = digest
                report['release_sha256_checked'] = True
            if not allow_fixture and digest != RELEASE['sha256']:
                issues.append('PTD file SHA256 differs from pinned complete release')
        else:
            issues.append('Pinned PTD SHA256 not checked; re-run with --verify-ptd-sha256')

    if base_path.is_file() and checkpoint_path.is_file() and not any(
            'missing' in item or 'layer' in item.lower() for item in issues):
        try:
            base = BundleReader(base_path)
            checkpoint = PTDCheckpoint(checkpoint_path, base)
            _verify_inputs(base, checkpoint, layers, allow_pruned_adaln, allow_fixture)
            store = CompositeLoRAStore(checkpoint, 0, base, tuple(normalized_loras))
            budget = estimate_hop_scratch(base, store, layers)
            report['base'] = dict(kind=base.kind, bundle_source=base.plan['sources'][base.kind],
                                  descriptor_sha256=_file_hash(base_path),
                                  bundle_index_sha256=base.bundle_digest,
                                  adapted_blocks=checkpoint.blocks,
                                  adapted_refiners=checkpoint.refiners,
                                  hop_count=checkpoint.hops, pdd_intervals=checkpoint.steps,
                                  pruned_adaln_approximation='adaln_t_table' in base.header)
            if scratch_path.is_dir():
                free = shutil.disk_usage(scratch_path).free
                report['scratch'] = dict(**budget, available_bytes=free,
                                         enough_for_one_hop=free >= budget['minimum_free_bytes'],
                                         # Kaggle -r tar can separately materialize a compressed archive.
                                         upload_archive_space_not_included=True)
                if free < budget['minimum_free_bytes']:
                    issues.append('Scratch free space is below the Baker minimum for one hop')
        except (OSError, ValueError, KeyError, RuntimeError, TypeError) as exc:
            issues.append('Native/PTD/ordinary-LoRA metadata admission failed: '
                          + type(exc).__name__ + ': ' + str(exc))

    report['source_ready'] = (not issues and layers == 50 and not allow_fixture
                              and report['release_sha256_checked']
                              and report['scratch'] is not None)
    if check_t4:
        try:
            _devices_ok(('cuda:0', 'cuda:1'), False)
            report['cuda_gate'] = 'two_t4_admitted'
        except (OSError, ValueError, RuntimeError, AssertionError) as exc:
            report['cuda_gate'] = 'blocked'
            issues.append('Physical dual-T4 gate failed: ' + str(exc))

    report['physical_baker_ready'] = (report['source_ready']
                                      and report['cuda_gate'] == 'two_t4_admitted')
    report['status'] = ('passed' if report['physical_baker_ready']
                        else 'source_qualified' if report['source_ready']
                        else 'blocked')
    report['issues'] = issues
    report['pending'] = (['Physical dual-T4 has not been checked on the Baker']
                         if report['cuda_gate'] == 'not_checked' else [])
    report['next_step'] = ('Run the separately authorized single-hop Baker on physical dual T4'
                           if report['physical_baker_ready'] else
                           'Resolve pending gates; no GPU Bake or remote Dataset operations were performed')
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, required=True,
                        help='Full 50-block Native DiT TP2 descriptor')
    parser.add_argument('--checkpoint', type=Path, required=True,
                        help='Pinned 11.11 GB trained 8-hop PTD checkpoint')
    parser.add_argument('--scratch-dir', type=Path, required=True)
    parser.add_argument('--lora', nargs=2, action='append', default=[],
                        metavar=('FILE', 'STRENGTH'))
    parser.add_argument('--layers', type=int, default=50)
    parser.add_argument('--allow-pruned-adaln', action='store_true')
    parser.add_argument('--verify-ptd-sha256', action='store_true',
                        help='Stream the full pinned checkpoint SHA256; required for release admission')
    parser.add_argument('--check-t4', action='store_true',
                        help='Read local CUDA device properties; no kernels are launched')
    parser.add_argument('--output', type=Path,
                        help='New immutable JSON report path; never overwrites')
    args = parser.parse_args(argv)
    if args.output is not None and args.output.exists():
        raise FileExistsError('Preserve the previous PTD preflight report')
    report = inspect_inputs(args.base, args.checkpoint, args.scratch_dir,
                            ordinary_loras=tuple(args.lora), layers=args.layers,
                            allow_pruned_adaln=args.allow_pruned_adaln,
                            verify_ptd_sha256=args.verify_ptd_sha256,
                            check_t4=args.check_t4)
    if args.output is not None:
        from patches.tp2_disk_bake import atomic_json
        if not args.output.parent.is_dir():
            raise FileNotFoundError('Preflight report parent must exist')
        atomic_json(args.output, report)
    print(json.dumps(report, indent=2), flush=True)
    return 0 if report['physical_baker_ready'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
