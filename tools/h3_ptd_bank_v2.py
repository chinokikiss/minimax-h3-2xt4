"""Offline PTD Bank v2 entry point. Do not run from the inference notebook.

Each hop is a bounded, independently publishable unit. This tool can stage
private Kaggle creation and audit mounted remote bytes, but never uploads,
remounts, or reclaims scratch automatically.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _common(parser):
    parser.add_argument('--base', required=True, type=Path)
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--lora', nargs=2, metavar=('FILE', 'STRENGTH'),
                        action='append', default=[])
    parser.add_argument('--layers', type=int, default=50)
    parser.add_argument('--allow-pruned-adaln', action='store_true')
    parser.add_argument('--fixture', action='store_true')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parts = parser.add_subparsers(dest='command', required=True)
    shared = parts.add_parser('shared', help='Bake shared static auxiliary and Native schema once')
    _common(shared)
    shared.add_argument('--output', type=Path, required=True)
    hop = parts.add_parser('hop', help='Bake only one immutable PTD hop on two T4 GPUs')
    _common(hop)
    hop.add_argument('--hop', type=int, required=True)
    hop.add_argument('--output', type=Path, required=True)
    hop.add_argument('--devices', nargs='+', default=['cuda:0', 'cuda:1'])
    assemble = parts.add_parser('assemble', help='Admit all mounted parts and publish bank manifest last')
    assemble.add_argument('--manifest', required=True, type=Path)
    assemble.add_argument('--shared-root', required=True, type=Path)
    assemble.add_argument('--hop-roots', required=True, type=Path, nargs=8)
    assemble.add_argument('--fixture', action='store_true')
    assemble.add_argument('--readback-proofs', nargs=9, type=Path,
                          help='Shared, then hop00..hop07 remote readback proof JSON paths')
    verify = parts.add_parser('verify-part', help='Full SHA256 remote readback of one mounted immutable part')
    verify.add_argument('--root', type=Path, required=True)
    verify.add_argument('--part-id', required=True)
    verify.add_argument('--expected-receipt-sha256', default=None)
    verify.add_argument('--published-identity', default=None)
    verify.add_argument('--output-proof', type=Path, required=True)
    kaggle = parts.add_parser('prepare-kaggle',
                              help='Verify and hardlink-stage one private Kaggle Dataset, without uploading')
    kaggle.add_argument('--root', type=Path, required=True)
    kaggle.add_argument('--part-id', required=True)
    kaggle.add_argument('--staging', type=Path, required=True)
    kaggle.add_argument('--dataset-id', required=True, help='Unversioned Kaggle owner/slug')
    kaggle.add_argument('--license', required=True, dest='license_name',
                        help='Kaggle license id authorized for these model weights')
    kaggle.add_argument('--title', default=None)
    args = parser.parse_args(argv)

    if args.command == 'prepare-kaggle':
        from patches.ptd.publish_v2 import prepare_private_kaggle_part
        result = prepare_private_kaggle_part(
            args.root, args.part_id, args.staging, dataset_id=args.dataset_id,
            license_name=args.license_name, title=args.title)
        print(json.dumps(result, indent=2), flush=True)
        return 0
    if args.command == 'verify-part':
        from patches.ptd.publish_v2 import verify_part
        from patches.tp2_disk_bake import atomic_json
        result = verify_part(args.root, args.part_id,
                             expected_receipt_sha256=args.expected_receipt_sha256,
                             published_identity=args.published_identity)
        proof_path = args.output_proof.resolve()
        if proof_path.exists():
            raise FileExistsError('Refusing to overwrite a previous PTD readback proof')
        atomic_json(proof_path, result)
        print(proof_path, flush=True)
        return 0
    if args.command == 'assemble':
        from patches.ptd.bake_v2 import assemble_bank_v2
        result = assemble_bank_v2(args.manifest, args.shared_root, args.hop_roots,
                                  allow_fixture=args.fixture, readback_proofs=args.readback_proofs)
    else:
        from patches.ptd.bake_v2 import bake_hop_part, bake_shared_part
        kwargs = dict(ordinary_loras=tuple((str(Path(path).resolve()), float(strength))
                                          for path, strength in args.lora),
                      layers=args.layers, allow_pruned_adaln=args.allow_pruned_adaln,
                      allow_fixture=args.fixture)
        if args.command == 'shared':
            result = bake_shared_part(args.base, args.checkpoint, args.output, **kwargs)
        else:
            result = bake_hop_part(args.base, args.checkpoint, args.hop, args.output,
                                   devices=tuple(args.devices), **kwargs)
    print(result, flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
