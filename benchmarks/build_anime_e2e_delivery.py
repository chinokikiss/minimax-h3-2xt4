"""Freeze the current H3 implementation while retaining the validated core."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import re
import tarfile

from build_kaggle_light_delivery import payload_from_distribution, write_distribution

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--plans', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Preserve previous frozen sources and submissions')
    raw = args.base.read_bytes()
    parent = hashlib.sha256(raw).hexdigest()
    assert args.base.name == parent + '.tar.gz'
    with tarfile.open(fileobj=io.BytesIO(raw), mode='r:gz') as archive:
        distribution = {m.name: archive.extractfile(m).read() for m in archive}
    protected = {name: hashlib.sha256(value).hexdigest() for name, value in distribution.items()
                 if name not in ('capsule.tar.gz.bin', 'delivery_manifest.json')}
    files = payload_from_distribution(distribution)
    suffixes = {'.py', '.cu', '.cuh', '.json', '.js'}
    paths = []
    for name in ('patches', 'custom_nodes/minimax_h3_tp2', 'benchmarks', 'tools'):
        paths.extend(p for p in (ROOT/name).rglob('*') if p.is_file() and p.suffix in suffixes
                     and '__pycache__' not in p.parts)
    snapshot = {p.relative_to(ROOT).as_posix(): p.read_bytes() for p in paths}
    assert all((ROOT/name).read_bytes() == value for name, value in snapshot.items()), 'Source changed during snapshot'
    files.update(snapshot)
    for name in ('turbo_plan.json', 'turbo_lora_header.json', 'l1_assets_identity.json'):
        files[name] = (args.plans/name).read_bytes()
    from anime_e2e_contract import validate
    validate(json.loads(files['turbo_plan.json']), json.loads(files['l1_assets_identity.json'])['images'])
    assert files['bundle_assets.json'] == (args.plans/'bundle_assets.json').read_bytes()
    credential = re.compile(rb'(?:hf_[A-Za-z0-9]{25,}|gh[pousr]_[A-Za-z0-9]{30,}|KGAT_[A-Za-z0-9_\-]{20,})')
    assert not any(credential.search(value) for value in files.values()), 'Credential-like public payload'
    write_distribution(args.output, distribution, files, parent=parent)
    assert all(hashlib.sha256(distribution[name]).hexdigest() == value for name, value in protected.items())
    binding = dict(parent_distribution_sha256=parent, frozen_core_unchanged=protected,
                   runtime_snapshot_sha256={name: hashlib.sha256(value).hexdigest() for name, value in snapshot.items()},
                   request_sha256=hashlib.sha256(files['turbo_plan.json']).hexdigest(),
                   full_pipeline_authorized=True, credentials_included=False,
                   new_reference_image_bytes_included=False, dataset_mounts=0, kernel_mounts=0,
                   scope='One exact eight-step anime REF2VA generation; current source, frozen legacy ComfyUI and core pins')
    (args.output/'anime_binding.json').write_text(json.dumps(binding, indent=2))


if __name__ == '__main__':
    main()
