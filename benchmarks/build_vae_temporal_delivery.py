"""Overlay current decoder/tools on the immutable, CPU-qualified H3 source base."""
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
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Preserve previously built/submitted sources')
    raw = args.base.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if args.base.name != digest + '.tar.gz':
        raise ValueError('Immutable parent archive checksum mismatch')
    with tarfile.open(fileobj=io.BytesIO(raw), mode='r:gz') as archive:
        distribution = {m.name: archive.extractfile(m).read() for m in archive}
    protected = {k: hashlib.sha256(v).hexdigest() for k, v in distribution.items()
                 if k not in ('capsule.tar.gz.bin','delivery_manifest.json')}
    original = payload_from_distribution(distribution)
    files = dict(original)
    paths = list((ROOT/'patches').glob('*.py')) + list((ROOT/'benchmarks').glob('*.py'))
    paths += list((ROOT/'tools').rglob('*.py'))
    paths += list((ROOT/'custom_nodes/minimax_h3_tp2').rglob('*.py'))
    paths += list((ROOT/'custom_nodes/minimax_h3_tp2/web').rglob('*.js'))
    paths += list((ROOT/'workflows').glob('*tp2*.json'))
    for path in paths:
        files[path.relative_to(ROOT).as_posix()] = path.read_bytes()
    for old in json.loads((ROOT/'tools/path_migration.json').read_text())['paths']:
        files.pop(old, None)
    files.pop('patches/te_loading.py', None)
    credential = re.compile(rb'(?:hf_[A-Za-z0-9]{25,}|gh[pousr]_[A-Za-z0-9]{30,}|KGAT_[A-Za-z0-9_\-]{20,})')
    if any(credential.search(value) for value in files.values()):
        raise RuntimeError('Credential-like source content; public delivery blocked')
    write_distribution(args.output, distribution, files, parent=digest)
    assert protected == {k: hashlib.sha256(v).hexdigest() for k, v in distribution.items() if k in protected}
    (args.output/'vae_binding.json').write_text(json.dumps(dict(
        parent_distribution_sha256=digest, frozen_distribution_unchanged=protected,
        reference_sha256=hashlib.sha256(args.reference.read_bytes()).hexdigest(),
        source_only=True, captured_activations_included=False, credentials_included=False,
        default_query_preserved=True, audio_parallelism=False, full_pipeline=False,
        changed={n:hashlib.sha256(v).hexdigest() for n,v in files.items() if original.get(n)!=v}), indent=2))


if __name__ == '__main__':
    main()
