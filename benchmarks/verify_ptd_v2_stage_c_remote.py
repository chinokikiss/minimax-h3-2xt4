"""Independent, CPU-only SHA256 audit for Kaggle-mounted PTD Bank v2 parts.

Produces the same mount-bound proof fields as patches.ptd.publish_v2.verify_part,
without importing torch or the inference runtime. An authenticated Dataset
version check is a separate gate: published_identity is caller-supplied.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import struct

PART_FORMAT = 'h3-ptd-tp2-part-v2'
BANK_FORMAT = 'h3-ptd-tp2-bank-v2'
SHARD_FORMAT = 'h3-tp2-disk-baked-shard-v1'
READBACK_FORMAT = 'h3-ptd-tp2-part-readback-v1'
PIN_RE = re.compile(r'^([A-Za-z0-9][A-Za-z0-9_-]*/[A-Za-z0-9][A-Za-z0-9_-]*)/versions/([1-9][0-9]*)$')


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def hash_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def recipe_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def header_only(path):
    """Validate safetensors header, offsets and bounds without loading tensors."""
    path = Path(path)
    size = path.stat().st_size
    if size < 10:
        raise ValueError('Safetensors file is truncated: ' + path.name)
    with path.open('rb') as stream:
        length = struct.unpack('<Q', stream.read(8))[0]
        if not 2 <= length <= min(128 * 1024 * 1024, size - 8):
            raise ValueError('Invalid safetensors header size: ' + path.name)
        header = json.loads(stream.read(length))
    if not isinstance(header, dict) or len(header) == 0:
        raise ValueError('Invalid safetensors header: ' + path.name)
    data_size = size - 8 - length
    count = 0
    for name, row in header.items():
        if name == '__metadata__':
            continue
        if not isinstance(row, dict) or not isinstance(row.get('shape'), list):
            raise ValueError('Invalid safetensors tensor schema: ' + name)
        bounds = row.get('data_offsets')
        if (not isinstance(bounds, list) or len(bounds) != 2
                or any(type(value) is not int for value in bounds)
                or not 0 <= bounds[0] <= bounds[1] <= data_size):
            raise ValueError('Invalid safetensors data bounds: ' + name)
        count += 1
    if count == 0:
        raise ValueError('Safetensors has no tensors: ' + path.name)
    return count


def verify(root, part, source_sha256, recipe_sha256, pinned_identity, *, expected_layers=50):
    root = Path(root).resolve(strict=True)
    if (not root.is_dir() or not re.fullmatch(r'[a-f0-9]{64}', source_sha256)
            or not re.fullmatch(r'[a-f0-9]{64}', recipe_sha256)):
        raise ValueError('Mounted root or pinned local digest is invalid')
    if type(expected_layers) is not int or not 1 <= expected_layers <= 50:
        raise ValueError('Invalid expected PTD layer count')
    match = PIN_RE.fullmatch(pinned_identity)
    if match is None:
        raise ValueError('Publication identity must pin Kaggle Dataset /versions/N')
    if part != 'shared' and not re.fullmatch(r'hop_0[0-7]', part):
        raise ValueError('Unexpected PTD part ID')
    required = {'receipt.json'}
    blobs = {}

    def audit(name, sha, *, size=None, safetensors=False):
        if (not isinstance(name, str) or not name or '/' in name or '\\' in name
                or name in ('..', '.') or name in blobs
                or not re.fullmatch(r'[a-f0-9]{64}', sha)):
            raise ValueError('Invalid file name or expected SHA256: ' + str(name))
        path = root / name
        if path.is_symlink() or not path.is_file() or path.resolve() != path:
            raise ValueError('Mounted part file is absent or unsafe: ' + name)
        count = path.stat().st_size
        if size is not None and count != size:
            raise ValueError('Mounted part file size differs: ' + name)
        if digest(path) != sha:
            raise ValueError('Mounted part SHA256 mismatch: ' + name)
        if safetensors:
            header_only(path)
        required.add(name)
        blobs[name] = dict(sha256=sha, bytes=count)
        return path

    receipt_path = root / 'receipt.json'
    if receipt_path.is_symlink() or not receipt_path.is_file():
        raise FileNotFoundError('Mounted part receipt is absent')
    if digest(receipt_path) != source_sha256:
        raise ValueError('Mounted source receipt SHA256 mismatch')
    receipt = read_json(receipt_path)
    if receipt.get('format') != PART_FORMAT or receipt.get('status') != 'complete':
        raise ValueError('PTD part receipt status or format differs')
    if receipt.get('recipe_sha256') != recipe_sha256:
        raise ValueError('PTD recipe SHA256 differs from pre-upload source')

    if part == 'shared':
        recipe = receipt.get('recipe')
        if not isinstance(recipe, dict) or recipe_hash(recipe) != recipe_sha256:
            raise ValueError('Shared recipe hash differs')
        if recipe.get('format') != BANK_FORMAT or recipe.get('layers') != expected_layers:
            raise ValueError('Shared part must contain complete 50-layer recipe')
        aux = audit(receipt['aux'], receipt['aux_sha256'],
                    size=receipt['aux_bytes'], safetensors=True)
        header_path = audit(receipt['native_header'], receipt['native_header_sha256'],
                            size=receipt['native_header_bytes'])
        plan_path = audit(receipt['block_plan'], receipt['block_plan_sha256'],
                          size=receipt['block_plan_bytes'])
        header, plan = read_json(header_path), read_json(plan_path)
        expected = {f'dit.block{i:02d}.rank{rank}' for i in range(expected_layers) for rank in (0, 1)}
        if not isinstance(header, dict) or not isinstance(plan, dict) or set(plan.get('blocks', {})) != expected:
            raise ValueError('Native shared header or rank plan incomplete')
        _ = aux
        labels = {'layers': expected_layers}
    else:
        hop = int(part[-2:])
        if receipt.get('hop') != hop:
            raise ValueError('Mounted hop index differs')
        recipe_path = root / 'recipe.json'
        if recipe_path.is_symlink() or not recipe_path.is_file():
            raise ValueError('Hop recipe is not a normal file')
        recipe = read_json(recipe_path)
        if recipe_hash(recipe) != recipe_sha256 or recipe.get('format') != BANK_FORMAT or recipe.get('layers') != expected_layers:
            raise ValueError('Mounted hop recipe is not the pinned full-model recipe')
        audit('recipe.json', digest(recipe_path))
        expected = {f'dit.block{i:02d}.rank{rank}' for i in range(expected_layers) for rank in (0, 1)}
        if set(receipt.get('shards', {})) != expected:
            raise ValueError('Mounted hop lacks complete 100-rank shard coverage')
        aux = receipt['auxiliary']
        audit(aux['file'], aux['sha256'], size=aux['bytes'], safetensors=True)
        for stem, item in sorted(receipt['shards'].items()):
            if not item.get('state_sha256') or not item.get('base_state_sha256'):
                raise ValueError('Missing rank tensor state proof: ' + stem)
            audit(stem + '.safetensors', item['file_sha256'],
                  size=item['bytes'], safetensors=True)
            sidecar = audit(stem + '.json', item['manifest_sha256'])
            entry = read_json(sidecar)
            if (entry.get('format') != SHARD_FORMAT
                    or entry.get('recipe_sha256') != recipe_sha256
                    or entry.get('state_sha256') != item['state_sha256']
                    or entry.get('base_state_sha256') != item['base_state_sha256']
                    or entry.get('file_sha256') != item['file_sha256']):
                raise ValueError('Rank state/manifest provenance mismatch: ' + stem)
        labels = {'layers': expected_layers, 'hop': hop}
    found = {p.name for p in root.iterdir() if p.is_file()}
    if found != required or any(not p.is_file() for p in root.iterdir()):
        raise ValueError('Unexpected or missing immutable payload files')
    expected_files = 3 if part == 'shared' else 4 * expected_layers + 2
    if len(blobs) != expected_files:
        raise ValueError('Incomplete PTD readback file count')
    return dict(format=READBACK_FORMAT, status='verified', part=part,
                published_identity=pinned_identity, root=str(root),
                recipe_sha256=recipe_sha256, receipt_sha256=source_sha256,
                source_receipt_sha256=source_sha256,
                payload_files=len(blobs),
                payload_bytes=sum(item['bytes'] for item in blobs.values()),
                payload_tree_sha256=hash_json(blobs),
                verifier='standalone-stage-c-mount-sha256-v1',
                version_identity_source='operator-pinned; independently authenticate provider metadata',
                **labels)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--part', required=True)
    p.add_argument('--receipt-sha256', required=True)
    p.add_argument('--recipe-sha256', required=True)
    p.add_argument('--published-identity', required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError('Preserve previous readback proof')
    result = verify(a.root, a.part, a.receipt_sha256,
                    a.recipe_sha256, a.published_identity)
    a.output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print('STAGE_C_REMOTE_READBACK_PASS', a.part, result['payload_files'],
          result['payload_bytes'], flush=True)


if __name__ == '__main__':
    main()
