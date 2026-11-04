"""Offline-only immutable PTD Bank v2 part readback verification.

Run against the actual remotely mounted/read-back part, not the source scratch.
The resulting proof is an explicit input to full-model Bank assembly.
Inference deliberately does not import or execute payload SHA256 verification.
"""
import hashlib
import json
import os
from pathlib import Path
import re

from patches.file_metadata import FileChecks
from patches.tp2_bundle import sha
from patches.tp2_disk_bake import SHARD
from patches.windowed_tp import CheckpointReader
from .bake_v2 import PART_FORMAT

READBACK_FORMAT = 'h3-ptd-tp2-part-readback-v1'
BAKE_ADMISSION_FORMAT = 'h3-ptd-v2-bake-receipt-admission-v1'
_KAGGLE_HANDLE = re.compile(r'^[A-Za-z0-9][A-Za-z0-9_-]*/[A-Za-z0-9][A-Za-z0-9_-]*$')
_KAGGLE_PIN = re.compile(r'^[A-Za-z0-9][A-Za-z0-9_-]*/[A-Za-z0-9][A-Za-z0-9_-]*/versions/[1-9][0-9]*$')
_HF_PIN = re.compile(r'^hf:[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+@[a-f0-9]{40}$')
_LICENSE = re.compile(r'^[A-Za-z0-9][A-Za-z0-9_.+-]*$')


def _relative_name(root, name):
    if not isinstance(name, str) or not name or '\\' in name:
        raise ValueError('Invalid PTD part-relative path')
    relative = Path(name)
    if relative.is_absolute() or any(bit in ('', '.', '..') for bit in name.split('/')):
        raise ValueError('PTD part path escapes immutable root')
    resolved = (root / relative).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError('PTD part symlink escapes immutable root')
    return resolved


def _proof_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def _recipe_hash(value):
    # Match the existing Baker recipe digest contract byte for byte.
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def _source_snapshot(root, *, hash_windows=True):
    """Cheap inode/ctime fingerprint. Optional Windows byte-hash for strict readback."""
    result = {}
    for path in sorted(root.iterdir()):
        if path.is_symlink() or not path.is_file() or path.parent != root:
            raise ValueError('PTD verified snapshot requires only regular root files')
        st = path.stat()
        # Windows st_ctime_ns is creation time, unlike Linux inode-change
        # time. Without a reliable change clock, fail closed with SHA there.
        extra = (sha(path),) if os.name == 'nt' and hash_windows else ()
        result[path.name] = (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns,
                             st.st_ctime_ns, st.st_mode) + extra
    return result


def verify_part(root, part_id, *, expected_receipt_sha256=None, published_identity=None,
                capture_source_snapshot=False):
    """Stream SHA256 over one shared/hop part and return a mount-bound proof.

    This function is deliberately expensive and is never called in inference.
    Its strict per-file checks use no GPU and bounded host memory.
    """
    root = Path(root).resolve()
    if not root.is_dir():
        raise FileNotFoundError('PTD immutable part is not mounted: ' + str(root))
    before_snapshot = _source_snapshot(root) if capture_source_snapshot else None
    if part_id != 'shared' and not (part_id.startswith('hop_')
                                    and len(part_id) == 6
                                    and part_id[4:].isdigit()
                                    and 0 <= int(part_id[4:]) < 8):
        raise ValueError('Unrecognized PTD v2 part identifier')
    checks = FileChecks()
    receipt_path = _relative_name(root, 'receipt.json')
    receipt = checks.read_json(receipt_path)
    receipt_digest = sha(receipt_path)
    if expected_receipt_sha256 is not None and receipt_digest != expected_receipt_sha256:
        raise ValueError('PTD part remote receipt differs from the local published source')
    if receipt.get('format') != PART_FORMAT or receipt.get('status') != 'complete':
        raise ValueError('PTD v2 part receipt incomplete or unsupported')
    blobs = {}
    labels = {}

    def verify_file(name, expected_sha256, *, size=None):
        path = _relative_name(root, name)
        checks.admit(path, size=size)
        digest = sha(path)
        if digest != expected_sha256:
            raise ValueError('PTD remote readback SHA256 mismatch: ' + name)
        checks.admit(path, size=size)
        if name in blobs:
            raise ValueError('Duplicate PTD part payload path')
        blobs[name] = dict(sha256=digest, bytes=path.stat().st_size)
        return path

    if part_id == 'shared':
        recipe = receipt.get('recipe')
        if not isinstance(recipe, dict) or _recipe_hash(recipe) != receipt.get('recipe_sha256'):
            raise ValueError('PTD shared part recipe proof mismatch')
        path = verify_file(receipt['aux'], receipt['aux_sha256'], size=receipt['aux_bytes'])
        if not CheckpointReader(path).header:
            raise ValueError('PTD shared static auxiliary has no tensor schema')
        header_path = verify_file(receipt['native_header'], receipt['native_header_sha256'],
                                  size=receipt['native_header_bytes'])
        plan_path = verify_file(receipt['block_plan'], receipt['block_plan_sha256'],
                                size=receipt['block_plan_bytes'])
        header = json.loads(header_path.read_text(encoding='utf-8'))
        plan = json.loads(plan_path.read_text(encoding='utf-8'))
        layers = recipe['layers']
        expected = {f'dit.block{i:02d}.rank{rank}' for i in range(layers) for rank in (0, 1)}
        if (not isinstance(header, dict) or not isinstance(plan, dict)
                or set(plan.get('blocks', {})) != expected):
            raise ValueError('PTD shared Native/rank schema incomplete')
        labels['layers'] = layers
        recipe_sha = receipt['recipe_sha256']
    else:
        hop = int(part_id[4:])
        if receipt.get('hop') != hop:
            raise ValueError('PTD remote hop index differs from its mount identity')
        recipe_sha = receipt['recipe_sha256']
        recipe_path = _relative_name(root, 'recipe.json')
        recipe = checks.read_json(recipe_path)
        if _recipe_hash(recipe) != recipe_sha:
            raise ValueError('PTD remote hop recipe differs from shared source')
        layers = recipe['layers']
        expected = {f'dit.block{i:02d}.rank{rank}' for i in range(layers) for rank in (0, 1)}
        if set(receipt.get('shards', {})) != expected:
            raise ValueError('PTD remote part has incomplete rank coverage')
        blobs['recipe.json'] = dict(sha256=sha(recipe_path), bytes=recipe_path.stat().st_size)
        aux = receipt['auxiliary']
        verify_file(aux['file'], aux['sha256'], size=aux['bytes'])
        for stem, row in sorted(receipt['shards'].items()):
            if (row.get('base_state_sha256') is None
                    or row.get('state_sha256') is None):
                raise ValueError('PTD remote rank tensor proof missing')
            path = verify_file(stem + '.safetensors', row['file_sha256'], size=row['bytes'])
            sidecar = verify_file(stem + '.json', row['manifest_sha256'])
            proof = json.loads(sidecar.read_text(encoding='utf-8'))
            if (proof.get('format') != SHARD
                    or proof.get('recipe_sha256') != recipe_sha
                    or proof.get('state_sha256') != row['state_sha256']
                    or proof.get('base_state_sha256') != row['base_state_sha256']
                    or proof.get('file_sha256') != row['file_sha256']):
                raise ValueError('PTD remote rank state/provenance differs from receipt: ' + stem)
            # Safetensors range/schema parsing, but no tensor materialization.
            CheckpointReader(path)
        labels['layers'] = layers
        labels['hop'] = hop
    checks.check()
    if before_snapshot is not None and _source_snapshot(root) != before_snapshot:
        raise RuntimeError('PTD source changed during local SHA256 verification')
    result = dict(format=READBACK_FORMAT, status='verified', part=part_id,
                  published_identity=published_identity or '',
                  root=str(root), recipe_sha256=recipe_sha,
                  receipt_sha256=receipt_digest,
                  source_receipt_sha256=expected_receipt_sha256 or '',
                  payload_files=len(blobs),
                  payload_bytes=sum(x['bytes'] for x in blobs.values()),
                  payload_tree_sha256=_proof_hash(blobs), **labels)
    if before_snapshot is not None:
        result['source_snapshot'] = before_snapshot
    return result


def admit_baked_part(root, part_id):
    """Admit a fresh Baker part from write-time SHA256 receipts, without rehashing payloads.

    This is *not* an independent SHA256/readback proof. The Baker already fsynced
    and hashed every safetensors/sidecar while writing. Check exact names, sizes,
    cheap source identity and small JSON sidecars before allowing hardlink staging.
    Remote-mounted verify_part remains the final byte-level authority.
    """
    root = Path(root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    if part_id != 'shared' and not (
            part_id.startswith('hop_') and len(part_id) == 6
            and part_id[4:].isdigit() and 0 <= int(part_id[4:]) < 8):
        raise ValueError('Invalid PTD baked-part identifier')
    receipt_path = _relative_name(root, 'receipt.json')
    receipt = json.loads(receipt_path.read_text(encoding='utf-8'))
    if receipt.get('format') != PART_FORMAT or receipt.get('status') != 'complete':
        raise ValueError('Incomplete PTD Baker receipt')
    receipt_sha = sha(receipt_path)
    blobs = {}
    expected_files = {'receipt.json'}

    def admit_file(name, expected_sha, expected_bytes=None, *, small_json=False):
        if (not isinstance(expected_sha, str)
                or not re.fullmatch(r'[0-9a-f]{64}', expected_sha)):
            raise ValueError('Missing write-time file SHA256: ' + name)
        path = _relative_name(root, name)
        if path.is_symlink() or not path.is_file() or path.parent != root:
            raise ValueError('Baked payload must be a regular file: ' + name)
        size = path.stat().st_size
        if size <= 0 or (expected_bytes is not None and size != expected_bytes):
            raise ValueError('Baked payload length changed: ' + name)
        if small_json and sha(path) != expected_sha:
            raise ValueError('Baked JSON sidecar changed: ' + name)
        if name in expected_files:
            raise ValueError('Duplicate baked payload: ' + name)
        expected_files.add(name)
        blobs[name] = dict(sha256=expected_sha, bytes=size)
        return path

    if part_id == 'shared':
        recipe = receipt.get('recipe')
        if not isinstance(recipe, dict) or _recipe_hash(recipe) != receipt.get('recipe_sha256'):
            raise ValueError('Baked shared recipe invalid')
        admit_file(receipt['aux'], receipt['aux_sha256'], receipt['aux_bytes'])
        admit_file(receipt['native_header'], receipt['native_header_sha256'],
                   receipt['native_header_bytes'], small_json=True)
        admit_file(receipt['block_plan'], receipt['block_plan_sha256'],
                   receipt['block_plan_bytes'], small_json=True)
    else:
        if receipt.get('hop') != int(part_id[4:]):
            raise ValueError('Baked Hop receipt index mismatch')
        recipe_path = _relative_name(root, 'recipe.json')
        recipe = json.loads(recipe_path.read_text(encoding='utf-8'))
        if _recipe_hash(recipe) != receipt.get('recipe_sha256'):
            raise ValueError('Baked Hop recipe mismatch')
        admit_file('recipe.json', sha(recipe_path), small_json=True)
        expected = {f'dit.block{i:02d}.rank{rank}'
                    for i in range(recipe['layers']) for rank in (0, 1)}
        if set(receipt.get('shards', {})) != expected:
            raise ValueError('Baked rank-shard coverage mismatch')
        auxiliary = receipt['auxiliary']
        admit_file(auxiliary['file'], auxiliary['sha256'], auxiliary['bytes'])
        for stem, record in sorted(receipt['shards'].items()):
            admit_file(stem + '.safetensors', record['file_sha256'], record['bytes'])
            sidecar = admit_file(stem + '.json', record['manifest_sha256'],
                                 small_json=True)
            info = json.loads(sidecar.read_text(encoding='utf-8'))
            if (info.get('format') != SHARD
                    or info.get('recipe_sha256') != receipt['recipe_sha256']
                    or info.get('file_sha256') != record['file_sha256']
                    or info.get('state_sha256') != record['state_sha256']
                    or info.get('base_state_sha256') != record['base_state_sha256']):
                raise ValueError('Baked shard manifest differs from receipt: ' + stem)
    actual = {item.name for item in root.iterdir()
              if item.is_file() and not item.is_symlink()}
    if actual != expected_files or len(list(root.iterdir())) != len(expected_files):
        raise ValueError('Baked part contains missing, extra or non-regular files')
    snapshot = _source_snapshot(root, hash_windows=False)
    return dict(format=BAKE_ADMISSION_FORMAT, status='bake_receipt_admitted',
                part=part_id, root=str(root), receipt_sha256=receipt_sha,
                recipe_sha256=receipt['recipe_sha256'],
                payload_files=len(blobs),
                payload_bytes=sum(row['bytes'] for row in blobs.values()),
                payload_tree_sha256=_proof_hash(blobs),
                source_snapshot=snapshot, source_hash_policy='write_time_receipt',
                independent_local_payload_sha256=False, remote_readback='pending')


def check_readback_proof(path, root, part_id, receipt_sha256, recipe_sha256,
                         *, require_published_identity=True, expected_layers=None):
    """Check an immutable part's externally saved offline readback proof."""
    record = json.loads(Path(path).read_text(encoding='utf-8'))
    if (record.get('format') != READBACK_FORMAT or record.get('status') != 'verified'
            or record.get('part') != part_id or record.get('root') != str(Path(root).resolve())
            or record.get('receipt_sha256') != receipt_sha256
            or record.get('recipe_sha256') != recipe_sha256):
        raise ValueError('Bank v2 part has no matching remote readback proof')
    if require_published_identity and record.get('source_receipt_sha256') != receipt_sha256:
        raise ValueError('Bank v2 remote proof must pin the independently saved source receipt SHA256')
    if require_published_identity and not (
            isinstance(record.get('published_identity'), str)
            and (_KAGGLE_PIN.fullmatch(record['published_identity'])
                 or _HF_PIN.fullmatch(record['published_identity']))):
        raise ValueError('Bank v2 production requires a version/commit-pinned dataset identity')
    if expected_layers is not None and record.get('layers') != expected_layers:
        raise ValueError('Bank v2 remote readback layer count differs from the shared recipe')
    expected_files = 3 if part_id == 'shared' else 4 * record.get('layers', -1) + 2
    if (type(record.get('payload_files')) is not int or record['payload_files'] != expected_files
            or type(record.get('payload_bytes')) is not int or record['payload_bytes'] <= 0):
        raise ValueError('Incomplete PTD remote readback payload accounting')
    if not isinstance(record.get('payload_tree_sha256'), str) or len(record['payload_tree_sha256']) != 64:
        raise ValueError('Incomplete PTD remote payload-tree proof')
    return record



def prepare_private_kaggle_part(root, part_id, staging, *, dataset_id, license_name,
                                title=None, verified_proof=None):
    """Prepare an explicit, non-uploading Kaggle private-create staging tree.

    Every part byte is a same-filesystem hardlink under payload/, avoiding
    Kaggle's 50 top-level-file limit. The CLI must use -r tar because its
    default directory mode skips payload/. The normal Baker is unmodified.
    This does not authenticate to Kaggle or create a dataset; operators must
    use 'datasets create' without -u.
    Do not edit hardlinked payloads in staging (the source shares their inodes).
    """
    from patches.tp2_disk_bake import atomic_json

    root = Path(root).resolve()
    staging = Path(staging).resolve()
    if not _KAGGLE_HANDLE.fullmatch(dataset_id):
        raise ValueError('Dataset id must be an unversioned Kaggle owner/slug')
    if not _LICENSE.fullmatch(license_name):
        raise ValueError('A valid, explicitly authorized Kaggle license id is required')
    if title is None:
        title = f'H3 PTD v2 {part_id} {sha(root / "receipt.json")[:12]}'
    if not isinstance(title, str) or not title.strip() or len(title) > 100:
        raise ValueError('Kaggle Dataset title must be nonempty and at most 100 characters')
    if (staging == root or staging.is_relative_to(root)
            or root.is_relative_to(staging)):
        raise ValueError('Kaggle upload staging must be separate from the immutable part')
    if staging.exists() or staging.with_name(staging.name + '.partial').exists():
        raise FileExistsError('Refusing to overwrite existing Kaggle staging or partial')
    if not staging.parent.is_dir():
        raise FileNotFoundError('Kaggle staging parent must already exist')
    if not root.is_dir() or root.stat().st_dev != staging.parent.stat().st_dev:
        raise OSError('Kaggle staging must share the source filesystem for zero-copy hardlinks')

    if verified_proof is None:
        proof = verify_part(root, part_id)
    else:
        # Accept either independently hashed local proof or fresh Baker write-time
        # receipt admission. Neither ever substitutes for mounted remote readback.
        proof = verified_proof
        snapshot = proof.get('source_snapshot')
        strict = (proof.get('format') == READBACK_FORMAT and proof.get('status') == 'verified')
        baked = (proof.get('format') == BAKE_ADMISSION_FORMAT
                 and proof.get('status') == 'bake_receipt_admitted'
                 and proof.get('source_hash_policy') == 'write_time_receipt')
        if (not (strict or baked) or proof.get('part') != part_id
                or proof.get('root') != str(root) or not isinstance(snapshot, dict)
                or _source_snapshot(root, hash_windows=strict) != snapshot):
            raise ValueError('PTD verified/admitted source changed or lacks source snapshot')
    receipt_digest = proof['receipt_sha256']
    # Every baked part contains exactly the SHA256-verified payloads and receipt.
    # Refuse extra or nested files so no unrelated data can be uploaded by accident.
    source_files = sorted(root.iterdir(), key=lambda item: item.name)
    if len(source_files) != proof['payload_files'] + 1:
        raise ValueError('Kaggle staging refuses unexpected part-root files')
    if verified_proof is not None and set(p.name for p in source_files) != set(snapshot):
        raise ValueError('PTD verified snapshot payload coverage changed')
    if not any(p.name == 'receipt.json' for p in source_files):
        raise ValueError('Kaggle staging requires a verified source receipt')
    for source in source_files:
        if source.is_symlink() or not source.is_file() or source.parent != root:
            raise ValueError('Kaggle staging accepts only plain part-root files')
    if any(p.name == 'dataset-metadata.json' for p in source_files):
        raise ValueError('Kaggle staging must not reuse preexisting Dataset metadata')

    partial = staging.with_name(staging.name + '.partial')
    partial.mkdir()
    payload = partial / 'payload'
    payload.mkdir()
    for source in source_files:
        stat_before = source.stat()
        if verified_proof is not None:
            expected = snapshot[source.name]
            observed = (stat_before.st_dev, stat_before.st_ino, stat_before.st_size,
                        stat_before.st_mtime_ns, stat_before.st_ctime_ns,
                        stat_before.st_mode) + ((sha(source),) if os.name == 'nt' and strict else ())
            if observed != expected:
                raise ValueError('PTD verified source changed after SHA256 admission')
        destination = payload / source.name
        os.link(source, destination)
        stat_after = source.stat()
        linked = destination.stat()
        if (stat_before.st_dev, stat_before.st_ino, stat_before.st_size,
                stat_before.st_mtime_ns) != (
                stat_after.st_dev, stat_after.st_ino, stat_after.st_size,
                stat_after.st_mtime_ns):
            raise RuntimeError('PTD part changed during Kaggle hardlink staging')
        if (linked.st_dev, linked.st_ino, linked.st_size) != (
                stat_before.st_dev, stat_before.st_ino, stat_before.st_size):
            raise RuntimeError('PTD Kaggle staging did not preserve immutable payload inodes')
    if sha(payload / 'receipt.json') != receipt_digest:
        raise ValueError('Kaggle staging source receipt changed during publication preparation')
    atomic_json(partial / 'dataset-metadata.json', dict(
        id=dataset_id, title=title, licenses=[dict(name=license_name)]))
    partial.replace(staging)
    return dict(status='staged_not_uploaded', part=part_id, source_root=str(root),
                staging=str(staging), dataset_id=dataset_id,
                source_receipt_sha256=receipt_digest,
                payload_bytes=proof['payload_bytes'], payload_subdir='payload',
                upload_command=['kaggle', 'datasets', 'create', '-p', str(staging),
                                '-q', '-t', '-r', 'tar'],
                privacy='Kaggle create defaults to private; never append -u',
                packaging='Kaggle tar mode may require one additional part-sized temporary archive',
                note='After publication, verify a separately mounted version-pinned dataset')
