"""Fail-closed scratch admission and failure evidence; no models or CUDA."""
import json
import os
from pathlib import Path
import shutil
import tempfile
import time
import zipfile


def snapshot(root):
    root = Path(root).resolve()
    result = dict(root=str(root), capacity_is_not_a_reservation=True)
    try:
        usage = shutil.disk_usage(root)
        result.update(total_bytes=usage.total, free_bytes=usage.free, st_dev=root.stat().st_dev)
        if hasattr(os, 'statvfs'):
            stat = os.statvfs(root)
            result.update(available_inodes=stat.f_favail, block_bytes=stat.f_frsize)
    except OSError as error:
        result['error'] = repr(error)
    for name in ('/proc/self/mountinfo', '/proc/self/cgroup'):
        try:
            result[Path(name).name] = Path(name).read_text()
        except OSError:
            pass
    return result


def capacity_probe(root, required_bytes, receipt):
    """Allocate actual blocks, sync, write/read both ends, then release them.

    Unlike truncate/seek, posix_fallocate charges real filesystem allocation.
    A pass is a point-in-time capacity check, not a promise of future capacity
    or a qualification of another CPU/GPU instance. Unsupported allocation is
    a blocker; neither fsync errors nor sparse-file fallbacks are accepted.
    """
    root, receipt = Path(root).resolve(), Path(receipt)
    if type(required_bytes) is not int or required_bytes < 8192:
        raise ValueError('A positive allocation of at least 8192 bytes is required')
    started = time.monotonic()
    result = dict(status='starting', requested_bytes=required_bytes, before=snapshot(root),
                  gpu_used=False, models_loaded=False, scope='This instance and scratch path only')
    descriptor, path = None, None
    try:
        if not hasattr(os, 'posix_fallocate'):
            raise RuntimeError('Real scratch allocation requires posix_fallocate; no sparse fallback')
        descriptor, path = tempfile.mkstemp(prefix='.h3_capacity_', dir=root)
        result['probe_file'] = str(path)
        os.posix_fallocate(descriptor, 0, required_bytes)
        stat = os.fstat(descriptor)
        allocated = stat.st_blocks * 512
        result['allocated_bytes'] = allocated
        if stat.st_size != required_bytes or allocated < required_bytes:
            raise OSError('Scratch probe did not allocate the requested non-sparse blocks')
        payload = b'H3 scratch allocation verification\n' * 64
        for offset in (0, required_bytes - len(payload)):
            os.lseek(descriptor, offset, os.SEEK_SET)
            if os.write(descriptor, payload) != len(payload):
                raise OSError('Incomplete scratch probe write')
        os.fsync(descriptor)
        for offset in (0, required_bytes - len(payload)):
            os.lseek(descriptor, offset, os.SEEK_SET)
            if os.read(descriptor, len(payload)) != payload:
                raise OSError('Scratch probe readback mismatch')
        result.update(status='passed', write_sync_readback=True, during=snapshot(root))
    except BaseException as error:
        result.update(status='failed', error=repr(error), errno=getattr(error, 'errno', None),
                      during=snapshot(root))
        raise
    finally:
        try:
            if descriptor is not None:
                os.close(descriptor)
        finally:
            try:
                # Remove only the exact file returned by mkstemp, never a tree.
                if path is not None:
                    Path(path).unlink()
            except OSError as error:
                result.update(status='failed', cleanup_error=repr(error))
                raise
            finally:
                result.update(seconds=time.monotonic()-started, after=snapshot(root))
                receipt.parent.mkdir(parents=True, exist_ok=True)
                receipt.write_text(json.dumps(result, indent=2), encoding='utf-8')
    return result


def failure_evidence(root, output):
    """Save sizes and small provenance receipts; weight payloads stay on scratch."""
    root, output = Path(root).resolve(), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    result = dict(storage=snapshot(root), groups={}, errors=[], receipts=0)
    with zipfile.ZipFile(output/'partial_bank_receipts.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        def walk_error(error):
            result['errors'].append(repr(error))
        for directory, _, names in os.walk(root, followlinks=False, onerror=walk_error):
            for name in names:
                path = Path(directory)/name
                if path.is_symlink():
                    continue
                relative = path.relative_to(root)
                try:
                    stat = path.stat()
                    group = result['groups'].setdefault(relative.parts[0], dict(files=0, bytes=0, allocated_bytes=0))
                    group['files'] += 1
                    group['bytes'] += stat.st_size
                    group['allocated_bytes'] += getattr(stat, 'st_blocks', 0)*512
                    if (relative.parts[0].startswith('eight_hop_tp2') and name.endswith(('.json', '.json.partial'))
                            and stat.st_size < 4*2**20):
                        archive.write(path, str(relative))
                        result['receipts'] += 1
                except OSError as error:
                    result['errors'].append(dict(path=str(relative), error=repr(error)))
    (output/'storage_failure.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    return result
