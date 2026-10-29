"""CPU-only, bounded writeback capacity measurement of mounted writable disks.

Only fresh probe files are touched. No raw-device writes, mounts, model weights,
package installs, fallocate, truncation or sparse-file shortcuts are used.
"""
import argparse
import errno
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import time

GIB = 2**30
PSEUDO = {'proc', 'sysfs', 'cgroup', 'cgroup2', 'devtmpfs', 'devpts', 'tmpfs',
          'mqueue', 'securityfs', 'debugfs', 'tracefs', 'pstore', 'hugetlbfs',
          'configfs', 'fusectl', 'bpf', 'autofs', 'ramfs', 'nsfs', 'squashfs'}


def mounts(text):
    rows = []
    for line in text.splitlines():
        left, right = line.split(' - ', 1)
        fields, tail = left.split(), right.split()
        unescape = lambda s: re.sub(r'\\([0-7]{3})', lambda m: chr(int(m[1], 8)), s)
        rows.append(dict(mount=unescape(fields[4]), root=unescape(fields[3]),
            device=fields[2], options=fields[5], filesystem=tail[0],
            source=unescape(tail[1]), super_options=','.join(tail[2:])))
    return rows


def usage(path):
    value = shutil.disk_usage(path)
    row = dict(total_bytes=value.total, free_bytes=value.free, used_bytes=value.used,
               st_dev=Path(path).stat().st_dev)
    if hasattr(os, 'statvfs'):
        stat = os.statvfs(path)
        row.update(available_inodes=stat.f_favail, block_bytes=stat.f_frsize)
    return row


def inventory():
    raw = Path('/proc/self/mountinfo').read_text()
    table = mounts(raw)
    paths = list(dict.fromkeys(['/kaggle/working', '/kaggle/temp', '/tmp', '/var/tmp',
                               '/kaggle/huggingface', str(Path.home()),
                               *(r['mount'] for r in table)]))
    entries = []
    for name in paths:
        path = Path(name)
        if not path.is_dir():
            continue  # File bind mounts such as /etc/hosts are not scratch disks.
        matches = [r for r in table if name == r['mount'] or name.startswith(r['mount'].rstrip('/')+'/')]
        mount = max(matches, key=lambda r: len(r['mount']))
        row = dict(path=name, mount=mount)
        try:
            row.update(usage(path))
            row['writable'] = 'rw' in mount['options'].split(',') and os.access(path, os.W_OK | os.X_OK)
        except OSError as error:
            row.update(writable=False, error=repr(error))
        row['disk_candidate'] = row.get('writable', False) and mount['filesystem'] not in PSEUDO
        entries.append(row)
    commands = {}
    for argv in (['df', '-B1', '-T'], ['lsblk', '-b', '-o', 'NAME,SIZE,TYPE,RO,FSTYPE,MOUNTPOINTS']):
        try:
            p = subprocess.run(argv, capture_output=True, text=True, timeout=10)
            commands[argv[0]] = dict(returncode=p.returncode, stdout=p.stdout, stderr=p.stderr)
        except (OSError, subprocess.TimeoutExpired) as error:
            commands[argv[0]] = dict(error=repr(error))
    return dict(paths=entries, mountinfo=raw, commands=commands)


def select_disks(entries, report_device):
    groups = {}
    for row in entries:
        if row.get('disk_candidate'):
            groups.setdefault(row['st_dev'], []).append(row)
    chosen = []
    for device, aliases in groups.items():
        # Probe designated scratch paths first, with one fill per actual st_dev.
        order = {'/kaggle/working': 0, '/tmp': 1, '/kaggle/temp': 2, '/var/tmp': 3}
        best = min(aliases, key=lambda r: (order.get(r['path'], 10), len(r['path'])))
        chosen.append(dict(device=device, base=best['path'], filesystem=best['mount']['filesystem'],
                           source=best['mount']['source'], aliases=[r['path'] for r in aliases],
                           report_filesystem=device == report_device))
    # Fill the root overlay last: writeback failure can affect its other paths.
    return sorted(chosen, key=lambda r: (r['filesystem'] == 'overlay', not r['report_filesystem'], r['base']))


def write_chunk(path, size, deadline, buffer_bytes=8*2**20):
    digest = hashlib.sha256()
    written = 0
    with path.open('xb', buffering=0) as stream:
        while written < size:
            if time.monotonic() >= deadline:
                raise TimeoutError('Disk probe write deadline')
            payload = os.urandom(min(buffer_bytes, size-written))
            digest.update(payload)
            view = memoryview(payload)
            while view:
                count = stream.write(view)
                if not count:
                    raise OSError('Zero-length disk probe write')
                written += count
                view = view[count:]
        os.fsync(stream.fileno())
    verified = hashlib.sha256()
    with path.open('rb', buffering=0) as stream:
        while True:
            if time.monotonic() >= deadline:
                raise TimeoutError('Disk probe readback deadline')
            payload = stream.read(buffer_bytes)
            if not payload:
                break
            verified.update(payload)
    if verified.digest() != digest.digest() or path.stat().st_size != size:
        raise OSError('Disk probe full-chunk SHA256 readback mismatch')
    return dict(file=path.name, bytes=size, sha256=digest.hexdigest(), synced=True, readback_verified=True)


def measure(base, output, *, cap_bytes=512*GIB, deadline_seconds=1500,
            chunk_bytes=128*2**20, report_headroom=512*2**20):
    base, output = Path(base).resolve(), Path(output).resolve()
    if min(cap_bytes, deadline_seconds, chunk_bytes) <= 0 or chunk_bytes > 128*2**20:
        raise ValueError('Positive bounded probe sizes and deadline required')
    result = dict(status='starting', base=str(base), before=usage(base),
        method='random full writes + fsync + full SHA256 readback per retained chunk',
        cap_bytes=cap_bytes, chunk_bytes=chunk_bytes, completed_chunks=[], verified_synced_bytes=0,
        exact_capacity=False, report_headroom_bytes=report_headroom,
        meaning='Additional simultaneous verified payload on this instance, not total device capacity')
    started = time.monotonic()
    deadline = started + deadline_seconds
    owned, paths = None, []
    def save():
        result['seconds'] = time.monotonic()-started
        temporary = output.with_suffix('.partial')
        temporary.write_text(json.dumps(result, indent=2), encoding='utf-8')
        temporary.replace(output)
    try:
        save()
        owned = Path(tempfile.mkdtemp(prefix='h3_write_capacity_', dir=base)).resolve()
        if owned.parent != base or not owned.name.startswith('h3_write_capacity_'):
            raise RuntimeError('Probe directory escaped the selected filesystem')
        result['owned_directory'] = str(owned)
        while result['verified_synced_bytes'] < cap_bytes:
            if time.monotonic() >= deadline:
                result.update(status='lower_bound', stop_reason='deadline')
                break
            size = min(chunk_bytes, cap_bytes-result['verified_synced_bytes'])
            if base.stat().st_dev == output.parent.stat().st_dev:
                available = shutil.disk_usage(base).free-report_headroom
                if available < size:
                    result.update(status='lower_bound', stop_reason='report_filesystem_headroom',
                                  free_at_stop=shutil.disk_usage(base).free)
                    break
            path = owned/f'chunk_{len(paths):05d}.bin'
            paths.append(path)
            result['active_file'] = str(path)
            save()
            chunk = write_chunk(path, size, deadline)
            result['completed_chunks'].append(chunk)
            result['verified_synced_bytes'] += size
            save()
            if len(paths) % 16 == 0:
                print('DISK_VERIFIED', str(base), result['verified_synced_bytes'], flush=True)
        else:
            result.update(status='lower_bound', stop_reason='configured_byte_cap')
    except (OSError, TimeoutError) as error:
        boundary = getattr(error, 'errno', None) in (errno.ENOSPC, getattr(errno, 'EDQUOT', 122))
        result.update(status='observed_boundary' if boundary else 'writeback_error',
            stop_reason='ENOSPC_or_EDQUOT' if boundary else type(error).__name__,
            error=repr(error), errno=getattr(error, 'errno', None))
        # A file that failed sync/readback is never counted as durable capacity.
        if paths:
            try:
                stat = paths[-1].stat()
                result['unqualified_last_file_bytes'] = stat.st_size
            except OSError as stat_error:
                result['last_file_stat_error'] = repr(stat_error)
    finally:
        try:
            result['at_stop'] = usage(base)
            result['mountinfo_at_stop'] = Path('/proc/self/mountinfo').read_text() if os.name == 'posix' else None
        except OSError as error:
            result['snapshot_error'] = repr(error)
        save()  # Preserve boundary evidence before attempting cleanup.
        result['cleanup_errors'] = []
        for path in paths:
            try:
                if owned is None or path.parent != owned or owned.parent != base:
                    raise RuntimeError('Refusing cleanup outside the exact owned directory')
                path.unlink(missing_ok=True)
            except OSError as error:
                result['cleanup_errors'].append(dict(file=str(path), error=repr(error)))
        if owned is not None:
            try:
                owned.rmdir()  # No recursive deletion or removal of unrelated files.
            except OSError as error:
                result['cleanup_errors'].append(dict(directory=str(owned), error=repr(error)))
        try:
            result['after_cleanup'] = usage(base)
        except OSError as error:
            result['after_cleanup_error'] = repr(error)
        save()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cap-gib', type=int, default=512)
    parser.add_argument('--budget', type=int, default=3000)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    state = dict(status='running', cpu_only=True, packages_installed=False, raw_devices_written=False,
        inventory=inventory(), disks=[], caveats=[
            'Writable paths sharing st_dev are aliases, not additive disks; per-directory quotas can still differ.',
            'Visible unmounted block devices are inventoried only, never mounted or written.',
            'tmpfs/shm and read-only mounts are excluded from disk fill tests.',
            'Verified bytes are an observed additional-capacity lower bound, never a future reservation.',
            'Report filesystem retains 512 MiB; byte cap/deadline stops do not establish a disk limit.',
            'EIO/emergency read-only is a writeback fault, not a measured exact quota.',
            'CPU instance storage does not qualify another T4 instance.'])
    def save():
        (args.output/'result.json').write_text(json.dumps(state, indent=2), encoding='utf-8')
    save()
    report_device = args.output.stat().st_dev
    state['selected_disks'] = select_disks(state['inventory']['paths'], report_device)
    save()
    for index, disk in enumerate(state['selected_disks']):
        remaining = args.budget-(time.monotonic()-started)
        if remaining <= 10:
            state['disks'].append(dict(**disk, status='not_tested', reason='overall_deadline'))
            continue
        print('DISK_START', json.dumps(disk), flush=True)
        receipt = args.output/f'disk_{index:02d}.json'
        row = measure(disk['base'], receipt, cap_bytes=args.cap_gib*GIB,
                      deadline_seconds=min(1500, remaining-5))
        state['disks'].append(dict(**disk, receipt=receipt.name, status=row['status'],
            verified_synced_bytes=row['verified_synced_bytes'], stop_reason=row['stop_reason'],
            cleanup_errors=row['cleanup_errors']))
        save()
        print('DISK_END', json.dumps(state['disks'][-1]), flush=True)
    state.update(status='complete', seconds=time.monotonic()-started)
    save()


if __name__ == '__main__':
    main()
