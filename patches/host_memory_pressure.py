"""Small cgroup-v2 snapshots for admission; no profiler or background thread."""
import os
from pathlib import Path
import time


def _read(path):
    try:
        return path.read_text().strip()
    except OSError:
        return ''


def _numbers(path):
    result = {}
    for line in _read(path).splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[1].isdigit():
            result[parts[0]] = int(parts[1])
    return result


class CgroupMemory:
    def __init__(self, root=None):
        self.root = Path(root) if root is not None else self._discover()
        self.cached = {}; self.updated = 0.

    @staticmethod
    def _discover():
        if os.name != 'posix':
            return None
        base = Path('/sys/fs/cgroup')
        # Prefer the process's cgroup when the mount is not namespaced.
        for line in _read(Path('/proc/self/cgroup')).splitlines():
            if line.startswith('0::'):
                candidate = base / line[3:].lstrip('/')
                if (candidate / 'memory.current').exists():
                    return candidate
        return base if (base / 'memory.current').exists() else None

    def snapshot(self, fresh=False):
        now = time.monotonic()
        if not fresh and now-self.updated < .1:
            return self.cached
        root = self.root
        limit = _read(root/'memory.max') if root is not None else ''
        if not limit.isdigit():
            self.cached = {'available': False}
        else:
            stat = _numbers(root/'memory.stat')
            current = _read(root/'memory.current')
            if not current.isdigit():
                self.cached = {'available': False}
                return self.cached
            file_bytes = max(0, stat.get('file', 0)-stat.get('shmem', 0))
            psi = {}
            for line in _read(root/'memory.pressure').splitlines():
                parts = line.split()
                for item in parts[1:]:
                    k, _, v = item.partition('=')
                    try: psi[parts[0]+'_'+k] = float(v)
                    except ValueError: pass
            self.cached = dict(available=True, limit=int(limit), current=int(current),
                file_bytes=file_bytes, nonfile_bytes=max(0, int(current)-file_bytes),
                anon=stat.get('anon'), kernel=stat.get('kernel'), shmem=stat.get('shmem'),
                refault_file=stat.get('workingset_refault_file', 0),
                scan_kswapd=stat.get('pgscan_kswapd', 0), events=_numbers(root/'memory.events'),
                psi=psi)
        self.updated = now
        return self.cached
