"""Bounded, standard-library-only filesystem probe; never loads H3 or uses CUDA."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import ctypes
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import posixpath
import random
import shutil
import subprocess
import sys
import time

MIB = 2**20
GIB = 2**30
WARNING = ('First pass and DONTNEED hints do not establish physically cold storage. '
           'Kernel, filesystem, host and remote caches can remain warm. '
           'This is CPU filesystem I/O, not H3 load latency or GPU-instance qualification.')


def write(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.new')
    temporary.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def text(path):
    try:
        return Path(path).read_text()
    except OSError:
        return None


def mount_for(path, mountinfo):
    path = posixpath.normpath(str(path))
    matches = []
    for line in mountinfo.splitlines():
        left, right = line.split(' - ', 1)
        fields = left.split(); tail = right.split()
        mount = fields[4].replace('\\040', ' ')
        if path == mount or path.startswith(mount.rstrip('/') + '/'):
            matches.append(dict(mount_point=mount, major_minor=fields[2],
                                root=fields[3], mount_options=fields[5],
                                filesystem=tail[0], source=tail[1], super_options=tail[2:]))
    return max(matches, key=lambda v: len(v['mount_point'])) if matches else None


def inventory():
    mounts = text('/proc/self/mountinfo') or ''
    paths = []
    for name in ('/', '/kaggle/input', '/kaggle/working', '/kaggle/temp', '/tmp', '/dev/shm'):
        path = Path(name)
        row = dict(path=name, exists=path.exists())
        if path.exists():
            usage = shutil.disk_usage(path)
            row.update(total_bytes=usage.total, used_bytes=usage.used, free_bytes=usage.free,
                       st_dev=path.stat().st_dev, mount=mount_for(path.resolve().as_posix(), mounts) if mounts else None)
        paths.append(row)
    commands = {}
    for command in (['df', '-B1', '-T'], ['lsblk', '-b', '-o', 'NAME,SIZE,TYPE,ROTA,RO,FSTYPE,MOUNTPOINTS']):
        try:
            result = subprocess.run(command, capture_output=True, text=True, timeout=10)
            commands[command[0]] = dict(exit_code=result.returncode, stdout=result.stdout, stderr=result.stderr)
        except (OSError, subprocess.TimeoutExpired) as error:
            commands[command[0]] = dict(unavailable=type(error).__name__)
    versions = {}
    for name in ('torch', 'numpy', 'torchvision', 'torchaudio', 'comfy-kitchen', 'tilelang', 'kaggle'):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return dict(python=sys.version, platform=platform.platform(), cpu_count=os.cpu_count(),
                paths=paths, commands=commands, packages=versions, mountinfo=mounts,
                meminfo=text('/proc/meminfo'), cgroup=text('/proc/self/cgroup'),
                memory_max=text('/sys/fs/cgroup/memory.max'),
                cpu_max=text('/sys/fs/cgroup/cpu.max'),
                caveat='Filesystem capacity can differ from Kaggle saved-output quota; exposed block devices do not prove physical hardware.')


def select_files(root, total=2*GIB, per_file=512*MIB, minimum=MIB):
    candidates = []
    for base, dirs, files in os.walk(root, followlinks=False):
        for name in files:
            path = Path(base)/name
            if path.is_symlink():
                continue
            try:
                size = path.stat().st_size
            except OSError:
                continue
            if size >= minimum:
                candidates.append((size, str(path)))
    entries = []; remaining = total
    for size, path in sorted(candidates, key=lambda v: (-v[0], v[1])):
        length = min(size, per_file, remaining)
        if length <= 0:
            break
        entries.append(dict(path=path, offset=0, length=length, original_file_bytes=size))
        remaining -= length
        if not remaining:
            break
    if not entries:
        raise RuntimeError('No mounted dataset files large enough for the read probe')
    return entries, dict(eligible_files=len(candidates), selected_files=len(entries),
                         selected_bytes=sum(e['length'] for e in entries), requested_bytes=total)


def partitions(entries, workers):
    total = sum(e['length'] for e in entries)
    result = [[] for _ in range(workers)]
    cursor = 0
    for entry in entries:
        end = cursor + entry['length']
        for rank in range(workers):
            a, b = total*rank//workers, total*(rank+1)//workers
            lo, hi = max(cursor, a), min(end, b)
            if hi > lo:
                result[rank].append(dict(entry, offset=entry['offset']+lo-cursor, length=hi-lo))
        cursor = end
    return result


def cache_state(entries):
    if sys.platform != 'linux':
        return dict(available=False, reason='Linux mincore only')
    libc = ctypes.CDLL(None, use_errno=True)
    libc.mmap.restype = ctypes.c_void_p
    libc.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int,
                          ctypes.c_int, ctypes.c_int, ctypes.c_longlong]
    libc.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p]
    libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    pagesize = os.sysconf('SC_PAGE_SIZE'); pages = resident = 0; errors = []
    for entry in entries:
        start = entry['offset']//pagesize*pagesize
        length = entry['length']+entry['offset']-start
        count = math.ceil(length/pagesize)
        fd = os.open(entry['path'], os.O_RDONLY)
        try:
            address = libc.mmap(None, length, 1, 1, fd, start)
            if address == ctypes.c_void_p(-1).value:
                errors.append(ctypes.get_errno()); continue
            try:
                vector = (ctypes.c_ubyte*count)()
                if libc.mincore(address, length, vector):
                    errors.append(ctypes.get_errno()); continue
                pages += count; resident += sum(v & 1 for v in vector)
            finally:
                libc.munmap(address, length)
        finally:
            os.close(fd)
    return dict(available=bool(pages), pages=pages, resident_pages=resident,
                resident_fraction=resident/pages if pages else None, errors=errors,
                scope='Observed inode page residency only; no claim about lower-layer caches')


def advise(entries):
    if not hasattr(os, 'posix_fadvise'):
        return dict(available=False)
    accepted = 0; errors = []
    for entry in entries:
        fd = os.open(entry['path'], os.O_RDONLY)
        try:
            try:
                os.posix_fadvise(fd, entry['offset'], entry['length'], os.POSIX_FADV_DONTNEED)
                accepted += 1
            except OSError as error:
                errors.append(dict(errno=error.errno, message=str(error)))
        finally:
            os.close(fd)
    return dict(available=True, accepted_files=accepted, errors=errors, guarantee='Advisory only')


def proc_io():
    value = text('/proc/self/io')
    return {line.split(':')[0]:int(line.split(':')[1]) for line in value.splitlines()} if value else {}


def measure(plan):
    entries = plan['entries']; workers = plan.get('workers', 1); mode = plan['mode']
    advice = advise(entries) if plan.get('advise') else None
    before_cache = cache_state(entries); before_io = proc_io(); before_cpu = time.process_time()
    count = plan.get('requests', 8192); size = 4096 if mode == 'random_4k' else 12800
    jobs = partitions(entries, workers)
    if mode != 'sequential':
        rng = random.Random(20261024); requests = []
        eligible = [e for e in entries if e['length'] >= size]
        if not eligible:
            raise ValueError('Probe corpus is too small for range reads')
        for i in range(count):
            e = eligible[(i//5120) % len(eligible)] if mode == 'strided_12800' else rng.choice(eligible)
            slots = max(1, (e['length']-size)//(25600 if mode == 'strided_12800' else 4096)+1)
            offset = ((i % 5120) % slots)*25600 if mode == 'strided_12800' else rng.randrange(slots)*4096
            requests.append((e['path'], e['offset']+offset, size))
        jobs = [requests[r::workers] for r in range(workers)]
    started = time.perf_counter(); deadline = started+plan.get('seconds', 18.)
    def reader(items):
        read_bytes = operations = checksum = 0; latencies = []; expired = False
        if mode == 'sequential':
            buffer = bytearray(plan.get('block_bytes', 8*MIB))
            for e in items:
                with open(e['path'], 'rb', buffering=0) as stream:
                    stream.seek(e['offset']); remaining = e['length']
                    while remaining:
                        if time.perf_counter() >= deadline:
                            expired = True; break
                        view = memoryview(buffer)[:min(len(buffer), remaining)]
                        n = stream.readinto(view)
                        if not n:
                            raise IOError('Unexpected end of corpus')
                        checksum ^= buffer[0] ^ buffer[n-1]
                        read_bytes += n; operations += 1; remaining -= n
                if expired:
                    break
        else:
            descriptors = {}
            try:
                for path, offset, length in items:
                    if time.perf_counter() >= deadline:
                        expired = True; break
                    if path not in descriptors:
                        descriptors[path] = os.open(path, os.O_RDONLY | getattr(os, 'O_BINARY', 0))
                    begin = time.perf_counter_ns()
                    if hasattr(os, 'pread'):
                        data = os.pread(descriptors[path], length, offset)
                    else:
                        os.lseek(descriptors[path], offset, os.SEEK_SET)
                        data = os.read(descriptors[path], length)
                    latencies.append(time.perf_counter_ns()-begin)
                    if len(data) != length:
                        raise IOError('Unexpected short range read')
                    checksum ^= data[0] ^ data[-1]
                    read_bytes += len(data); operations += 1
            finally:
                for fd in descriptors.values():
                    os.close(fd)
        return dict(bytes=read_bytes, operations=operations, checksum=checksum,
                    expired=expired, latency_ns=latencies)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(reader, jobs))
    elapsed = time.perf_counter()-started; after_io = proc_io()
    read_bytes = sum(r['bytes'] for r in rows); latencies = sorted(t for r in rows for t in r['latency_ns'])
    def percentile(q):
        return latencies[min(len(latencies)-1, int((len(latencies)-1)*q))]/1e6 if latencies else None
    expected = sum(e['length'] for e in entries) if mode == 'sequential' else count*size
    return dict(label=plan.get('label'), mode=mode, workers=workers, bytes=read_bytes,
                expected_bytes=expected, complete=read_bytes == expected, seconds=elapsed,
                mib_per_second=read_bytes/MIB/elapsed, operations=sum(r['operations'] for r in rows),
                operations_per_second=sum(r['operations'] for r in rows)/elapsed,
                latency_ms=dict(p50=percentile(.5), p95=percentile(.95), p99=percentile(.99)),
                cpu_seconds=time.process_time()-before_cpu, advice=advice, cache_before=before_cache,
                cache_after=cache_state(entries), proc_io_delta={k:after_io[k]-v for k,v in before_io.items()},
                warning=WARNING)


def copy_corpus(entries, target):
    target = Path(target); target.mkdir(parents=True, exist_ok=False)
    result = []; start = time.perf_counter()
    for index, entry in enumerate(entries):
        path = target/f'part{index:03d}.bin'; digest = hashlib.sha256(); remaining = entry['length']
        with open(entry['path'], 'rb', buffering=0) as source, path.open('xb') as dest:
            source.seek(entry['offset'])
            while remaining:
                data = source.read(min(8*MIB, remaining))
                if not data:
                    raise IOError('Short source copy')
                digest.update(data); dest.write(data); remaining -= len(data)
            dest.flush(); os.fsync(dest.fileno())
        result.append(dict(path=str(path), offset=0, length=entry['length'], sha256=digest.hexdigest(),
                           source_path=entry['path'], source_offset=entry['offset']))
    return result, dict(seconds=time.perf_counter()-start, bytes=sum(e['length'] for e in result),
                        scope='Preparation includes source reads, SHA256, writes and fsync; not raw write bandwidth')


def verify_corpus(entries):
    for entry in entries:
        digest = hashlib.sha256()
        with open(entry['path'], 'rb') as stream:
            while data := stream.read(8*MIB):
                digest.update(data)
        if digest.hexdigest() != entry['sha256']:
            raise ValueError('Copied corpus hash mismatch')


def main(output, budget=340, total=2*GIB):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic(); deadline = started+budget
    result = dict(status='started', warning=WARNING, methodology='One allocation; one first pass and one immediate reread per path, plus DONTNEED-hinted sequential/concurrent/range probes; no physically-cold or repeated-session claim',
                  inventory=inventory(), paths=[], installed_packages=False, h3_inference=False, gpu=False)
    def save():
        result['seconds'] = time.monotonic()-started; write(output/'result.json', result)
    save()
    def run_phase(plan, identity):
        remaining = deadline-time.monotonic()
        if remaining < 8:
            return dict(label=plan['label'], skipped='Global benchmark time budget')
        plan['seconds'] = min(18., max(2., remaining-5))
        plan_file = output/(identity+'_plan.json'); path = output/(identity+'.json')
        write(plan_file, plan)
        try:
            proc = subprocess.run([sys.executable, str(Path(__file__).resolve()), '--child', str(plan_file),
                                   '--output', str(path)], capture_output=True, text=True,
                                  timeout=min(30., remaining))
            if proc.returncode:
                return dict(label=plan['label'], error=proc.stderr[-2000:], exit_code=proc.returncode)
            return json.loads(path.read_text())
        except subprocess.TimeoutExpired:
            return dict(label=plan['label'], timeout=True, timeout_seconds=min(30., remaining))
    def read_path(name, entries):
        row = dict(path=name, entries=entries, phases=[]); result['paths'].append(row); save()
        modes = [('first_pass', 'sequential', 1, False), ('immediate_reread', 'sequential', 1, False),
                 ('dontneed_hint', 'sequential', 1, True), ('two_readers_hint', 'sequential', 2, True),
                 ('random_4k_hint', 'random_4k', 1, True), ('strided_12800_hint', 'strided_12800', 1, True),
                 ('two_readers_strided_hint', 'strided_12800', 2, True)]
        for label, mode, workers, hint in modes:
            print('STORAGE_PHASE', name, label, flush=True)
            plan = dict(entries=entries, label=label, mode=mode, workers=workers, advise=hint, requests=8192)
            phase = run_phase(plan, f'p{len(result["paths"])}_{label}')
            row['phases'].append(phase); save()
    try:
        entries, selection = select_files('/kaggle/input', total=total)
        result['selection'] = selection; save()
        read_path('/kaggle/input', entries)
        for index, base in enumerate(('/kaggle/working', '/kaggle/temp', '/tmp')):
            remaining = deadline-time.monotonic()
            if remaining < 65:
                result['paths'].append(dict(path=base, skipped='Global benchmark time budget before copy')); save(); continue
            base = Path(base)
            if not base.exists():
                result['paths'].append(dict(path=str(base), skipped='Path absent')); save(); continue
            if shutil.disk_usage(base).free < selection['selected_bytes']+GIB:
                result['paths'].append(dict(path=str(base), skipped='Insufficient free space with 1-GiB margin')); save(); continue
            owned = base/f'h3-storage-probe-20261024-{os.getpid()}-{index}'
            if owned.exists():
                raise FileExistsError(owned)
            assert owned.resolve().is_relative_to(base.resolve())
            try:
                copied, preparation = copy_corpus(entries, owned)
                read_path(str(base), copied)
                result['paths'][-1]['copy_and_hash'] = preparation
                verify_corpus(copied); result['paths'][-1]['copied_bytes_verified'] = True; save()
            finally:
                if owned.exists():
                    assert owned.resolve().parent == base.resolve() and owned.name.startswith('h3-storage-probe-20261024-')
                    shutil.rmtree(owned)
        result['status'] = 'complete_with_limits' if any(p.get('skipped') or any(x.get('error') or x.get('timeout') or x.get('skipped') or not x.get('complete', False) for x in p.get('phases', [])) for p in result['paths']) else 'complete'
    except BaseException as error:
        result.update(status='failed', error=repr(error)); raise
    finally:
        save(); print('STORAGE_FINAL', result['status'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--child', type=Path); parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--budget', type=int, default=340)
    args = parser.parse_args()
    if args.child:
        write(args.output, measure(json.loads(args.child.read_text())))
    else:
        main(args.output, args.budget)
