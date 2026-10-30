"""Benchmark-only aligned O_DIRECT reader. No production default or OS policy change."""
import ctypes
import io
import json
import math
import mmap
import os
from pathlib import Path
import struct
import threading
import time
from types import SimpleNamespace

MIB = 2**20


def alignment_info(path):
    result = dict(statx_available=False, memory_alignment=None, offset_alignment=None)
    if os.name != 'posix':
        return result
    libc = ctypes.CDLL(None, use_errno=True)
    if not hasattr(libc, 'statx'):
        return result
    buffer = ctypes.create_string_buffer(256)
    function = libc.statx
    function.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_uint, ctypes.c_void_p]
    function.restype = ctypes.c_int
    rc = function(-100, os.fsencode(path), 0x4000, 0x2000, buffer)
    if rc:
        return dict(result, errno=ctypes.get_errno())
    mask, = struct.unpack_from('I', buffer, 0)
    mem, off = struct.unpack_from('II', buffer, 152)
    return dict(statx_available=True, mask=mask, memory_alignment=mem,
                offset_alignment=off, dio_alignment_reported=bool(mask & 0x2000 and mem and off))


class DirectFile(io.RawIOBase):
    """All payload syscalls use O_DIRECT + aligned preadv; never fall back.

    Rounded read sizes/offsets may reread small boundary sectors. EOF is handled
    without padding the source file. The anonymous bounce buffer is bounded.
    """
    def __init__(self, path, *, chunk_bytes=8*MIB, observer=None, opener=None, reader=None,
                 alignment=None):
        super().__init__()
        if opener is None and (not hasattr(os, 'O_DIRECT') or not hasattr(os, 'preadv')):
            raise NotImplementedError('Linux O_DIRECT and preadv are required')
        self.name = str(path)
        self.mode = 'rb'
        info = alignment_info(path) if alignment is None else {}
        self.alignment = alignment or math.lcm(mmap.PAGESIZE, info.get('memory_alignment') or 1,
                                              info.get('offset_alignment') or 1)
        self.chunk = max(self.alignment, chunk_bytes // self.alignment * self.alignment)
        self.fd = (opener or os.open)(path, os.O_RDONLY | getattr(os, 'O_DIRECT', 0))
        self.size = os.fstat(self.fd).st_size
        self.position = 0
        self._reader = reader or os.preadv
        self._observer = observer
        self._mapping = mmap.mmap(-1, self.chunk + self.alignment)
        address = ctypes.addressof(ctypes.c_char.from_buffer(self._mapping))
        shift = (-address) % self.alignment
        self._buffer = memoryview(self._mapping)[shift:shift+self.chunk]
        self.lock = threading.Lock()

    def fileno(self):
        self._checkClosed()
        return self.fd

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=os.SEEK_SET):
        value = offset if whence == os.SEEK_SET else self.position + offset if whence == os.SEEK_CUR else self.size + offset
        if value < 0 or whence not in (os.SEEK_SET, os.SEEK_CUR, os.SEEK_END):
            raise ValueError('Invalid seek')
        self.position = value
        return value

    def preadinto(self, destination, offset):
        self._checkClosed()
        if offset < 0:
            raise ValueError('Negative offset')
        view = memoryview(destination).cast('B')
        wanted = min(len(view), max(0, self.size-offset))
        done = 0
        with self.lock:
            while done < wanted:
                absolute = offset + done
                aligned = absolute // self.alignment * self.alignment
                lead = absolute - aligned
                useful = min(wanted-done, self.chunk-lead)
                requested = math.ceil((lead+useful)/self.alignment)*self.alignment
                start = time.monotonic_ns()
                cpu = time.thread_time_ns()
                returned = self._reader(self.fd, [self._buffer[:requested]], aligned)
                end = time.monotonic_ns()
                if self._observer:
                    self._observer(dict(kind='direct_syscall', file=self.name, offset=aligned,
                        requested_bytes=requested, returned_bytes=returned, logical_bytes=useful,
                        start_ns=start, end_ns=end, thread_cpu_ns=time.thread_time_ns()-cpu))
                if returned <= lead:
                    raise IOError('Incomplete direct range')
                count = min(useful, returned-lead)
                view[done:done+count] = self._buffer[lead:lead+count]
                done += count
        return done

    def pread(self, size, offset):
        result = bytearray(min(size, max(0, self.size-offset)))
        count = self.preadinto(result, offset)
        return bytes(result[:count])

    def readinto(self, destination):
        count = self.preadinto(destination, self.position)
        self.position += count
        return count

    def read(self, size=-1):
        if size < 0:
            raise ValueError('Whole-file reads are forbidden in this bounded probe')
        result = self.pread(size, self.position)
        self.position += len(result)
        return result

    def close(self):
        if not self.closed:
            if hasattr(self, 'fd'):
                os.close(self.fd)
            if hasattr(self, '_buffer'):
                self._buffer.release()
            if hasattr(self, '_mapping'):
                self._mapping.close()
        super().close()


class IOAudit:
    def __init__(self, output):
        self.stream = Path(output).open('w', encoding='utf-8')
        self.phase = 'startup'
        self.lock = threading.Lock()
        self.rows = 0
        self.dropped = 0
        self.local = threading.local()
        self.direct_files = {}

    def emit(self, row):
        row = dict(row, phase=self.phase, category=getattr(self.local, 'category', 'weights'),
                   layer=getattr(self.local, 'layer', None),
                   tid=threading.get_native_id(), pid=os.getpid())
        with self.lock:
            if self.rows < 150000:
                self.stream.write(json.dumps(row, separators=(',', ':'))+'\n')
                self.rows += 1
            else:
                self.dropped += 1

    def span(self, kind, function, *args, **kwargs):
        start, cpu = time.monotonic_ns(), time.thread_time_ns()
        try:
            return function(*args, **kwargs)
        finally:
            self.emit(dict(kind=kind, start_ns=start, end_ns=time.monotonic_ns(),
                           thread_cpu_ns=time.thread_time_ns()-cpu))

    def close(self):
        self.stream.close()


class BufferedFile:
    def __init__(self, file, audit):
        self.file, self.audit = file, audit

    def __getattr__(self, name):
        return getattr(self.file, name)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return self.file.__exit__(*args)

    def transfer(self, method, argument):
        offset = self.file.tell()
        start, cpu = time.monotonic_ns(), time.thread_time_ns()
        result = getattr(self.file, method)(argument)
        self.audit.emit(dict(kind='buffered_'+method, file=str(self.file.name), offset=offset,
            returned_bytes=len(result) if isinstance(result, bytes) else result,
            start_ns=start, end_ns=time.monotonic_ns(), thread_cpu_ns=time.thread_time_ns()-cpu))
        return result

    def read(self, size=-1):
        return self.transfer('read', size)

    def readinto(self, destination):
        return self.transfer('readinto', destination)


def install(root, mode, audit):
    """Intercept only this experiment's immutable TE safetensors and TE proofs."""
    from patches import windowed_tp as wt, tp2_bundle as bundle
    root = Path(root).resolve()
    original_open = Path.open
    original_pread = getattr(os, 'pread', None)

    def selected(path):
        return Path(path).suffix == '.safetensors' and Path(path).resolve().is_relative_to(root)

    def opened(path, *args, **kwargs):
        readmode = args[0] if args else kwargs.get('mode', 'r')
        if readmode != 'rb' or not selected(path):
            return original_open(path, *args, **kwargs)
        if mode == 'buffered':
            return BufferedFile(original_open(path, *args, **kwargs), audit)
        file = DirectFile(path, observer=audit.emit)
        audit.direct_files[file.fd] = file
        return file

    Path.open = opened
    # CheckpointReader's bounded os.pread must use the same aligned adapter;
    # patch its module-local OS reference, not the process-global os.pread.
    proxy = SimpleNamespace(**vars(os))
    def pread(fd, size, offset):
        file = audit.direct_files.get(fd)
        if mode == 'direct' and file is not None and not file.closed:
            return file.pread(size, offset)
        start, cpu = time.monotonic_ns(), time.thread_time_ns()
        result = original_pread(fd, size, offset)
        audit.emit(dict(kind='buffered_pread', requested_bytes=size, returned_bytes=len(result),
            offset=offset, start_ns=start, end_ns=time.monotonic_ns(),
            thread_cpu_ns=time.thread_time_ns()-cpu))
        return result
    proxy.pread = pread
    wt.os = proxy

    for name, label in [('sha', 'file_sha256'), ('state_hashes', 'tensor_sha256')]:
        function = getattr(bundle, name)
        def proof(*args, _function=function, _label=label, **kwargs):
            previous = getattr(audit.local, 'category', 'weights')
            audit.local.category = _label
            start, cpu = time.monotonic_ns(), time.thread_time_ns()
            try:
                return _function(*args, **kwargs)
            finally:
                row = dict(kind=_label, start_ns=start, end_ns=time.monotonic_ns(),
                           thread_cpu_ns=time.thread_time_ns()-cpu)
                if _label == 'file_sha256':
                    row.update(file=str(args[0]), bytes=Path(args[0]).stat().st_size)
                else:
                    row['bytes'] = sum(t.numel()*t.element_size() for t in args[0].state_dict().values())
                audit.emit(row)
                audit.local.category = previous
        setattr(bundle, name, proof)
    original_tensor = wt.CheckpointReader.read_tensor
    def tensor(self, name, **kwargs):
        return audit.span('read_tensor', original_tensor, self, name, **kwargs)
    wt.CheckpointReader.read_tensor = tensor
