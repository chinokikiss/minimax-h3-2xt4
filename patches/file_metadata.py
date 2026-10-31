"""Cheap inference-time admission and mutation guards; no payload reads.

These signatures detect ordinary replacement, truncation and modification.
They are deliberately not content proofs. Offline tools retain SHA256 checks.
"""
import json
from os.path import abspath
from pathlib import Path
import stat as stat_module
import threading


def file_metadata(path):
    stat = Path(path).stat()
    if not stat_module.S_ISREG(stat.st_mode):
        raise ValueError(f'Expected a regular file: {path}')
    return (stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns,
            stat.st_dev, stat.st_ino)


class FileChecks:
    """Admit each local file once; reject changes for the reader's lifetime."""
    def __init__(self):
        self.files = {}
        self.lock = threading.RLock()

    def admit(self, path, *, size=None):
        # Keep the admitted pathname stable if it is later replaced by a symlink.
        path = Path(abspath(path))
        current = file_metadata(path)
        with self.lock:
            previous = self.files.get(path)
            if previous is not None and previous != current:
                raise RuntimeError(f'File changed after admission: {path}')
            if size is not None and current[0] != size:
                raise ValueError(f'File size mismatch: {path}')
            self.files[path] = current
        return current

    def read_json(self, path):
        self.admit(path)
        value = json.loads(Path(path).read_text(encoding='utf-8'))
        self.admit(path)
        return value

    def check(self):
        with self.lock:
            paths = tuple(self.files)
        for path in paths:
            self.admit(path)


# Source/library admission is process-local, just like the loaded Python/DSO.
runtime_files = FileChecks()


def admit_library(path, sources):
    """Keep binary/receipt/source identities stable without rereading contents."""
    path = Path(path).resolve()
    runtime_files.admit(path)
    receipt = runtime_files.read_json(path.with_name('build_receipt.json'))
    if path.stat().st_size == 0:
        raise ValueError('Empty native operator library')
    if 'artifact_bytes' in receipt:
        runtime_files.admit(path, size=receipt['artifact_bytes'])
    for source in sources:
        runtime_files.admit(source)
    return receipt
