"""Experimental pinned Kaggle HTTP Range x2 -> final CPU tensor storage.

No whole-file staging buffer, disk copy, or runtime payload SHA256. The caller
hands the resulting tensors to native rank-module construction/HostWeightCache.
Mounted immutable Dataset files remain the header/inventory authority.
"""
from concurrent.futures import ThreadPoolExecutor
import re
import threading
import time
from urllib.parse import urlsplit

import requests
import torch

from patches.windowed_tp import CheckpointReader

_PINNED = re.compile(r'^([a-zA-Z0-9_-]+)/([a-zA-Z0-9_-]+)/versions/([1-9][0-9]*)$')
_CHUNK = 4 * 2**20


def pinned_raw_url(identity, filename):
    """Resolve an immutable Dataset-version raw-object URL, without logging it."""
    match = _PINNED.fullmatch(identity)
    if match is None or not filename.startswith('payload/') or '..' in filename.split('/'):
        raise ValueError('Expected a pinned Kaggle PTD payload identity and relative file')
    from kagglehub.clients import build_kaggle_client
    from kagglesdk.datasets.types.dataset_api_service import ApiDownloadDatasetRequest
    req = ApiDownloadDatasetRequest()
    req.owner_slug, req.dataset_slug = match.group(1), match.group(2)
    req.dataset_version_number = int(match.group(3))
    req.file_name, req.raw = filename, True
    response = build_kaggle_client().datasets.dataset_api_client.download_dataset(req)
    try:
        url = response.url
    finally:
        response.close()
    if urlsplit(url).scheme != 'https':
        raise RuntimeError('Pinned Dataset raw URL must be HTTPS')
    return url


class HTTPRange2TensorReader(CheckpointReader):
    """Two concurrent byte ranges scatter directly into final CPU tensors.

    Only small headers are read from the mounted object. Two HTTP GETs cover
    the requested tensor span; any holes in a sparse selection are discarded
    through a bounded 4 MiB scratch buffer, never a whole payload buffer.
    """

    def __init__(self, path, *, dataset_identity, remote_filename,
                 url_supplier=None, on_transfer=None):
        super().__init__(path)
        if _PINNED.fullmatch(dataset_identity) is None:
            raise ValueError('HTTP2 must use an immutable version-pinned Dataset')
        self.dataset_identity = dataset_identity
        self.remote_filename = remote_filename
        self.url_supplier = url_supplier or pinned_raw_url
        self.on_transfer = on_transfer
        self._url = None
        self._url_lock = threading.Lock()

    def _resolved_url(self):
        with self._url_lock:
            if self._url is None:
                self._url = self.url_supplier(self.dataset_identity, self.remote_filename)
                if urlsplit(self._url).scheme != 'https' and self.url_supplier is pinned_raw_url:
                    raise RuntimeError('Pinned Dataset raw URL must use HTTPS')
            return self._url

    @staticmethod
    def _fill(response, target, wanted):
        """Copy exactly wanted bytes into a direct destination or discard area."""
        remaining = wanted
        advance = len(target) >= wanted
        while remaining:
            view = target[:min(len(target), remaining, _CHUNK)]
            n = response.raw.readinto(view)
            if n is None or n <= 0:
                raise IOError('Incomplete HTTP PTD Range response')
            remaining -= n
            if advance:
                target = target[n:]
        return wanted

    @classmethod
    def _receive(cls, url, start, end, spans, total_file_bytes):
        expected = f'bytes {start}-{end-1}/{total_file_bytes}'
        headers = {'Range': f'bytes={start}-{end-1}',
                   'Accept-Encoding': 'identity', 'Cache-Control': 'no-transform'}
        try:
            with requests.get(url, headers=headers, stream=True, timeout=(15, 90)) as response:
                if (response.status_code != 206
                        or response.headers.get('Content-Range') != expected
                        or response.headers.get('Content-Encoding', 'identity').lower() != 'identity'):
                    raise IOError('Pinned HTTP Range server violated exact 206 range contract')
                if ('Content-Length' in response.headers
                        and int(response.headers['Content-Length']) != end-start):
                    raise IOError('Pinned HTTP Range response length differs from request')
                response.raw.decode_content = False
                cursor = start
                scratch = memoryview(bytearray(_CHUNK))
                for lo, hi, destination in spans:
                    if hi <= cursor or lo >= end:
                        continue
                    lo, hi = max(lo, start), min(hi, end)
                    if lo > cursor:
                        cls._fill(response, scratch, lo-cursor)
                    cursor = hi
                    offset = lo - next(a for a,b,v in spans if v is destination and a <= lo < b)
                    cls._fill(response, destination[offset:offset+(hi-lo)], hi-lo)
                if cursor < end:
                    cls._fill(response, scratch, end-cursor)
                # A complying Range response cannot contain trailing payload.
                if response.raw.read(1):
                    raise IOError('Pinned HTTP Range response contained excess bytes')
        except requests.RequestException:
            # Do not leak signed GCS URL in the error log/exception message.
            raise RuntimeError('Pinned PTD HTTP Range transfer failed') from None
        return end-start

    def read_tensors(self, names):
        self.check()
        if (self.path.stat().st_size, self.path.stat().st_mtime_ns) != self.signature:
            raise RuntimeError('PTD shard changed after HTTP Range planning')
        names = tuple(names)
        if not names or len(names) != len(set(names)):
            raise ValueError('HTTP2 requires a nonempty list of distinct tensor names')
        result, spans = {}, []
        for name in names:
            info = self.header[name]
            shape = info['shape']
            value = torch.empty(shape, dtype=self.DTYPES[info['dtype']], device='cpu')
            dest = memoryview(value.reshape(-1).view(torch.uint8).numpy()).cast('B')
            lo, hi = (self.offset+offset for offset in info['data_offsets'])
            if hi > lo:
                spans.append((lo, hi, dest))
            result[name] = value
        spans.sort(key=lambda v: v[0])
        if not spans:
            return result
        for first, second in zip(spans, spans[1:]):
            if first[1] > second[0]:
                raise ValueError('Overlapping HTTP2 tensor destinations')
        start, end = spans[0][0], spans[-1][1]
        midpoint = start + (end-start)//2
        if midpoint == start:
            ranges = [(start, end)]
        else:
            ranges = [(start, midpoint), (midpoint, end)]
        started = time.perf_counter()
        url = self._resolved_url()
        transport_started = time.perf_counter()
        try:
            with ThreadPoolExecutor(max_workers=2, thread_name_prefix='ptd-http-range') as pool:
                futures = [pool.submit(self._receive, url, a, b, spans, self.signature[0])
                           for a,b in ranges]
                transferred = sum(future.result() for future in futures)
        except BaseException:
            # No partially populated tensor may escape into HostWeightCache.
            raise
        transferred_seconds = time.perf_counter()-transport_started
        ready_seconds = time.perf_counter()-started
        self.check()
        logical = sum(value.numel()*value.element_size() for value in result.values())
        with self.read_lock:
            self.logical_read_bytes += logical
            self.source_read_bytes += transferred
            self.range_read_calls += len(ranges)
        if self.on_transfer is not None:
            self.on_transfer(logical, transferred, len(ranges), transferred_seconds, ready_seconds)
        return result

    def read_tensor(self, name, *, rows=None, columns=None):
        if rows is not None or columns is not None:
            raise ValueError('HTTP2 direct host tensors require whole tensor reads')
        return self.read_tensors((name,))[name]
