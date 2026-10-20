"""Assemble a locally fingerprinted block fixture from exact original ranges."""
import base64
import hashlib
import json
from pathlib import Path
import struct


def read_header(path):
    with Path(path).open('rb') as stream:
        prefix = stream.read(8)
        size = struct.unpack('<Q', prefix)[0]
        header = stream.read(size)
    return json.loads(header), prefix + header


def make_plan(original, fixture, remote_file):
    source, source_header = read_header(original)
    selected, header = read_header(fixture)
    ranges = []
    for name, tensor in selected.items():
        if name == '__metadata__':
            continue
        old = source[name]
        if old['dtype'] != tensor['dtype'] or old['shape'] != tensor['shape']:
            raise ValueError('Tensor identity differs')
        start, end = old['data_offsets']
        dst_start, dst_end = tensor['data_offsets']
        if end - start != dst_end - dst_start:
            raise ValueError('Tensor size differs')
        ranges.append({'key': name, 'start': len(source_header) + start,
                       'end': len(source_header) + end, 'target': len(header) + dst_start})
    with Path(fixture).open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    return {'remote_file': remote_file, 'source_bytes': Path(original).stat().st_size,
            'bytes': Path(fixture).stat().st_size, 'sha256': digest,
            'header_base64': base64.b64encode(header).decode(), 'ranges': ranges}


def assemble(plan, output, read_range):
    # Coalesce nearby tensors, then bound individual responses to 64MiB.
    grouped = []
    for tensor in sorted(plan['ranges'], key=lambda t: t['start']):
        if grouped and tensor['start'] <= grouped[-1][1] + 65536:
            grouped[-1][1] = max(grouped[-1][1], tensor['end'])
        else:
            grouped.append([tensor['start'], tensor['end']])
    output = Path(output)
    with output.open('wb') as stream:
        stream.write(base64.b64decode(plan['header_base64']))
        stream.truncate(plan['bytes'])
        for start, end in grouped:
            for a in range(start, end, 64 * 1024 ** 2):
                b = min(end, a + 64 * 1024 ** 2)
                payload = read_range(a, b)
                if len(payload) != b - a:
                    raise RuntimeError('Range response has the wrong length')
                for tensor in plan['ranges']:
                    lo, hi = max(a, tensor['start']), min(b, tensor['end'])
                    if hi > lo:
                        stream.seek(tensor['target'] + lo - tensor['start'])
                        stream.write(memoryview(payload)[lo - a:hi - a])
    with output.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    if digest != plan['sha256']:
        raise RuntimeError('Assembled trained block differs from locally validated bytes')


def download(plan, output):
    import requests
    url = ('https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/'
           'e5eb578a89295337b8ff433a035929ce0279e0b6/' + plan['remote_file'])
    def read_range(a, b):
        with requests.get(url, params={'download': 'true', 'offset': str(a)},
                          headers={'Range': f'bytes={a}-{b - 1}'}, stream=True, timeout=(10, 40)) as response:
            if response.status_code != 206 or response.headers.get('Content-Range') != f'bytes {a}-{b - 1}/{plan["source_bytes"]}':
                raise RuntimeError('Server did not provide the requested exact checkpoint range')
            result = bytearray()
            for part in response.iter_content(4 * 1024 ** 2):
                result.extend(part)
                if len(result) > b - a:
                    raise RuntimeError('Oversized range response')
            return result
    assemble(plan, output, read_range)
