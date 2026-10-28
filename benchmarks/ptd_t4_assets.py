"""Fetch only real trained PTD tensors used by the T4 qualification fixture.

Never imported by the local builder; all LoRA byte requests require two T4s.
The subset is bound to immutable source ranges, not advertised as the full
11 GB release's SHA256. Full deployment uses checkpoint.download_on_t4.
"""
import hashlib
import json
from pathlib import Path
import struct


def fetch_subset(directory, layers=2):
    import requests
    import torch
    from patches.ptd.checkpoint import RELEASE
    from tp2_public_dit_assets import recover
    if (torch.cuda.device_count() != 2 or any(torch.cuda.get_device_capability(i) != (7, 5)
            or 'T4' not in torch.cuda.get_device_name(i) for i in (0, 1))):
        raise RuntimeError('PTD weight downloads are allowed only on physical two-T4 hardware')
    if layers != 2:
        raise ValueError('This reduced qualification fixture covers exactly two trained blocks')
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=False)
    url = f'https://huggingface.co/{RELEASE["repo"]}/resolve/{RELEASE["revision"]}/{RELEASE["file"]}'
    ranges = []

    def read(start, stop):
        def once(attempt):
            with requests.get(url, params={'download': 'true', 'offset': str(start)},
                    headers={'Range': f'bytes={start}-{stop-1}'}, stream=True, timeout=(15, 90)) as response:
                response.raise_for_status()
                expected = f'bytes {start}-{stop-1}/{RELEASE["bytes"]}'
                if response.status_code != 206 or response.headers.get('Content-Range') != expected:
                    raise ValueError('PTD source range identity mismatch')
                data = bytearray()
                for chunk in response.iter_content(4 * 2**20):
                    data.extend(chunk)
                    if len(data) > stop-start:
                        raise ValueError('Oversized PTD source response')
                if len(data) != stop-start:
                    raise ConnectionError('Truncated PTD source response')
                return data
        data = recover(once, label=f'ptd:{start}:{stop}')
        ranges.append(dict(start=start, stop=stop, sha256=hashlib.sha256(data).hexdigest()))
        return data

    length = struct.unpack('<Q', read(0, 8))[0]
    if not 2 <= length <= 16 * 2**20:
        raise ValueError('Invalid PTD safetensors header size')
    header_bytes = read(8, 8+length)
    original = json.loads(header_bytes)
    metadata = dict(original.get('__metadata__', {}), h3_qualification_only='two-block trained subset')
    selected = {key: value for key, value in original.items() if key != '__metadata__'
        and (not key.startswith('transformer_blocks.') or int(key.split('.')[1]) < layers)}
    header, rows, offset = {'__metadata__': metadata}, [], 0
    for key, info in sorted(selected.items(), key=lambda item: item[1]['data_offsets'][0]):
        start, stop = info['data_offsets']; size = stop-start
        if size <= 0:
            raise ValueError('Empty PTD source tensor')
        header[key] = dict(info, data_offsets=[offset, offset+size])
        rows.append((8+length+start, 8+length+stop, offset))
        offset += size
    raw = json.dumps(header, separators=(',', ':')).encode('utf-8')
    raw += b' ' * (-len(raw) % 8)
    path = directory/'ptd_two_block.safetensors'; partial = path.with_suffix('.partial')
    groups = []
    for row in rows:
        if groups and row[0] == groups[-1][-1][1] and row[1]-groups[-1][0][0] <= 64 * 2**20:
            groups[-1].append(row)
        else:
            groups.append([row])
    with partial.open('wb') as stream:
        stream.write(struct.pack('<Q', len(raw))); stream.write(raw)
        for index, group in enumerate(groups):
            data = read(group[0][0], group[-1][1])
            for start, stop, destination in group:
                stream.seek(8+len(raw)+destination)
                stream.write(memoryview(data)[start-group[0][0]:stop-group[0][0]])
            print('PTD_TRAINED_RANGE', index+1, '/', len(groups), flush=True)
    partial.replace(path)
    from patches.tp2_bundle import sha
    provenance = dict(release=RELEASE, source_url=url, source_header_sha256=hashlib.sha256(header_bytes).hexdigest(),
        ranges=ranges, selected_tensors=len(selected), full_release_sha256_verified=False,
        subset_sha256=sha(path), subset_bytes=path.stat().st_size, deployment_allowed=False)
    (directory/'provenance.json').write_text(json.dumps(provenance, indent=2), encoding='utf-8')
    return path
