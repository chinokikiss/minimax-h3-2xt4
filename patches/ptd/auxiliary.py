"""Compact PTD auxiliary units, bound to a bake receipt and verified on use.

The caller admits the bank manifest as its trust root. This module never opens
an unrequested unit and never retains tensor payloads outside the host cache.
"""
import hashlib
import json
import math
from pathlib import Path
import re
import struct
import threading
import os

import torch
from safetensors.torch import save_file

from patches.tp2_bundle import signature
from patches.tp2_disk_bake import state_proof
from patches.windowed_tp import CheckpointReader

FORMAT = 'h3-ptd-aux-split-v1'
LAYOUT = 'native-merged-compact-safetensors-v1'
HEADS = {f'final_layer.{kind}_out.{field}' for kind in ('video', 'audio') for field in ('weight', 'bias')}
REFINER = re.compile(r'token_refiner\.blocks\.\d+\.(?:attn\.(?:qkv_proj|out_proj)|mlp\.fc[12])\.weight\Z')


def tensor_hash(value):
    return hashlib.sha256(memoryview(value.reshape(-1).view(torch.uint8).numpy()).cast('B')).hexdigest()


def _digest(value):
    return isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None


def _path(root, relative):
    path = Path(relative)
    if path.is_absolute() or '..' in path.parts or '\\' in relative:
        raise ValueError('PTD auxiliary unit escaped its hop')
    resolved = (root / path).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError('PTD auxiliary unit escaped its hop')
    return resolved


def write_split(values, output, recipe_sha256, hop, verified_targets):
    """Persist Native-verified tensors without a cast, fold or re-quantization."""
    if not _digest(recipe_sha256) or type(hop) is not int or not 0 <= hop < 8:
        raise ValueError('Invalid split auxiliary provenance')
    expected = {name + '.weight' for name in verified_targets} | HEADS
    if set(values) != expected or any(not REFINER.fullmatch(n) for n in expected - HEADS):
        raise ValueError('Incomplete / unknown split auxiliary coverage')
    root = Path(output).resolve()
    groups = [(f'refiner/{name.removesuffix(".weight").replace(".", "_")}.safetensors', 'refiner', [name])
              for name in sorted(expected - HEADS)]
    groups += [(f'heads/{kind}.safetensors', kind + '_head',
                sorted(n for n in HEADS if n.startswith(f'final_layer.{kind}_out.')))
               for kind in ('video', 'audio')]
    units, states = [], {}
    for relative, role, names in groups:
        path = _path(root, relative)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {n: values[n].detach().cpu().contiguous() for n in names}
        if any(v.dtype not in (torch.float16, torch.bfloat16, torch.float32) for v in payload.values()):
            raise ValueError('PTD auxiliary units require Native floating weights')
        partial = path.with_suffix('.partial')
        save_file(payload, str(partial))
        with partial.open('r+b') as stream:
            os.fsync(stream.fileno())
            size = struct.unpack('<Q', stream.read(8))[0]
            header = stream.read(size)
        partial.replace(path)
        units.append(dict(file=relative, role=role, bytes=path.stat().st_size,
            header_bytes=8 + size, header_sha256=hashlib.sha256(struct.pack('<Q', size) + header).hexdigest(),
            header=json.loads(header), tensors={n: tensor_hash(v) for n, v in payload.items()}))
        states.update(state_proof(payload))
        del payload  # Repack must release this unit before reading the next.
    result = dict(format=FORMAT, layout=LAYOUT, recipe_sha256=recipe_sha256, hop=hop,
        units=units, bytes=sum(u['bytes'] for u in units), state_sha256=states,
        verified_targets=sorted(verified_targets))
    SplitAuxiliary(root, result, recipe_sha256, hop)  # Metadata admission before publication.
    return result


class _UnitReader(CheckpointReader):
    def __init__(self, path, proof, current):
        # Allocate only from receipt-bound metadata, never an unchecked file header.
        self.path, self.header, self.offset = path, proof['header'], proof['header_bytes']
        self.signature, self.identity = current, (str(path), *current)
        self.read_lock = threading.Lock()
        self.logical_read_bytes = self.range_read_calls = self.source_read_bytes = 0


class SplitAuxiliary:
    def __init__(self, root, proof, recipe_sha256, hop):
        self.root = Path(root).resolve()
        self.hop = hop
        if (proof.get('format') != FORMAT or proof.get('layout') != LAYOUT
                or proof.get('recipe_sha256') != recipe_sha256 or proof.get('hop') != hop):
            raise ValueError('Split auxiliary layout / bake / hop mismatch')
        self.by_name = {}
        files = set()
        for unit in proof['units']:
            # Paths are resolved on consumption, so metadata admission does not
            # stat/open the other rank's refiner files or follow their symlinks.
            relative = unit['file']
            if (not isinstance(relative, str) or Path(relative).is_absolute() or '..' in Path(relative).parts
                    or '\\' in relative or relative in files):
                raise ValueError('Invalid / duplicate PTD auxiliary path')
            files.add(relative)
            header, tensors, role = unit['header'], unit['tensors'], unit['role']
            if set(header) != set(tensors) or not tensors or not _digest(unit['header_sha256']):
                raise ValueError('Invalid PTD unit header proof')
            if (type(unit['header_bytes']) is not int or not 8 < unit['header_bytes'] <= 65536
                    or type(unit['bytes']) is not int):
                raise ValueError('Invalid PTD unit byte bounds')
            end = 0
            for name, info in sorted(header.items(), key=lambda item: item[1]['data_offsets'][0]):
                shape, dtype, offsets = info['shape'], info['dtype'], info['data_offsets']
                if (dtype not in ('F32', 'F16', 'BF16') or not isinstance(shape, list)
                        or not 1 <= len(shape) <= 2 or any(type(d) is not int or d <= 0 for d in shape)
                        or len(offsets) != 2 or any(type(d) is not int for d in offsets)
                        or offsets[0] != end or offsets[1] - end != math.prod(shape) * (4 if dtype == 'F32' else 2)
                        or not _digest(tensors[name]) or name in self.by_name):
                    raise ValueError('Invalid PTD tensor layout / payload proof')
                if role == 'refiner':
                    valid = len(tensors) == 1 and REFINER.fullmatch(name)
                else:
                    kind = role.removesuffix('_head')
                    valid = role in ('video_head', 'audio_head') and set(tensors) == {
                        f'final_layer.{kind}_out.{field}' for field in ('weight', 'bias')}
                if not valid:
                    raise ValueError('PTD auxiliary role / tensor mismatch')
                self.by_name[name] = unit
                end = offsets[1]
            if end + unit['header_bytes'] != unit['bytes']:
                raise ValueError('PTD auxiliary payload size mismatch')
        expected = {name + '.weight' for name in proof['verified_targets']} | HEADS
        if (set(self.by_name) != expected or set(proof['state_sha256']) != expected
                or proof['bytes'] != sum(unit['bytes'] for unit in proof['units'])):
            raise ValueError('Incomplete PTD auxiliary manifest')
        for name, unit in self.by_name.items():
            info = unit['header'][name]
            dtype = str(CheckpointReader.DTYPES[info['dtype']])
            if proof['state_sha256'][name] != dict(dtype=dtype, shape=info['shape'], sha256=unit['tensors'][name]):
                raise ValueError('PTD auxiliary differs from the Native tensor proof')
        self.lock = threading.Lock()
        self.stats = {role: dict(header_bytes=0, payload_bytes=0, verified_tensors=0)
                      for role in ('refiner', 'video_head', 'audio_head')}

    def read_tensor(self, name):
        """Verify every newly read payload, including reloads after eviction."""
        proof = self.by_name[name]
        with self.lock:
            path = _path(self.root, proof['file'])
            before = signature(path)
            if before[0] != proof['bytes']:
                raise ValueError('PTD auxiliary unit size mismatch')
            with path.open('rb', buffering=0) as stream:
                header = stream.read(proof['header_bytes'])
            if (hashlib.sha256(header).hexdigest() != proof['header_sha256']
                    or struct.unpack('<Q', header[:8])[0] != len(header) - 8
                    or json.loads(header[8:]) != proof['header']):
                raise ValueError('PTD auxiliary header checksum / layout mismatch')
            reader = _UnitReader(path, proof, before)
            value = reader.read_tensor(name)
            if tensor_hash(value) != proof['tensors'][name]:
                raise ValueError('PTD auxiliary tensor checksum mismatch: ' + name)
            if signature(path) != before or _path(self.root, proof['file']) != path:
                raise RuntimeError('PTD auxiliary changed during consumption')
            row = self.stats[proof['role']]
            row['header_bytes'] += len(header)
            row['payload_bytes'] += value.numel() * value.element_size()
            row['verified_tensors'] += 1
            return value
