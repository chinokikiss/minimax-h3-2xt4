"""Strict metadata admission and one-hop range reads; never load the 8-hop bank."""
import json
import math
from pathlib import Path

import torch
from safetensors import safe_open

from patches.windowed_tp import CheckpointReader
from .mapping import expected_adapter_keys, expected_head_keys, map_adapter_key

RELEASE = json.loads(Path(__file__).with_name('reference_sources.json').read_text(encoding='utf-8'))['release']


class PTDCheckpoint:
    def __init__(self, path, base=None):
        self.path = Path(path).resolve()
        self.reader = CheckpointReader(self.path)
        self.header = self.reader.header
        self.signature = self.reader.signature
        meta = self.header.get('__metadata__', {})
        self.hops = int(meta.get('ptd_num_hops', 0))
        self.steps = int(meta.get('pdd_num_steps', 32))
        self.rank = int(meta.get('lora_rank', 128))
        self.alpha = float(meta.get('lora_alpha', self.rank))
        if self.hops != 8 or self.steps != 32 or self.rank < 1 or not math.isfinite(self.alpha):
            raise ValueError('This adapter requires the released 8-hop / 32-interval PTD format')
        keys = set(self.header) - {'__metadata__'}
        self.targets = {}
        for key in sorted(keys - expected_head_keys()):
            target, factor = map_adapter_key(key)
            self.targets.setdefault(target.module, {}).setdefault(key.rsplit('.lora_', 1)[0], {})[factor] = key
        blocks = {int(t.split('.')[1]) for t in self.targets if t.startswith('blocks.')}
        refiners = {int(t.split('.')[2]) for t in self.targets if t.startswith('token_refiner.blocks.')}
        self.blocks, self.refiners = len(blocks), len(refiners)
        if blocks != set(range(self.blocks)) or refiners != set(range(self.refiners)):
            raise ValueError('PTD layer indices must be contiguous')
        if not blocks or keys != expected_adapter_keys(self.blocks, self.refiners) | expected_head_keys():
            raise ValueError('Missing or unconsumed PTD factors / PDD heads')
        for target, modules in self.targets.items():
            for raw, pair in modules.items():
                if set(pair) != {'down', 'up'}:
                    raise ValueError('Incomplete PTD factor pair: ' + raw)
                a, b = self.header[pair['down']], self.header[pair['up']]
                if (len(a['shape']) != 3 or len(b['shape']) != 3
                        or a['shape'][:2] != [self.hops, self.rank]
                        or b['shape'][0] != self.hops or b['shape'][2] != self.rank
                        or a['dtype'] != b['dtype'] or a['dtype'] not in ('BF16', 'F16', 'F32')):
                    raise ValueError('Invalid stacked PTD factor geometry: ' + raw)
                if map_adapter_key(pair['down'])[0].swap_halves and b['shape'][1] % 2:
                    raise ValueError('PTD SwiGLU output width must be even')
                if base is not None:
                    spec = base.header.get(target + '.weight', {}).get('shape')
                    part = map_adapter_key(pair['down'])[0]
                    width = b['shape'][1] * (3 if part.qkv_index is not None else 1)
                    if spec != [width, a['shape'][2]]:
                        raise ValueError('PTD factor does not match base projection: ' + target)
        for prefix, target in (('proj_out', 'final_layer.video_out'), ('audio_proj_out', 'final_layer.audio_out')):
            w, b = self.header[prefix + '.weight'], self.header[prefix + '.bias']
            if (len(w['shape']) != 3 or w['shape'][0] != self.steps or b['shape'] != w['shape'][:2]
                    or w['dtype'] not in ('BF16', 'F16', 'F32') or b['dtype'] != w['dtype']):
                raise ValueError('Invalid PDD head bank: ' + prefix)
            if base is not None and base.header.get(target + '.weight', {}).get('shape') != w['shape'][1:]:
                raise ValueError('PTD head does not match base projection: ' + prefix)

    def check(self):
        stat = self.path.stat()
        if (stat.st_size, stat.st_mtime_ns) != self.signature:
            raise RuntimeError('PTD checkpoint changed during baking')

    def slice(self, key, index):
        self.check()
        with safe_open(str(self.path), framework='pt', device='cpu') as handle:
            value = handle.get_slice(key)[index].contiguous()
        self.check()
        if not torch.isfinite(value).all():
            raise ValueError('Nonfinite PTD weights: ' + key)
        return value


class HopStore:
    """Adapter for the existing Native merge and independent-verification APIs."""
    def __init__(self, checkpoint, hop, strength=1.):
        if type(hop) is not int or not 0 <= hop < checkpoint.hops or not math.isfinite(strength):
            raise ValueError('A valid PTD hop and finite strength are required')
        self.checkpoint, self.hop, self.strength = checkpoint, hop, float(strength)
        self.targets = set(checkpoint.targets)

    def check(self):
        self.checkpoint.check()

    def terms(self, target):
        self.check()
        result = []
        for _, pair in sorted(self.checkpoint.targets.get(target, {}).items()):
            mapped, _ = map_adapter_key(pair['down'])
            a = self.checkpoint.slice(pair['down'], self.hop)
            b = self.checkpoint.slice(pair['up'], self.hop)
            if mapped.swap_halves:
                up, gate = b.chunk(2, 0)
                b = torch.cat((gate, up), 0)
            if mapped.qkv_index is not None:
                # Keep separate A matrices for Q/K/V. A shared concatenated A
                # or an accidental QKV broadcast changes this checkpoint.
                width = b.shape[0]
                padded = torch.zeros((3 * width, b.shape[1]), dtype=b.dtype)
                padded[mapped.qkv_index * width:(mapped.qkv_index + 1) * width].copy_(b)
                b = padded
            result.append((a, b.contiguous(), self.strength * self.checkpoint.alpha / self.checkpoint.rank))
        return result


def download_on_t4(directory):
    """Explicit device-side delivery; no weight request is made on Windows/SM120."""
    if (not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 5)
            or 'T4' not in torch.cuda.get_device_name()):
        raise RuntimeError('Download the PTD LoRA only on the target T4, never on the local machine')
    from huggingface_hub import hf_hub_download
    from patches.tp2_bundle import sha
    path = Path(hf_hub_download(RELEASE['repo'], RELEASE['file'], revision=RELEASE['revision'],
                               local_dir=str(directory), token=False))
    if path.stat().st_size != RELEASE['bytes'] or sha(path) != RELEASE['sha256']:
        raise ValueError('PTD release size / SHA256 mismatch')
    return path
