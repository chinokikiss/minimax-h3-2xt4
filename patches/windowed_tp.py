"""Checkpoint-backed TP layers for a request-wide residency window.

The caller retains Native embeddings, preprocessing, masks, RoPE, heads and
sampler interfaces. Only the already-validated TP block weights are streamed.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import struct
import types
import threading
from contextlib import contextmanager

import torch
from torch import nn

from patches.weight_residency import WeightSpec, module_bytes


class CheckpointReader:
    """Read selected safetensors ranges without mapping a 20–27GB file."""
    DTYPES = {'F32': torch.float32, 'F16': torch.float16, 'BF16': torch.bfloat16,
              'I8': torch.int8, 'U8': torch.uint8, 'I64': torch.int64,
              'I32': torch.int32, 'BOOL': torch.bool}

    def __init__(self, path, identity=None):
        self.path = Path(path).resolve()
        with self.path.open('rb') as stream:
            length = struct.unpack('<Q', stream.read(8))[0]
            self.header = json.loads(stream.read(length))
        self.offset = 8 + length
        stat = self.path.stat()
        self.signature = (stat.st_size, stat.st_mtime_ns)
        self.identity = identity or (str(self.path), *self.signature)
        self.read_lock = threading.Lock()
        self.logical_read_bytes = self.range_read_calls = 0

    def read_tensor(self, name, *, rows=None, columns=None):
        """Read only owned ranges into final CPU storage, without a full tensor.

        Column slices issue one range per row. Positional reads avoid file cursor
        mutation; os.pread returns bytes, copied into final tensor storage in
        bounded chunks. This is buffered filesystem I/O, not O_DIRECT.
        Logical bytes count requested payload, not physical disk traffic.
        """
        stat = self.path.stat()
        if (stat.st_size, stat.st_mtime_ns) != self.signature:
            raise RuntimeError('Checkpoint changed after the request was planned')
        info = self.header[name]
        shape = info['shape']
        dtype = self.DTYPES[info['dtype']]
        element = torch.empty((), dtype=dtype).element_size()
        intervals = rows or [(0, shape[0] if shape else 1)]
        if len(shape) not in (1, 2):
            if rows or columns:
                raise ValueError('Only vectors/matrices support shard ranges')
            intervals = [(0, 1)]
        width = shape[1] if len(shape) == 2 else 1
        first, last = columns or (0, width)
        if not 0 <= first < last <= width:
            raise ValueError('Invalid column range')
        if shape and any(not 0 <= a < b <= shape[0] for a, b in intervals):
            raise ValueError('Invalid row range')
        output_shape = ([sum(b-a for a,b in intervals), last-first]
                        if len(shape) == 2 else
                        [sum(b-a for a,b in intervals)] if len(shape) == 1 else shape)
        result = torch.empty(output_shape, dtype=dtype, device='cpu')
        destination = memoryview(result.reshape(-1).view(torch.uint8).numpy()).cast('B')
        count = calls = 0
        with self.path.open('rb', buffering=0) as stream:
            pread = getattr(os, 'pread', None)
            def transfer(offset, size, destination_offset):
                nonlocal count, calls, pread
                absolute = self.offset + info['data_offsets'][0] + offset
                remaining = destination[destination_offset:destination_offset+size]
                done = 0
                while remaining:
                    if pread is not None:
                        try:
                            data = pread(stream.fileno(), min(len(remaining), 8*1024*1024), absolute+done)
                        except (OSError, ValueError, TypeError, NotImplementedError):
                            # E.g. Windows or a non-standard/non-positional fd.
                            # Resume at the same absolute byte after short reads.
                            pread = None
                            continue
                        n = len(data)
                        remaining[:n] = data
                    else:
                        stream.seek(absolute+done)
                        n = stream.readinto(remaining)
                    calls += 1
                    if not n:
                        raise IOError('Checkpoint range is incomplete')
                    remaining = remaining[n:]
                    done += n
                count += size
            if len(shape) > 2:
                transfer(0, result.numel()*element, 0)
            else:
                output_row = 0
                for a,b in intervals:
                    if columns is None:
                        transfer(a*width*element, (b-a)*width*element, output_row*(last-first)*element)
                    else:
                        for row in range(a,b):
                            transfer((row*width+first)*element, (last-first)*element,
                                     (output_row+row-a)*(last-first)*element)
                    output_row += b-a
        with self.read_lock:
            self.logical_read_bytes += count
            self.range_read_calls += calls
        return result

    def read(self, prefix, exclude=()):
        stat = self.path.stat()
        if (stat.st_size, stat.st_mtime_ns) != self.signature:
            raise RuntimeError('Checkpoint changed after the request was planned')
        result = {}
        with self.path.open('rb') as stream:
            for name, info in self.header.items():
                if name == '__metadata__' or not name.startswith(prefix) or name.startswith(tuple(exclude)):
                    continue
                start, end = info['data_offsets']
                stream.seek(self.offset + start)
                data = bytearray(end - start)
                if stream.readinto(data) != end - start:
                    raise IOError('Checkpoint range is incomplete')
                result[name[len(prefix):]] = torch.frombuffer(
                    data, dtype=self.DTYPES[info['dtype']]).reshape(info['shape'])
        if not result:
            raise ValueError(f'No checkpoint tensors under {prefix}')
        with self.read_lock:
            self.logical_read_bytes += sum(t.numel()*t.element_size() for t in result.values())
            self.range_read_calls += len(result)
        return result

    def bytes(self, prefix):
        return sum(v['data_offsets'][1] - v['data_offsets'][0]
                   for k, v in self.header.items() if k != '__metadata__' and k.startswith(prefix))


def prepare_tp_block_legacy(reader, kind, index, rank):
    """Compatibility entry point for explicit reference qualification only."""
    from patches.reference_loaders import prepare_tp_block_legacy as reference
    return reference(reader, kind, index, rank)


def prepare_tp_block(reader, kind, index, rank):
    from patches.rank_local_int8 import load_rank_block, load_offline_block
    if getattr(reader, 'rank_checkpoint_dir', None):
        block=load_offline_block(reader,kind,index,rank,reader.rank_checkpoint_dir)
    else:block=load_rank_block(reader, kind, index, rank)
    store=getattr(reader,'lora_store',None)
    if store is not None and kind=='dit':block=store.attach_tp(block,index,rank)
    policy=getattr(reader,'operator_policy',None)
    if policy is not None:
        from patches.h3_operator_policy import configure_block
        block=configure_block(block,kind,policy)
    return block


def tp_specs(reader, kind, rank, layers=50):
    specs = []
    for index in range(layers):
        prefix = f'model.layers.{index}.' if kind == 'te' else f'blocks.{index}.'
        store=getattr(reader,'lora_store',None) if kind=='dit' else None
        identity=(reader.identity,store.identity) if store is not None else reader.identity
        policy=getattr(reader,'operator_policy',None)
        if policy is not None:identity=(identity,'h3-operators-v1',policy)
        key = (identity, kind, index, rank)
        # Full checkpoint block size conservatively bounds a half-weight shard
        # with replicated norms/scales/AdaLN. Admission uses this upper bound;
        # GPU accounting uses measured physical payload bytes after preparation.
        from patches.rank_local_int8 import rank_block_bytes
        bound = rank_block_bytes(reader, kind, index, rank) + 2**20
        adapter_bytes=store.block_bytes(kind,index,rank) if store is not None else 0
        bound+=adapter_bytes
        specs.append(WeightSpec(key,
            lambda i=index: prepare_tp_block(reader, kind, i, rank), bound,
            prepare_peak_bytes=bound + adapter_bytes*4 + 8*2**20))
    return specs


class WindowedTPBlock(nn.Module):
    def __init__(self, kind, window, key):
        super().__init__()
        self.kind, self.window, self.key = kind, window, key

    def forward(self, x, *args, **kwargs):
        with self.window.use(self.key) as block:
            if self.kind == 'te':
                if kwargs.get('past_key_value') is not None:
                    raise ValueError('TE streaming currently supports Native prefill only')
                out = block(x, kwargs.get('attention_mask'), kwargs.get('freqs_cis'))
                return out, None
            if kwargs.get('attention') is not None:
                raise ValueError('Custom attention requires separate TP qualification')
            kwargs.pop('attention', None)
            return block(x, *args, **kwargs)


def attach_window(model, kind, window, keys):
    """Attach to a Native model shell without changing its outer forward."""
    name = 'layers' if kind == 'te' else 'blocks'
    setattr(model, name, nn.ModuleList([WindowedTPBlock(kind, window, key) for key in keys]))
    return model


class DiTPreprocessing:
    """Native once-per-request text refinement and per-call input projection.

    Rank zero computes Native preprocessing, then broadcasts exact outputs.
    Model shells are private per rank. CPU canonical modules remain immutable;
    CUDA modules are leased from the same window as TE/DiT transformer blocks.
    """
    def __init__(self, model, rank, device, comm, identity):
        self.model, self.rank = model, rank
        self.device, self.comm = torch.device(device), comm
        self.window = None
        self.refine_key = (identity, 'dit_refine', rank)
        self.project_key = (identity, 'dit_project', rank)
        names = ('condition_proj', 'token_refiner', 'video_patch_proj', 'audio_patch_proj')
        self.canonical = {name: getattr(model, name).cpu() for name in names} if rank == 0 else {}
        self.original_refine = model.preprocess_text_embeds
        self.original_pack = getattr(model,'_embed_and_pack',None)
        self.project_uses = 1 if self.original_pack is not None else 2
        self.proxies={}
        if rank != 0:
            # These are never executed on this rank; release duplicate backing.
            for name in names:
                setattr(model, name, nn.Identity())
        model.preprocess_text_embeds = types.MethodType(lambda owner, text: self.refine(text), model)
        if self.original_pack is not None:
            model._embed_and_pack = types.MethodType(
                lambda owner, video, audio, context, layout, payload, options:
                    self.pack(video, audio, context, layout, payload, options), model)
        else:
            # Current ComfyUI inlines packing in _forward. Keep that code intact
            # and broadcast the two projection outputs at their module boundary.
            for name in ('video_patch_proj','audio_patch_proj'):
                proxy=InlineDiTProjection(self,name);self.proxies[name]=proxy
                setattr(model,name,proxy)

    def specs(self):
        if self.rank != 0:
            return []
        def make(names):
            return nn.ModuleDict({name: self.canonical[name] for name in names})
        result = []
        for key, names in [(self.refine_key, ('condition_proj', 'token_refiner')),
                           (self.project_key, ('video_patch_proj', 'audio_patch_proj'))]:
            bound = module_bytes(make(names)) + 2**20
            # Canonical CPU modules already exist and are budgeted by the
            # runtime. Preparing this group creates only a ModuleDict shell.
            result.append(WeightSpec(key, lambda names=names: make(names), bound,
                                    prepare_peak_bytes=bound, resident_backing=True))
        return result

    def close(self):
        """Break bound-method cycles before releasing auxiliary reservations."""
        del self.model.preprocess_text_embeds
        if self.original_pack is not None:del self.model._embed_and_pack
        else:
            for name in self.proxies:setattr(self.model,name,self.canonical.get(name,nn.Identity()))
            self.proxies.clear()
        self.original_refine = self.original_pack = self.window = None
        self.canonical.clear()

    @contextmanager
    def _modules(self, key):
        if self.window is None:
            raise RuntimeError('Bind a residency window before invoking preprocessing')
        with self.window.use(key) as modules:
            previous={name:getattr(self.model,name) for name in modules}
            for name, module in modules.items():
                setattr(self.model, name, module)
            try:
                yield
            finally:
                for name in modules:
                    setattr(self.model, name, previous[name])

    def refine(self, text):
        if text.shape[-1] == self.model.hidden_size:
            return text
        if self.rank == 0:
            with self._modules(self.refine_key):
                refined = self.original_refine(text)
        else:
            # Native MiniMax refiner currently consumes text_states[0].
            if text.shape[0] != 1:
                raise ValueError('This Native Ref2VA adapter supports batch one')
            refined = torch.empty((1, text.shape[1], self.model.hidden_size),
                                  dtype=text.dtype, device=self.device)
        self.comm.broadcast(refined, src=0)
        return refined

    def pack(self, video, audio, context, layout, payload, options):
        if context.shape[-1] != self.model.hidden_size:
            raise ValueError('Call Native text refinement once before denoising')
        if self.rank == 0:
            with self._modules(self.project_key):
                packed = self.original_pack(video, audio, context, layout, payload, options)
        else:
            packed = torch.empty((layout.seq_len, self.model.hidden_size),
                                 dtype=context.dtype, device=self.device)
        self.comm.broadcast(packed, src=0)
        return packed

    def project(self,name,x):
        if self.rank==0:
            with self._modules(self.project_key):out=getattr(self.model,name)(x)
            if out.shape!=(*x.shape[:-1],self.model.hidden_size) or out.dtype!=x.dtype:
                raise ValueError('Unqualified inline H3 projection shape/dtype')
            out=out.contiguous()
        else:out=torch.empty((*x.shape[:-1],self.model.hidden_size),dtype=x.dtype,device=self.device)
        self.comm.broadcast(out,src=0)
        return out


class InlineDiTProjection(nn.Module):
    """Weightless outer-model proxy; canonical projection weights stay streamed."""
    def __init__(self,owner,name):super().__init__();self.owner=owner;self.name=name
    def forward(self,x):return self.owner.project(self.name,x)
