"""Optional dense FP32 refiner sidecars; original Native overlays are retained."""
from dataclasses import asdict
import hashlib
import os
from pathlib import Path

import torch
from safetensors.torch import save_file

from patches.h3_tensor_flow import TensorFlowPolicy
from patches.tp2_bundle import sha, signature
from patches.windowed_tp import CheckpointReader

FORMAT = 'ptd-refiner-fp32-native-fold-v1'


def tensor_hash(value):
    return hashlib.sha256(memoryview(value.contiguous().view(torch.uint8).numpy())).hexdigest()


def write_layout(values, directory, source_sha256, flow):
    """Exactly Native load-to-FP32, then FP32 division for branch outputs.

    This is a dense parameter layout, not a new packed SM75 GEMM format.
    No activation operation or normalization/epsilon fold is reordered.
    """
    prepared, proofs = {}, {}
    for name, source in values.items():
        if not name.startswith('token_refiner.'): continue
        value = source.to(dtype=torch.float32)
        divisor = flow.divisor if flow.enabled and name.endswith(('attn.out_proj.weight','mlp.fc2.weight')) else 1.
        if divisor != 1.: value = value / divisor
        value = value.contiguous()
        # Independent Native construction order: destination FP32 copy first,
        # then the existing output-weight fold. Retain byte-exact validation.
        expected = torch.empty(source.shape, dtype=torch.float32)
        expected.copy_(source)
        if divisor != 1.: expected = expected / divisor
        if not torch.equal(value, expected):
            raise ValueError('Runtime refiner layout differs from Native folding: '+name)
        prepared[name] = value
        proofs[name] = dict(shape=list(value.shape), sha256=tensor_hash(value))
        del expected
    if not prepared: raise ValueError('No refiner matrices to prepare')
    path = Path(directory)/'refiner_runtime.safetensors'
    save_file(prepared, str(path))
    with path.open('r+b') as stream: os.fsync(stream.fileno())
    return dict(format=FORMAT, file=path.name, bytes=path.stat().st_size,
        sha256=sha(path), source_auxiliary_sha256=source_sha256,
        flow=asdict(flow), tensors=proofs, native_byte_exact_checks=len(proofs))


def layout_proof(reader, flow):
    auxiliary = reader.bank.hops[reader.hop][1]['auxiliary']
    proof = auxiliary.get('runtime_layout')
    if (not proof or proof['format'] != FORMAT or proof['flow'] != asdict(flow)
            or proof['source_auxiliary_sha256'] != auxiliary['sha256']):
        raise ValueError('Missing or incompatible Native refiner runtime layout')
    return proof


def read_layout_tensor(reader, name, flow):
    proof = layout_proof(reader, flow)
    path = reader.root/proof['file']
    if path.resolve().parent != reader.root.resolve() or path.stat().st_size != proof['bytes']:
        raise ValueError('Invalid refiner runtime layout path/size')
    if not hasattr(reader, '_runtime_refiner'):
        reader._runtime_refiner = CheckpointReader(path)
    handle = reader._runtime_refiner
    if signature(path) != handle.signature:
        raise RuntimeError('Refiner runtime sidecar changed during sampling')
    value = handle.read_tensor(name)
    tensor = proof['tensors'][name]
    if value.dtype != torch.float32 or list(value.shape) != tensor['shape'] or tensor_hash(value) != tensor['sha256']:
        raise ValueError('Refiner runtime tensor failed integrity validation: '+name)
    return value
