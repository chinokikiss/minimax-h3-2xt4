"""Direct TP2 checkpoint ranges and optional contiguous offline rank blocks.

Only the pinned Native INT8 ConvRot TE/DiT configurations are supported.
Norm/AdaLN casting follows Native construction; quantization never changes.
"""
import hashlib
import json
from pathlib import Path

import torch

FORMAT = 'minimax-h3-tp2-native-int8-v1'
CONFIG = dict(world_size=2, convrot=True, convrot_groupsize=256,
              te_weight_dtype='float16', dit_weight_dtype='float32',
              dit_apply_silu=False, te_hidden=5120, dit_hidden=5376,
              dit_time_dim=8)


def shell(kind, rank):
    if rank not in (0, 1):
        raise ValueError('Expected rank 0 or 1')
    with torch.device('meta'):
        if kind == 'te':
            from patches.tp_qwen3vl import TPTransformerBlockINT8
            module = TPTransformerBlockINT8(rank=rank, device='meta')
        elif kind == 'dit':
            from patches.tp_dit import TPDiTBlockINT8
            module = TPDiTBlockINT8(rank=rank, device='meta')
        else:
            raise ValueError('Only TE and DiT TP2 blocks are supported')
    for child in module.modules():
        if hasattr(child, 'device'):
            child.device = torch.device('cpu')
    return module


def plan(reader, kind, index, rank):
    """Explicit component-wise packed rows, columns, replicated scales/norms."""
    expected = shell(kind, rank).state_dict()
    prefix = f'model.layers.{index}.' if kind == 'te' else f'blocks.{index}.'
    if kind == 'te':
        linear = {n: ('self_attn.' if n in ('q_proj','k_proj','v_proj','o_proj') else 'mlp.')+n
                  for n in ('q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj')}
        column = {'q_proj':1,'k_proj':1,'v_proj':1,'gate_proj':1,'up_proj':1}
        replicated = {n+'.weight': ('self_attn.' if n in ('q_norm','k_norm') else '')+n+'.weight'
                      for n in ('input_layernorm','post_attention_layernorm','q_norm','k_norm')}
    else:
        linear = dict(qkv_proj='attn.qkv_proj',out_proj='attn.out_proj',fc1='mlp.fc1',fc2='mlp.fc2')
        column = {'qkv_proj':3,'fc1':2}
        replicated = {n+'.weight': ('attn.' if n in ('q_norm','k_norm') else '')+n+'.weight'
                      for n in ('norm1','norm2','q_norm','k_norm')}
        replicated.update({'adaln_proj.'+n:'adaln_proj.linear.'+n for n in ('weight','bias')})
    result = []
    known = set()
    for target, source in replicated.items():
        result.append((target,prefix+source,None,None))
        known.add(prefix+source)
    for target,source in linear.items():
        for suffix in ('weight','weight_scale'):
            name = prefix+source+'.'+suffix
            info = reader.header[name]
            if info['dtype'] != ('I8' if suffix == 'weight' else 'F32'):
                raise ValueError('Unsupported Native quantization dtype: '+name)
            rows = columns = None
            if target in column:
                components = column[target]
                full = info['shape'][0]
                if full % (2*components):
                    raise ValueError('Invalid packed component shape')
                chunk = full//components
                rows = [(i*chunk+rank*chunk//2, i*chunk+(rank+1)*chunk//2)
                        for i in range(components)]
            elif suffix == 'weight':
                width = info['shape'][1]
                if width % 512:
                    raise ValueError('Row shard must preserve 256-channel ConvRot groups')
                columns = (rank*width//2,(rank+1)*width//2)
            result.append((target+'.'+suffix,name,rows,columns))
            known.add(name)
        known.add(prefix+source+'.comfy_quant')
    unexpected = {n for n in reader.header if n.startswith(prefix)}-known
    if unexpected:
        raise ValueError('Unqualified Native block metadata/tensors: '+str(sorted(unexpected)))
    if set(expected) != {t for t,_,_,_ in result}:
        raise ValueError('Rank block schema differs from qualified TP module')
    for target,name,rows,columns in result:
        shape = list(reader.header[name]['shape'])
        if rows: shape[0] = sum(b-a for a,b in rows)
        if columns: shape[1] = columns[1]-columns[0]
        if shape != list(expected[target].shape):
            raise ValueError('Checkpoint shape differs from Native TP configuration: '+name)
    return result, expected, [prefix+source+'.comfy_quant' for source in linear.values()]


def rank_block_bytes(reader, kind, index, rank):
    _, expected, _ = plan(reader,kind,index,rank)
    return sum(t.numel()*t.element_size() for t in expected.values())


def load_rank_block(reader, kind, index, rank):
    mappings, expected, quant_names = plan(reader,kind,index,rank)
    metadata = {}
    for name in quant_names:
        raw = reader.read_tensor(name)
        config = json.loads(bytes(raw.tolist()).decode('utf8'))
        if config != {'format':'int8_tensorwise','convrot':True,'convrot_groupsize':256}:
            raise ValueError('Unqualified quantization metadata: '+str(config))
        metadata[name] = config
    state = {target: reader.read_tensor(name,rows=rows,columns=columns).to(expected[target].dtype)
             for target,name,rows,columns in mappings}
    module = shell(kind,rank)
    module.load_state_dict(state,strict=True,assign=True)
    module._rank_checkpoint_metadata = metadata
    return module.eval().requires_grad_(False)


def state_hashes(module):
    return {k: dict(dtype=str(t.dtype),shape=list(t.shape),
                   sha256=hashlib.sha256(memoryview(t.detach().contiguous().reshape(-1)
                       .view(torch.uint8).numpy()).cast('B')).hexdigest())
            for k,t in module.state_dict().items()}


def export_block(reader, kind, index, rank, directory, *, native_comparator):
    """Offline conversion requires actual Native-loaded tensor equality first."""
    from safetensors.torch import save_file
    module = load_rank_block(reader,kind,index,rank)
    native = native_comparator(reader,kind,index,rank)
    if state_hashes(module) != state_hashes(native):
        raise ValueError('Offline conversion is not Native bit-exact')
    root = Path(directory).resolve(); root.mkdir(parents=True,exist_ok=True)
    stem = f'{kind}.block{index:02d}.rank{rank}'
    path = root/(stem+'.safetensors')
    manifest = dict(format=FORMAT,configuration=CONFIG,source_identity=reader.identity,
        kind=kind,index=index,rank=rank,state_sha256=state_hashes(module),
        quant_metadata=module._rank_checkpoint_metadata,native_tensor_bit_exact=True)
    temporary = root/(stem+'.partial')
    save_file(module.state_dict(),str(temporary),metadata={'format':FORMAT})
    temporary.replace(path)
    with path.open('rb') as stream:
        manifest['file_sha256'] = hashlib.file_digest(stream,'sha256').hexdigest()
    (root/(stem+'.json')).write_text(json.dumps(manifest,indent=2))
    return path


def load_offline_block(source_reader, kind, index, rank, directory):
    from patches.windowed_tp import CheckpointReader
    root = Path(directory).resolve(); stem = f'{kind}.block{index:02d}.rank{rank}'
    source_stat = source_reader.path.stat()
    if (source_stat.st_size,source_stat.st_mtime_ns) != source_reader.signature:
        raise RuntimeError('Source checkpoint changed since offline planning')
    manifest_path = root/(stem+'.json')
    manifest = json.loads(manifest_path.read_text())
    if (manifest['format'] != FORMAT or manifest['configuration'] != CONFIG or
        manifest['source_identity'] != json.loads(json.dumps(source_reader.identity)) or
        (manifest['kind'],manifest['index'],manifest['rank']) != (kind,index,rank) or
        manifest['native_tensor_bit_exact'] is not True):
        raise ValueError('Offline checkpoint provenance/configuration mismatch')
    reader = CheckpointReader(root/(stem+'.safetensors'))
    module = shell(kind,rank)
    state = {name:reader.read_tensor(name) for name in module.state_dict()}
    module.load_state_dict(state,strict=True,assign=True)
    verified = getattr(source_reader,'rank_offline_verified',{})
    signature = (*reader.signature,manifest_path.stat().st_mtime_ns,manifest_path.stat().st_size)
    # Verify each immutable artifact once per reader lifetime. Subsequent loads
    # use file signatures, just like the Native checkpoint reader. Verification
    # belongs to cold/startup latency and is never silently omitted there.
    if verified.get(str(reader.path)) != signature:
        if state_hashes(module) != manifest['state_sha256']:
            raise ValueError('Offline rank tensor checksum mismatch')
        verified[str(reader.path)] = signature
        source_reader.rank_offline_verified = verified
    module._rank_checkpoint_metadata = manifest['quant_metadata']
    module._rank_offline_read_bytes = reader.logical_read_bytes
    with source_reader.read_lock:
        source_reader.logical_read_bytes += reader.logical_read_bytes
        source_reader.range_read_calls += reader.range_read_calls
    return module.eval().requires_grad_(False)
