"""Standalone bounded, single-hop PTD/ordinary-LoRA Baker and Bank v2 assembler.

The eight expert parts are baked separately. A caller must publish/read back
each completed immutable part before reclaiming its owned scratch directory.
This module never downloads, uploads, remounts, or deletes remote datasets.
"""
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import math
import shutil

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from patches.tp2_bundle import BundleReader, DIT_LINEAR, sha, split_matrix
from patches.tp2_disk_bake import _LayerSource, _persist_layer, atomic_json, state_proof
from patches.windowed_tp import CheckpointReader
from .bank_v2 import FORMAT_V2, PTDBankV2
from .checkpoint import PTDCheckpoint
from .composite_lora import CompositeLoRAStore
from .schedule import hop_plans, hop_sigmas

PART_FORMAT = 'h3-ptd-tp2-part-v2'


def _recipe_fields():
    return ('format', 'layers', 'hops', 'steps', 'video_shift', 'audio_shift',
            'quantization', 'base_bundle_sha256', 'base_descriptor_sha256',
            'ptd_checkpoint_sha256', 'ordinary_loras', 'pruned_adaln_approximation')


def _merge_verified_once(reader, store, target, device):
    """Native FP32 composition once; validate exported bytes without another GEMM.

    Set PTD_V2_INDEPENDENT_VERIFY=1 for the slower full per-target independent
    A/B reference audit. The default fast path verifies quantization from the
    retained Native FP32 value, not the independent LoRA algebra.
    """
    from patches.offline_lora import merged_weight
    from patches.offline_lora_reference import verify_cached_weight, verify_weight

    weight, scale, reference = merged_weight(reader, store, target, device,
                                             return_reference=True)
    try:
        verify_cached_weight(target, weight, scale, reference)
        if os.environ.get('PTD_V2_INDEPENDENT_VERIFY') == '1':
            verify_weight(reader, store, target, weight, scale, device)
    finally:
        del reference
    return weight, scale


def _recipe(base, checkpoint, ordinary_loras, layers):
    return dict(format=FORMAT_V2, layers=layers, hops=checkpoint.hops, steps=checkpoint.steps,
                video_shift=12., audio_shift=3., quantization='native-convrot256-int8-tp2',
                base_bundle_sha256=sha(base.bundle_file),
                base_descriptor_sha256=sha(base.path),
                ptd_checkpoint_sha256=sha(checkpoint.path),
                ordinary_loras=[dict(sha256=sha(path), bytes=Path(path).stat().st_size,
                                     strength=float(strength))
                                for path, strength in ordinary_loras],
                pruned_adaln_approximation='adaln_t_table' in base.header)


def _verify_inputs(base, checkpoint, layers, allow_pruned_adaln, allow_fixture):
    if base.kind != 'dit' or not 1 <= layers <= 50 or checkpoint.blocks < layers:
        raise ValueError('PTD Bank v2 requires a qualified DiT base and all adapted layers')
    if (layers != 50 or checkpoint.blocks != 50) and not allow_fixture:
        raise ValueError('Production PTD Bank v2 requires 50 adapted layers')
    if checkpoint.hops != 8 or checkpoint.steps != 32:
        raise ValueError('PTD Bank v2 requires eight hops and 32 PDD intervals')
    refiner_indices = {int(name.split('.')[2]) for name in base.header
                       if name.startswith('token_refiner.blocks.')}
    if refiner_indices != set(range(checkpoint.refiners)):
        raise ValueError('PTD v2 must adapt all Native token-refiner blocks')
    if 'adaln_t_table' in base.header and not allow_pruned_adaln:
        raise ValueError('Pruned AdaLN requires explicit approximation consent')


def estimate_hop_scratch(base, store, layers):
    """Match the Baker's pre-write scratch gate without allocating any payload."""
    block_bytes = sum(info['data_offsets'][1] - info['data_offsets'][0]
                      for name, info in base.header.items()
                      if name.startswith('blocks.') and int(name.split('.')[1]) < layers)
    overlay_bytes = sum(math.prod(base.header[target + '.weight']['shape']) * 4
                        for target in store.targets if target.startswith('token_refiner.'))
    overlay_bytes += sum(math.prod(base.header[name]['shape']) * 4
                         for name in ('final_layer.video_out.weight',
                                      'final_layer.video_out.bias',
                                      'final_layer.audio_out.weight',
                                      'final_layer.audio_out.bias'))
    minimum_free_bytes = math.ceil(1.15 * (block_bytes + overlay_bytes)) + 2**30
    return dict(base_block_bytes=block_bytes, overlay_fp32_bytes=overlay_bytes,
                minimum_free_bytes=minimum_free_bytes)


def _read_layer_states(reader, index):
    """Bounded CPU prefetch with the original rank-state SHA256 admission."""
    states = []
    for rank in (0, 1):
        source = reader._shard(index, rank)
        state = {name: source.read_tensor(name)
                 for name in source.header if name != '__metadata__'}
        if state_proof(state) != reader.plan['blocks'][f'dit.block{index:02d}.rank{rank}']['state_sha256']:
            raise ValueError('PTD v2 base rank-state proof mismatch')
        states.append(state)
    return states


def _partition(base_path, checkpoint_path, ordinary_loras, hop, output, layers, device,
               recipe_sha, *, prefetch_layers=False):
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    if torch.device(device).type == 'cuda':
        torch.cuda.set_device(device)
    base = BundleReader(base_path)
    checkpoint = PTDCheckpoint(checkpoint_path, base)
    store = CompositeLoRAStore(checkpoint, hop, base, ordinary_loras)
    records, verified = {}, []
    output = Path(output)
    pending = None
    # Use an independent I/O reader; the GPU thread owns its own metadata caches.
    # One prefetched layer plus one durable-write layer bounds RAM ownership.
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix='ptd-v2-durable-write') as writer, \
         ThreadPoolExecutor(max_workers=1, thread_name_prefix='ptd-v2-prefetch') as prefetcher:
        source_reader = BundleReader(base_path) if prefetch_layers else base
        indices = iter(layers)
        next_index = next(indices, None)
        next_future = (prefetcher.submit(_read_layer_states, source_reader, next_index)
                       if prefetch_layers and next_index is not None else None)
        while next_index is not None:
            index = next_index
            states = (next_future.result() if prefetch_layers
                      else _read_layer_states(source_reader, index))
            next_index = next(indices, None)
            next_future = (prefetcher.submit(_read_layer_states, source_reader, next_index)
                           if prefetch_layers and next_index is not None else None)
            snapshot = _LayerSource(base, index, states)
            targets = sorted(t for t in store.targets if t.startswith(f'blocks.{index}.'))
            merged = [dict(state) for state in states]
            for target in targets:
                q, scale = _merge_verified_once(snapshot, store, target, device)
                snapshot.cache.clear()
                local = DIT_LINEAR[target.split('.', 2)[2]]
                for rank in (0, 1):
                    merged[rank][local + '.weight'] = split_matrix(q, local, rank)
                    if scale is not None:
                        merged[rank][local + '.weight_scale'] = split_matrix(scale, local, rank, scale=True)
                verified.append(target)
                del q, scale
            if pending is not None:
                records.update(pending.result())
            pending = writer.submit(_persist_layer, merged, index, targets, output,
                                    base.plan['blocks'], recipe_sha)
            del snapshot, states, merged
            print('PTD_V2_LAYER_QUEUED', hop, index, device, flush=True)
        if pending is not None:
            records.update(pending.result())
        if prefetch_layers:
            source_reader.check()
    store.check()
    base.check()
    return dict(shards=records, targets=verified)


def _hop_auxiliary(base, checkpoint, ordinary_loras, hop, destination, device):
    store = CompositeLoRAStore(checkpoint, hop, base, ordinary_loras)
    targets = sorted(t for t in store.targets if t.startswith('token_refiner.'))
    values = {}
    for target in targets:
        weight, scale = _merge_verified_once(base, store, target, device)
        if scale is not None:
            raise ValueError('Refiner projection must retain its Native floating-point dtype')
        values[target + '.weight'] = weight.contiguous()
    plans = hop_plans(8, 32, 12., 3.)[hop]
    for prefix, native, plan in zip(
            ('proj_out', 'audio_proj_out'),
            ('final_layer.video_out', 'final_layer.audio_out'), plans):
        with safe_open(str(checkpoint.path), framework='pt', device='cpu') as handle:
            weight = handle.get_tensor(prefix + '.weight').to(device, dtype=torch.float32)
            bias = handle.get_tensor(prefix + '.bias').to(device, dtype=torch.float32)
        p = plan.to(device=device, dtype=torch.float32)
        values[native + '.weight'] = torch.einsum('pn,noi->poi', p, weight).flatten(0, 1).cpu().contiguous()
        values[native + '.bias'] = torch.einsum('pn,no->po', p, bias).flatten().cpu().contiguous()
    for name, value in values.items():
        if name not in base.header or base.header[name]['shape'] != list(value.shape) or not torch.isfinite(value).all():
            raise ValueError('PTD v2 auxiliary tensor schema/finiteness mismatch: ' + name)
    path = destination / 'aux_overlay.safetensors'
    save_file(values, str(path))
    with path.open('r+b') as stream:
        os.fsync(stream.fileno())
    store.check()
    return dict(file=path.name, bytes=path.stat().st_size,
                sha256=sha(path), verified_targets=targets,
                state_sha256=state_proof(values))


def _devices_ok(devices, fixture, *, independent_hop_gpu=False):
    if fixture and devices == ('cpu',):
        return
    if independent_hop_gpu and len(devices) == 1:
        pass  # Explicit opt-in: each physical T4 produces both TP2 rank shards.
    elif len(devices) != 2 or len(set(devices)) != 2:
        raise ValueError('PTD Bank v2 production Baker requires two separate T4 CUDA devices')
    for device in devices:
        if (torch.device(device).type != 'cuda' or
                torch.cuda.get_device_capability(device) != (7, 5) or
                'T4' not in torch.cuda.get_device_name(device)):
            raise RuntimeError('PTD Bank v2 Baker must run on two physical T4 GPUs')


def bake_hop_part(base_descriptor, checkpoint_path, hop, output, *,
                  ordinary_loras=(), layers=50, devices=('cuda:0', 'cuda:1'),
                  allow_pruned_adaln=False, allow_fixture=False,
                  independent_hop_gpu=False, prefetch_layers=False,
                  preverified_recipe=None):
    """Bake exactly one immutable hop, never requiring the other seven on /tmp."""
    if type(hop) is not int or not 0 <= hop < 8:
        raise ValueError('Hop must be 0-7')
    _devices_ok(devices, allow_fixture, independent_hop_gpu=independent_hop_gpu)
    base = BundleReader(base_descriptor)
    checkpoint = PTDCheckpoint(checkpoint_path, base)
    ordinary_loras = tuple((str(Path(path).resolve()), float(scale))
                           for path, scale in ordinary_loras)
    _verify_inputs(base, checkpoint, layers, allow_pruned_adaln, allow_fixture)
    store = CompositeLoRAStore(checkpoint, hop, base, ordinary_loras)
    targets = {t for t in store.targets if not t.startswith('blocks.')
               or int(t.split('.')[1]) < layers}
    if preverified_recipe is not None:
        if ordinary_loras or not independent_hop_gpu or allow_fixture:
            raise ValueError('Preverified recipe requires production independent-Hop mode without ordinary LoRA')
        recipe = dict(preverified_recipe)
        if (recipe.get('format') != FORMAT_V2 or recipe.get('layers') != layers
                or recipe.get('hops') != checkpoint.hops or recipe.get('steps') != checkpoint.steps
                or recipe.get('base_bundle_sha256') != base.bundle_digest
                or recipe.get('ordinary_loras') != []
                or recipe.get('pruned_adaln_approximation') != ('adaln_t_table' in base.header)
                or recipe.get('quantization') != 'native-convrot256-int8-tp2'
                or recipe.get('video_shift') != 12. or recipe.get('audio_shift') != 3.
                or set(recipe) != set(_recipe_fields())):
            raise ValueError('Preverified PTD v2 recipe does not match admitted source/schema')
    else:
        recipe = _recipe(base, checkpoint, ordinary_loras, layers)
    recipe_sha = hashlib.sha256(json.dumps(recipe, sort_keys=True).encode()).hexdigest()
    output = Path(output).resolve()
    if output.exists() or output.with_name(output.name + '.partial').exists():
        raise FileExistsError('Do not overwrite an existing hop part or resumable partial')
    # One-hop admission is independent of the other seven expert parts.
    required = estimate_hop_scratch(base, store, layers)['minimum_free_bytes']
    free = shutil.disk_usage(output.parent).free
    if free < required:
        raise OSError(f'Insufficient scratch for one PTD hop: need {required} free bytes, have {free}')
    partial = output.with_name(output.name + '.partial')
    partial.mkdir(parents=True)
    atomic_json(partial / 'recipe.json', recipe)
    partitions = [list(range(rank, layers, len(devices))) for rank in range(len(devices))]
    if len(devices) == 1:
        results = [_partition(str(base.path), str(checkpoint.path), ordinary_loras, hop,
                              str(partial), partitions[0], devices[0], recipe_sha,
                               prefetch_layers=prefetch_layers)]
    else:
        with ProcessPoolExecutor(max_workers=2, mp_context=mp.get_context('spawn')) as pool:
            jobs = [pool.submit(_partition, str(base.path), str(checkpoint.path),
                                ordinary_loras, hop, str(partial), part, device, recipe_sha)
                    for part, device in zip(partitions, devices)]
            results = [job.result() for job in jobs]
    shards = {}
    verified = []
    for result in results:
        if shards.keys() & result['shards'].keys():
            raise ValueError('Duplicate PTD v2 rank-shard worker output')
        shards.update(result['shards'])
        verified.extend(result['targets'])
    expected = {f'dit.block{i:02d}.rank{rank}' for i in range(layers) for rank in (0, 1)}
    if set(shards) != expected:
        raise ValueError('PTD v2 per-hop rank coverage incomplete')
    auxiliary = _hop_auxiliary(base, checkpoint, ordinary_loras, hop, partial, devices[0])
    verified.extend(auxiliary['verified_targets'])
    if set(verified) != targets or len(verified) != len(targets):
        raise ValueError('PTD v2 merged Native targets incomplete or duplicated')
    store.check()
    base.check()
    atomic_json(partial / 'receipt.json', dict(format=PART_FORMAT, status='complete',
                hop=hop, recipe_sha256=recipe_sha, shards=shards, auxiliary=auxiliary,
                verified_targets=sorted(verified)))
    partial.replace(output)
    return output / 'receipt.json'


def bake_shared_part(base_descriptor, checkpoint_path, output, *,
                     ordinary_loras=(), layers=50, allow_pruned_adaln=False,
                     allow_fixture=False):
    """Write only static auxiliary, Native header, and rank-state schema."""
    base = BundleReader(base_descriptor)
    checkpoint = PTDCheckpoint(checkpoint_path, base)
    _verify_inputs(base, checkpoint, layers, allow_pruned_adaln, allow_fixture)
    ordinary_loras = tuple((str(Path(p).resolve()), float(s)) for p, s in ordinary_loras)
    store = CompositeLoRAStore(checkpoint, 0, base, ordinary_loras)
    overlay_names = {t + '.weight' for t in store.targets if t.startswith('token_refiner.')}
    overlay_names.update({
        'final_layer.video_out.weight', 'final_layer.video_out.bias',
        'final_layer.audio_out.weight', 'final_layer.audio_out.bias'})
    static = set(base.aux.header) - {'__metadata__'} - overlay_names
    all_aux = {name for name in base.header
               if name != '__metadata__' and not name.startswith('blocks.')}
    if static | overlay_names != all_aux or static & overlay_names:
        raise ValueError('PTD v2 static/overlay Native schema is incomplete')
    output = Path(output).resolve()
    if output.exists() or output.with_name(output.name + '.partial').exists():
        raise FileExistsError('Refusing to replace existing Bank v2 shared part')
    partial = output.with_name(output.name + '.partial')
    partial.mkdir(parents=True)
    values = {name: base.aux.read_tensor(name).contiguous() for name in sorted(static)}
    path = partial / 'shared_base_aux.safetensors'
    save_file(values, str(path))
    with path.open('r+b') as stream:
        os.fsync(stream.fileno())
    # A two-layer physical qualification uses the real trained PTD subset but
    # must not advertise the other 48, unbaked Native blocks in its v2 header.
    # Production layers=50 preserves the full header byte-for-byte.
    exported_header = {
        name: info for name, info in base.header.items()
        if not name.startswith('blocks.') or int(name.split('.')[1]) < layers}
    atomic_json(partial / 'native_header_v2.json', exported_header)
    blocks = {f'dit.block{i:02d}.rank{rank}': base.plan['blocks'][f'dit.block{i:02d}.rank{rank}']
              for i in range(layers) for rank in (0, 1)}
    atomic_json(partial / 'block_plan_v2.json', {'blocks': blocks})
    header_path = partial / 'native_header_v2.json'
    plan_path = partial / 'block_plan_v2.json'
    recipe = _recipe(base, checkpoint, ordinary_loras, layers)
    atomic_json(partial / 'receipt.json', dict(format=PART_FORMAT, status='complete',
                recipe=recipe,
                recipe_sha256=hashlib.sha256(json.dumps(recipe, sort_keys=True).encode()).hexdigest(),
                aux='shared_base_aux.safetensors', aux_sha256=sha(path),
                aux_bytes=path.stat().st_size,
                native_header=header_path.name, native_header_sha256=sha(header_path),
                native_header_bytes=header_path.stat().st_size,
                block_plan=plan_path.name, block_plan_sha256=sha(plan_path),
                block_plan_bytes=plan_path.stat().st_size))
    store.check()
    base.check()
    partial.replace(output)
    return output / 'receipt.json'


def assemble_bank_v2(manifest, shared_root, hop_roots, *, allow_fixture=False, readback_proofs=None):
    """Join already published immutable parts; no payload copy or runtime bake.

    hop_roots is an eight-element ordered list of mounted/readback-verified
    directories. External mounts are explicit Loader arguments, not hot mounts.
    """
    if len(hop_roots) != 8:
        raise ValueError('Bank v2 manifest requires eight independently complete hop parts')
    manifest = Path(manifest).resolve()
    if manifest.exists():
        raise FileExistsError('Refusing to replace published Bank v2 manifest')
    roots = {'shared': Path(shared_root).resolve()}
    roots.update({f'hop_{hop:02d}': Path(root).resolve() for hop, root in enumerate(hop_roots)})
    if len(set(roots.values())) != 9:
        raise ValueError('Bank v2 requires nine distinct immutable part mounts')
    source = json.loads((roots['shared'] / 'receipt.json').read_text(encoding='utf-8'))
    if source.get('format') != PART_FORMAT or source.get('status') != 'complete':
        raise ValueError('Incomplete Bank v2 shared part')
    recipe, recipe_sha = source['recipe'], source['recipe_sha256']
    if not allow_fixture and (readback_proofs is None or len(readback_proofs) != 9):
        raise ValueError('Production Bank v2 assembly requires nine remote readback proofs')
    proof_paths = dict(zip(roots, readback_proofs)) if readback_proofs is not None else {}
    readbacks = {}
    if proof_paths:
        from .publish_v2 import check_readback_proof
        for part_id, root in roots.items():
            receipt = root / 'receipt.json'
            readbacks[part_id] = check_readback_proof(
                proof_paths[part_id], root, part_id, sha(receipt), recipe_sha,
                require_published_identity=not allow_fixture,
                expected_layers=recipe['layers'])
    parts = {}
    for part_id, root in roots.items():
        if not root.is_dir():
            raise FileNotFoundError(root)
        relative = root.relative_to(manifest.parent).as_posix() if root.is_relative_to(manifest.parent) else part_id
        if relative == '':
            relative = '.'
        identity = readbacks[part_id]['published_identity'] if part_id in readbacks else ''
        parts[part_id] = dict(root=relative, identity=identity or recipe_sha + ':' + part_id)
    hops = []
    for hop in range(8):
        part_id = f'hop_{hop:02d}'
        receipt_path = roots[part_id] / 'receipt.json'
        receipt = json.loads(receipt_path.read_text(encoding='utf-8'))
        if (receipt.get('format') != PART_FORMAT or receipt.get('status') != 'complete'
                or receipt.get('hop') != hop or receipt.get('recipe_sha256') != recipe_sha):
            raise ValueError('Bank v2 hop part belongs to another recipe or is incomplete')
        shards = {stem: dict(row, part=part_id, file=stem + '.safetensors',
                             manifest=stem + '.json') for stem, row in receipt['shards'].items()}
        auxiliary = dict(receipt['auxiliary'], part=part_id)
        hops.append(dict(hop=hop, status='complete', source_recipe_sha256=recipe_sha,
                         auxiliary=auxiliary, shards=shards))
    shared = dict(part='shared', aux=source['aux'], aux_sha256=source['aux_sha256'],
                  aux_bytes=source['aux_bytes'],
                  native_header=source['native_header'],
                  block_plan=source['block_plan'])
    staging = manifest.with_name(manifest.name + '.staging')
    if staging.exists():
        raise FileExistsError('Preserve the previous incomplete Bank v2 assembly attempt')
    atomic_json(staging, dict(format=FORMAT_V2, status='complete', online_lora=False,
        full_model=recipe['layers'] == 50, recipe=recipe, recipe_sha256=recipe_sha,
        parts=parts, shared=shared, hops=hops, sigmas=hop_sigmas(8, 32, 12.).tolist()))
    # Reject an incomplete assembly before publishing the final manifest name.
    PTDBankV2(staging, allow_fixture=allow_fixture,
              part_roots=tuple((part, str(root)) for part, root in roots.items()))
    staging.replace(manifest)
    return manifest
