"""Bake a Native LoRA once from TP2 shards into immutable disk shards.

Each affected full-width matrix is merged/quantized once, then split into both
owners. Whole layers are assigned to separate GPU processes. Inference never
loads the adapters or remerges evicted blocks. Base shards remain immutable.
"""
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import shutil
from types import SimpleNamespace

import torch
from safetensors.torch import save_file
from patches.tp2_bundle import BundleReader, DIT_LINEAR, sha, signature, split_matrix
from patches.rank_local_int8 import state_hashes
from patches.windowed_tp import CheckpointReader

DESCRIPTOR = 'h3-tp2-disk-baked-v1'
SHARD = 'h3-tp2-disk-baked-shard-v1'


def atomic_json(path, value):
    path = Path(path)
    partial = path.with_name(path.name + '.partial')
    with partial.open('w', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    partial.replace(path)


def state_proof(state):
    return state_hashes(SimpleNamespace(state_dict=lambda: state))


class DiskBakedReader(BundleReader):
    """Base provenance plus hash-bound disk replacements, without online LoRA."""
    def __init__(self, path, descriptor=None,*,verify_payloads=True):
        path = Path(path).resolve()
        value = descriptor or json.loads(path.read_text())
        if value.get('format') != DESCRIPTOR:
            raise ValueError('Unknown disk-baked descriptor')
        base = (path.parent / value['base_descriptor']).resolve()
        receipt = (path.parent / value['receipt']).resolve()
        if sha(base) != value['base_descriptor_sha256'] or sha(receipt) != value['receipt_sha256']:
            raise ValueError('Disk-baked source/receipt checksum mismatch')
        super().__init__(base,verify_payloads=verify_payloads)
        if self.kind != 'dit':
            raise ValueError('Only DiT receives offline LoRA')
        record = json.loads(receipt.read_text())
        if (record.get('status') != 'complete' or record.get('format') != DESCRIPTOR
                or record['recipe']['base_descriptor_sha256'] != value['base_descriptor_sha256']
                or record['recipe']['base_bundle_sha256'] != sha(self.bundle_file)
                or record['recipe']['base_source'] != json.loads(base.read_text())['source']
                or not record.get('native_independent_verification')):
            raise ValueError('Incomplete or mismatched disk bake')
        self.disk_path, self.disk_receipt = path, receipt
        self.disk_signatures = (signature(path), signature(receipt))
        self.recipe_sha256 = record['recipe_sha256']
        if hashlib.sha256(json.dumps(record['recipe'], sort_keys=True).encode()).hexdigest() != self.recipe_sha256:
            raise ValueError('Disk bake recipe checksum mismatch')
        expected = {f'dit.block{i:02d}.rank{r}' for i in range(50) for r in (0, 1)}
        if set(record['shards']) != expected:
            raise ValueError('Incomplete baked DiT layer coverage')
        self.root = receipt.parent
        self.baked_records = record['shards']
        self.plan = dict(self.plan, blocks=dict(self.plan['blocks']))
        for stem, row in self.baked_records.items():
            original = self.plan['blocks'][stem]
            if row['base_state_sha256'] != original['state_sha256']:
                raise ValueError('Baked shard base proof mismatch')
            self.plan['blocks'][stem] = dict(original, state_sha256=row['state_sha256'])
        auxiliary = self.root / 'dit_aux.safetensors'
        if self.verify_payloads and sha(auxiliary) != record['auxiliary_sha256']:
            raise ValueError('Baked auxiliary checksum mismatch')
        self.aux = CheckpointReader(auxiliary)
        self.identity = (*self.identity, DESCRIPTOR, self.recipe_sha256, value['receipt_sha256'])
        self.stats.update(persistent_baked_bytes=record['persistent_baked_bytes'], merge_device='none',
                          disk_baked=True, verified_native_targets=record['verified_native_targets'])

    def _check_roots(self):
        super()._check_roots()
        if (signature(self.disk_path), signature(self.disk_receipt)) != self.disk_signatures:
            raise RuntimeError('Disk-baked descriptor/receipt changed')

    def set_offline_lora(self, stack):
        if stack:
            raise ValueError('Disk-baked weights must not apply LoRA twice')

    def _shard(self, index, rank):
        self._check_roots()
        stem = f'dit.block{index:02d}.rank{rank}'
        row = self.baked_records[stem]
        path, manifest = self.root / (stem + '.safetensors'), self.root / (stem + '.json')
        before = (signature(path), signature(manifest))
        for artifact, current in zip((path, manifest), before):
            if str(artifact) in self.verified and self.verified[str(artifact)] != current:
                raise RuntimeError('Verified disk-baked shard changed')
        if str(manifest) not in self.verified:
            if sha(manifest) != row['manifest_sha256']:
                raise ValueError('Baked shard manifest checksum mismatch')
            proof = json.loads(manifest.read_text())
            if (proof['format'] != SHARD or proof['recipe_sha256'] != self.recipe_sha256
                    or proof['state_sha256'] != row['state_sha256']
                    or proof['base_state_sha256'] != row['base_state_sha256']
                    or proof['file_sha256'] != row['file_sha256']
                    or (self.verify_payloads and sha(path) != row['file_sha256'])):
                raise ValueError('Baked shard payload/proof mismatch')
            if before != (signature(path), signature(manifest)):
                raise RuntimeError('Baked shard changed during validation')
            self.verified[str(path)], self.verified[str(manifest)] = before
        if stem not in self.shard_readers:
            self.shard_readers[stem] = CheckpointReader(path)
        return self.shard_readers[stem]


def _persist_layer(states,index,targets,output,plan,recipe_sha):
    """Own immutable CPU states until both payloads and manifests are durable."""
    records={}
    for rank,state in enumerate(states):
        stem=f'dit.block{index:02d}.rank{rank}'
        path=output/(stem+'.safetensors');partial=path.with_name(path.name+'.partial')
        save_file(state,str(partial))
        with partial.open('r+b') as stream:os.fsync(stream.fileno())
        partial.replace(path)
        proof=dict(format=SHARD,kind='dit',index=index,rank=rank,recipe_sha256=recipe_sha,
            state_sha256=state_proof(state),base_state_sha256=plan[stem]['state_sha256'],
            file_sha256=sha(path),native_verified_targets=targets)
        manifest=output/(stem+'.json');atomic_json(manifest,proof)
        records[stem]=dict(state_sha256=proof['state_sha256'],base_state_sha256=proof['base_state_sha256'],
            file_sha256=proof['file_sha256'],manifest_sha256=sha(manifest),bytes=path.stat().st_size)
    print('TP2_DISK_LAYER_DURABLE',index,flush=True)
    return records


class _LayerSource:
    """One admitted layer; production/reference get independent tensor copies."""
    def __init__(self,reader,index,states):
        self.reader,self.index,self.states=reader,index,states
        self.header=reader.header;self.cache={}

    def read_tensor(self,name):
        target=self.reader._target(name)
        if target is None or target[0]!=self.index:return self.reader.read_tensor(name)
        if name not in self.cache:
            _,local,suffix=target
            if suffix=='comfy_quant':return self.reader.read_tensor(name)
            from patches.tp2_bundle import join_matrix
            value=join_matrix(self.states[0][local+'.'+suffix],self.states[1][local+'.'+suffix],
                              local,scale=suffix=='weight_scale')
            self.cache[name]=value.to(CheckpointReader.DTYPES[self.header[name]['dtype']])
        return self.cache[name].clone()


def _bake_layers(base_path, stack, output, layers, device, recipe_sha, *, pipeline=True):
    from patches.h3_lora import H3LoRAStore
    from patches.offline_lora import merged_weight
    from patches.offline_lora_reference import verify_weight
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    if torch.device(device).type == 'cuda':
        torch.cuda.set_device(device)
    reader = BundleReader(base_path)
    store = H3LoRAStore(tuple(stack), reader)
    active = {t for _, _, mapping in store.sources for t, (_, _, strength) in mapping.items() if strength}
    output = Path(output)
    records, verified = {}, []
    writer=ThreadPoolExecutor(max_workers=1,thread_name_prefix='h3-bake-write') if pipeline else None
    pending=None
    try:
        for index in layers:
            targets = [f'blocks.{index}.{name}' for name in DIT_LINEAR if f'blocks.{index}.{name}' in active]
            states = []
            for rank in (0, 1):
                source = reader._shard(index, rank)
                state = {name: source.read_tensor(name) for name in source.header if name != '__metadata__'}
                expected = reader.plan['blocks'][f'dit.block{index:02d}.rank{rank}']['state_sha256']
                if state_proof(state) != expected:
                    raise ValueError('Base shard differs from Native state proof')
                states.append(state)
            snapshot=_LayerSource(reader,index,states) if pipeline else reader
            for target in targets:
                q, scale = merged_weight(snapshot, store, target, device)
                verify_weight(snapshot, store, target, q, scale, device)
                if pipeline:snapshot.cache.clear()
                local = DIT_LINEAR[target.split('.', 2)[2]]
                for rank in (0, 1):
                    states[rank][local + '.weight'] = split_matrix(q, local, rank)
                    if scale is not None:
                        states[rank][local + '.weight_scale'] = split_matrix(scale, local, rank, scale=True)
                del q, scale
                verified.append(target)
                print('TP2_DISK_BAKED', device, target, flush=True)
            del snapshot
            if writer is None:
                records.update(_persist_layer(states,index,targets,output,reader.plan['blocks'],recipe_sha))
            else:
                # One outstanding layer plus the next in production: bounded ownership.
                if pending is not None:records.update(pending.result())
                pending=writer.submit(_persist_layer,states,index,targets,output,reader.plan['blocks'],recipe_sha)
            del states, state
            print('TP2_DISK_LAYER_QUEUED' if writer else 'TP2_DISK_LAYER_COMPLETE', device, index, flush=True)
        if pending is not None:records.update(pending.result())
    finally:
        if writer is not None:writer.shutdown(wait=True,cancel_futures=True)
    store.check()
    reader.check()
    return dict(shards=records, verified_targets=verified)


def _bake_layers_serial(*args):
    return _bake_layers(*args,pipeline=False)


def _write_auxiliary(reader, store, targets, path, device):
    from patches.offline_lora import merged_weight
    from patches.offline_lora_reference import verify_weight
    shutil.copyfile(reader.aux.path, path)
    destination = CheckpointReader(path)
    with path.open('r+b', buffering=0) as stream:
        for target in targets:
            value, scale = merged_weight(reader, store, target, device)
            verify_weight(reader, store, target, value, scale, device)
            if scale is not None:
                raise ValueError('Unexpected quantized auxiliary LoRA target')
            info = destination.header[target + '.weight']
            raw = memoryview(value.reshape(-1).view(torch.uint8).numpy()).cast('B')
            if info['shape'] != list(value.shape) or len(raw) != info['data_offsets'][1] - info['data_offsets'][0]:
                raise ValueError('Auxiliary serialization schema changed')
            stream.seek(destination.offset + info['data_offsets'][0])
            while raw:
                count = stream.write(raw)
                if not count:
                    raise IOError('Incomplete auxiliary write')
                raw = raw[count:]
            del value, raw
            print('TP2_DISK_BAKED_AUX', device, target, flush=True)
        os.fsync(stream.fileno())


def bake(base_descriptor, lora_stack, output, *, devices=('cuda:0', 'cuda:1'), pipeline=True):
    """Atomic publication; persistent files contain weights, never adapters."""
    from patches.h3_lora import H3LoRAStore
    from importlib.metadata import version
    base = BundleReader(base_descriptor)
    if base.kind != 'dit' or not lora_stack or not 1 <= len(devices) <= 2:
        raise ValueError('DiT base, nonempty LoRA stack and one/two devices required')
    store = H3LoRAStore(tuple(lora_stack), base)
    recipe = dict(format=DESCRIPTOR, base_descriptor_sha256=sha(base.path),
        base_bundle_sha256=sha(base.bundle_file), base_source=json.loads(base.path.read_text())['source'],
        native_environment=dict(torch=str(torch.__version__), kitchen=version('comfy-kitchen'),
            devices=[list(torch.cuda.get_device_capability(d)) if torch.device(d).type=='cuda' else 'cpu'
                     for d in devices],
            source_sha256={name:sha(Path(__file__).with_name(name))
                for name in ('offline_lora.py', 'offline_lora_reference.py')}),
        loras=[dict(bytes=Path(p).stat().st_size, sha256=sha(p), strength=float(s)) for p, s in lora_stack])
    recipe_sha = hashlib.sha256(json.dumps(recipe, sort_keys=True).encode()).hexdigest()
    output = Path(output).resolve()
    if output.exists():
        reader = DiskBakedReader(output / 'dit_baked.json')
        if reader.recipe_sha256 != recipe_sha:
            raise ValueError('Cached disk bake has a different base/LoRA recipe')
        reader.check()
        return output / 'dit_baked.json'
    partial = output.with_name(output.name + '.partial')
    partial.mkdir(parents=True, exist_ok=False)
    active = {t for _, _, mapping in store.sources for t, (_, _, strength) in mapping.items() if strength}
    shards, verified = {}, []
    partitions = [list(range(i, 50, len(devices))) for i in range(len(devices))]
    worker=_bake_layers if pipeline else _bake_layers_serial
    if len(devices) == 1:
        results = [worker(str(base.path), tuple(lora_stack), str(partial), partitions[0], devices[0], recipe_sha)]
    else:
        with ProcessPoolExecutor(max_workers=len(devices), mp_context=mp.get_context('spawn')) as pool:
            jobs = [pool.submit(worker, str(base.path), tuple(lora_stack), str(partial), layers, device, recipe_sha)
                    for layers, device in zip(partitions, devices)]
            results = [job.result() for job in jobs]
    for result in results:
        shards.update(result['shards'])
        verified.extend(result['verified_targets'])
    auxiliary_targets = sorted(t for t in active if not t.startswith('blocks.'))
    _write_auxiliary(base, store, auxiliary_targets, partial / 'dit_aux.safetensors', devices[0])
    verified.extend(auxiliary_targets)
    if set(verified) != active or len(verified) != len(active):
        raise ValueError('Incomplete/doubled Native target verification')
    store.check()
    base.check()
    persistent = sum(row['bytes'] for row in shards.values()) + (partial / 'dit_aux.safetensors').stat().st_size
    receipt = dict(format=DESCRIPTOR, status='complete', recipe=recipe, recipe_sha256=recipe_sha,
        shards=shards, auxiliary_sha256=sha(partial / 'dit_aux.safetensors'),
        native_independent_verification=True, verified_native_targets=len(verified),
        verified_targets=sorted(verified), devices=list(devices), persistent_baked_bytes=persistent,
        online_lora=False, cpu_remerge_during_inference=False)
    atomic_json(partial / 'bake_receipt.json', receipt)
    atomic_json(partial / 'dit_baked.json', dict(format=DESCRIPTOR,
        base_descriptor=os.path.relpath(base.path, partial), base_descriptor_sha256=sha(base.path),
        receipt='bake_receipt.json', receipt_sha256=sha(partial / 'bake_receipt.json')))
    partial.replace(output)
    DiskBakedReader(output / 'dit_baked.json').check()
    return output / 'dit_baked.json'


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--lora', nargs=2, action='append', required=True, metavar=('FILE', 'STRENGTH'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--devices', nargs='+', default=['cuda:0', 'cuda:1'])
    args = parser.parse_args()
    print(bake(args.base, tuple((p, float(s)) for p, s in args.lora), args.output, devices=tuple(args.devices)), flush=True)
