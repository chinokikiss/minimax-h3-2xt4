"""Eight immutable TP2 INT8 expert sets, shared base auxiliary, bounded baking."""
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import hashlib
from importlib.metadata import version
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import shutil
import threading

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from patches.tp2_bundle import BundleReader, DIT_LINEAR, sha, signature, split_matrix
from patches.tp2_disk_bake import DiskBakedReader, _LayerSource, _persist_layer, atomic_json, state_proof
from patches.windowed_tp import CheckpointReader
from .checkpoint import PTDCheckpoint, HopStore, RELEASE
from .schedule import hop_plans, hop_sigmas

FORMAT = 'h3-ptd-tp2-bank-v1'


def _layer_records(root, index, recipe):
    records = {}
    for rank in (0, 1):
        stem = f'dit.block{index:02d}.rank{rank}'
        file, manifest = root / (stem + '.safetensors'), root / (stem + '.json')
        if not file.exists() or not manifest.exists():
            return None
        proof = json.loads(manifest.read_text(encoding='utf-8'))
        if proof['recipe_sha256'] != recipe or sha(file) != proof['file_sha256']:
            raise ValueError('Existing PTD partial shard failed admission')
        records[stem] = dict(state_sha256=proof['state_sha256'], base_state_sha256=proof['base_state_sha256'],
            file_sha256=proof['file_sha256'], manifest_sha256=sha(manifest), bytes=file.stat().st_size)
    return records


def _bake_partition(base_path, checkpoint_path, output, layers, device, recipe_sha):
    """One complete layer per GPU; base read once, then eight bounded hop merges."""
    from patches.offline_lora import merged_weight
    from patches.offline_lora_reference import verify_weight
    torch.cuda.set_device(device)
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    base = BundleReader(base_path)
    checkpoint = PTDCheckpoint(checkpoint_path, base)
    stores = [HopStore(checkpoint, hop) for hop in range(checkpoint.hops)]
    result = [dict(shards={}, verified_targets=[]) for _ in stores]
    output = Path(output)
    pending = None
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix='ptd-durable-write') as writer:
        for index in layers:
            states = []
            for rank in (0, 1):
                reader = base._shard(index, rank)
                state = {key: reader.read_tensor(key) for key in reader.header if key != '__metadata__'}
                if state_proof(state) != base.plan['blocks'][f'dit.block{index:02d}.rank{rank}']['state_sha256']:
                    raise ValueError('PTD base layer tensor proof mismatch')
                states.append(state)
            snapshot = _LayerSource(base, index, states)
            for hop, store in enumerate(stores):
                targets = sorted(t for t in store.targets if t.startswith(f'blocks.{index}.'))
                destination = output / f'hop_{hop:02d}'
                old = _layer_records(destination, index, recipe_sha)
                if old is not None:
                    result[hop]['shards'].update(old)
                    result[hop]['verified_targets'].extend(targets)
                    continue
                merged = [dict(state) for state in states]
                for target in targets:
                    q, scale = merged_weight(snapshot, store, target, device)
                    verify_weight(snapshot, store, target, q, scale, device)
                    snapshot.cache.clear()
                    local = DIT_LINEAR[target.split('.', 2)[2]]
                    for rank in (0, 1):
                        merged[rank][local + '.weight'] = split_matrix(q, local, rank)
                        if scale is not None:
                            merged[rank][local + '.weight_scale'] = split_matrix(scale, local, rank, scale=True)
                    del q, scale
                    result[hop]['verified_targets'].append(target)
                if pending is not None:
                    previous, future = pending
                    result[previous]['shards'].update(future.result())
                pending = (hop, writer.submit(_persist_layer, merged, index, targets, destination,
                                              base.plan['blocks'], recipe_sha))
                del merged
                print('PTD_LAYER_QUEUED', device, index, hop, flush=True)
            del snapshot, states, state
        if pending is not None:
            previous, future = pending
            result[previous]['shards'].update(future.result())
    checkpoint.check()
    base.check()
    return result


def _auxiliary(base, checkpoint, hop, output, device, video_shift, audio_shift):
    """Only adapted refiner weights and two effective PDD heads are duplicated."""
    from patches.offline_lora import merged_weight
    from patches.offline_lora_reference import verify_weight
    store = HopStore(checkpoint, hop)
    targets = sorted(t for t in store.targets if t.startswith('token_refiner.'))
    values = {}
    for target in targets:
        weight, scale = merged_weight(base, store, target, device)
        verify_weight(base, store, target, weight, scale, device)
        if scale is not None:
            raise ValueError('PTD refiner must retain the native floating-point weight format')
        values[target + '.weight'] = weight
    plans = hop_plans(checkpoint.hops, checkpoint.steps, video_shift, audio_shift)[hop]
    for prefix, native, plan in zip(('proj_out', 'audio_proj_out'),
                                   ('final_layer.video_out', 'final_layer.audio_out'), plans):
        # Match the provided head bank's FP32 output boundary. Keep bias;
        # these are full heads, not offsets from the original native head.
        with safe_open(str(checkpoint.path), framework='pt', device='cpu') as handle:
            weight = handle.get_tensor(prefix + '.weight').to(device, dtype=torch.float32)
            bias = handle.get_tensor(prefix + '.bias').to(device, dtype=torch.float32)
        p = plan.to(device=device, dtype=torch.float32)
        values[native + '.weight'] = torch.einsum('pn,noi->poi', p, weight).flatten(0, 1).cpu().contiguous()
        values[native + '.bias'] = torch.einsum('pn,no->po', p, bias).flatten().cpu().contiguous()
        del weight, bias
    for name, value in values.items():
        if list(value.shape) != base.header[name]['shape'] or not torch.isfinite(value).all():
            raise ValueError('PTD auxiliary shape / finiteness mismatch: ' + name)
    path = Path(output) / 'aux_overlay.safetensors'
    save_file(values, str(path))
    with path.open('r+b') as stream:
        os.fsync(stream.fileno())
    result = dict(file=path.name, sha256=sha(path), bytes=path.stat().st_size,
                  state_sha256=state_proof(values), verified_targets=targets)
    return result


def disk_plan(base_descriptor, output, layers=50, include_download=False, reserve=4 * 2**30):
    base = BundleReader(base_descriptor)
    if base.kind != 'dit' or not 1 <= layers <= 50:
        raise ValueError('PTD baking requires the DiT base bundle and 1-50 layers')
    path = Path(output).resolve()
    parent = path
    while not parent.exists():
        parent = parent.parent
    blocks = math.ceil(1.02 * sum(info['data_offsets'][1]-info['data_offsets'][0]
        for key,info in base.header.items() if key.startswith('blocks.') and int(key.split('.')[1]) < layers))
    # Each hop gets the selected DiT shards plus refiner/head overlays, while
    # unmodified embeddings/norms/AdaLN remain in the shared base auxiliary.
    overlay = sum(math.prod(info['shape']) * 4 for key, info in base.header.items()
                  if key != '__metadata__' and key.endswith(('.weight', '.bias'))
                  and (key.startswith('token_refiner.blocks.') or key.startswith('final_layer.video_out.')
                       or key.startswith('final_layer.audio_out.')))
    needed = 8 * (blocks + overlay) + reserve + (RELEASE['bytes'] if include_download else 0)
    return dict(required_free_bytes=needed, free_bytes=shutil.disk_usage(parent).free,
                estimated_expert_bytes=8 * (blocks + overlay), shared_base_auxiliary=True,
                output=str(path), layers=layers, full_model=layers == 50)


def bake(base_descriptor, checkpoint_path, output='/tmp/h3_ptd_refasymopd', *,
         devices=('cuda:0', 'cuda:1'), layers=50, video_shift=12., audio_shift=3., allow_pruned_adaln=False):
    """Atomic eight-hop publication, resumable durable shards; no online LoRA."""
    if (len(devices) != 2 or len({torch.device(d).index for d in devices}) != 2
            or any(torch.cuda.get_device_capability(d) != (7, 5)
                   or 'T4' not in torch.cuda.get_device_name(d) for d in devices)):
        raise RuntimeError('PTD offline baking is restricted to two physical T4 devices')
    output = Path(output).resolve()
    base = BundleReader(base_descriptor)
    if 'adaln_t_table' in base.header and not allow_pruned_adaln:
        raise ValueError('Baking the current pruned base requires explicit allow_pruned_adaln approximation consent')
    checkpoint = PTDCheckpoint(checkpoint_path, base)
    if not 1 <= layers <= 50 or checkpoint.blocks < layers or (layers == 50 and checkpoint.blocks != 50):
        raise ValueError('Production requires all 50 adapted blocks; subsets are qualification-only')
    refiner_indices = {int(k.split('.')[2]) for k in base.header if k.startswith('token_refiner.blocks.')}
    if refiner_indices != set(range(checkpoint.refiners)):
        raise ValueError('PTD checkpoint must adapt every Native token-refiner block')
    if (video_shift, audio_shift) != (12., 3.):
        raise ValueError('The released PTD bank requires the trained video/audio shifts 12/3')
    recipe = dict(format=FORMAT, base=str(base.path), base_sha256=sha(base.path),
        base_bundle_sha256=sha(base.bundle_file), checkpoint_sha256=sha(checkpoint.path),
        checkpoint_bytes=checkpoint.path.stat().st_size, layers=layers, hops=checkpoint.hops,
        steps=checkpoint.steps, video_shift=float(video_shift), audio_shift=float(audio_shift),
        pruned_adaln_approximation='adaln_t_table' in base.header,
        source_sha256={name:sha(Path(__file__).with_name(name)) for name in ('bank.py','checkpoint.py','mapping.py','schedule.py')},
        native_merge_sha256=sha(Path(__file__).parents[1] / 'offline_lora.py'),
        native_reference_sha256=sha(Path(__file__).parents[1] / 'offline_lora_reference.py'),
        kitchen=version('comfy-kitchen'), torch=str(torch.__version__))
    digest = hashlib.sha256(json.dumps(recipe, sort_keys=True).encode()).hexdigest()
    if output.exists():
        bank = PTDBank(output / 'bank.json', allow_fixture=layers != 50)
        if bank.recipe_sha != digest:
            raise ValueError('Cached PTD bank belongs to another base / checkpoint / bake recipe')
        return output / 'bank.json'
    plan = disk_plan(base.path, output, layers)
    if plan['free_bytes'] < plan['required_free_bytes']:
        raise OSError('Insufficient /tmp disk for eight experts: ' + json.dumps(plan))
    partial = output.with_name(output.name + '.partial')
    partial.mkdir(parents=True, exist_ok=True)
    recipe_path = partial / 'recipe.json'
    if recipe_path.exists() and json.loads(recipe_path.read_text(encoding='utf-8')) != recipe:
        raise ValueError('Existing partial PTD bank has a different recipe; preserve it separately')
    atomic_json(recipe_path, recipe)
    for hop in range(8):
        (partial / f'hop_{hop:02d}').mkdir(exist_ok=True)
    with ProcessPoolExecutor(max_workers=2, mp_context=mp.get_context('spawn')) as pool:
        jobs = [pool.submit(_bake_partition, str(base.path), str(checkpoint.path), str(partial),
                            list(range(rank, layers, 2)), device, digest) for rank, device in enumerate(devices)]
        workers = [job.result() for job in jobs]
    hops = []
    for hop in range(8):
        records = {key:value for worker in workers for key,value in worker[hop]['shards'].items()}
        expected = {f'dit.block{i:02d}.rank{r}' for i in range(layers) for r in (0, 1)}
        if set(records) != expected:
            raise ValueError('Incomplete PTD TP2 expert coverage')
        auxiliary = _auxiliary(base, checkpoint, hop, partial / f'hop_{hop:02d}', devices[0], video_shift, audio_shift)
        targets = [t for worker in workers for t in worker[hop]['verified_targets']] + auxiliary['verified_targets']
        wanted = {t for t in checkpoint.targets if not t.startswith('blocks.') or int(t.split('.')[1]) < layers}
        if set(targets) != wanted or len(targets) != len(wanted):
            raise ValueError('Incomplete or duplicate PTD independent verification')
        receipt = dict(format=FORMAT, status='complete', hop=hop, recipe_sha256=digest,
                       shards=records, auxiliary=auxiliary, native_verified_targets=sorted(targets))
        path = partial / f'hop_{hop:02d}' / 'receipt.json'
        atomic_json(path, receipt)
        hops.append(dict(directory=path.parent.name, receipt_sha256=sha(path)))
        print('PTD_HOP_DURABLE', hop, len(records), flush=True)
    checkpoint.check()
    base.check()
    atomic_json(partial / 'bank.json', dict(format=FORMAT, status='complete', recipe=recipe,
                recipe_sha256=digest, hops=hops, full_model=layers == 50,
                sigmas=hop_sigmas(8, 32, video_shift).tolist(), online_lora=False))
    partial.replace(output)
    PTDBank(output / 'bank.json', allow_fixture=layers != 50)
    return output / 'bank.json'


class PTDBank:
    def __init__(self, path, *, allow_fixture=False, expected_sha256=None,verify_payloads=True):
        if type(verify_payloads) is not bool:raise ValueError('Explicit payload verification boolean required')
        self.verify_payloads=verify_payloads
        self.path = Path(path).resolve()
        from patches.file_metadata import FileChecks
        self.file_checks = FileChecks()
        admitted = self.file_checks.admit(self.path)
        self.signature = signature(self.path)
        raw = self.path.read_bytes()
        self.digest = hashlib.sha256(raw).hexdigest() if verify_payloads else ('metadata', str(self.path), *admitted)
        if not verify_payloads and isinstance(expected_sha256,tuple) and self.digest != expected_sha256:
            raise ValueError('PTD bank metadata differs between parent and worker')
        if verify_payloads and expected_sha256 is not None and self.digest != expected_sha256:
            raise ValueError('PTD bank does not match the admitted manifest digest')
        if signature(self.path) != self.signature:
            raise RuntimeError('PTD manifest changed during admission')
        value = json.loads(raw)
        self.recipe = value['recipe']
        self.recipe_sha = value['recipe_sha256']
        if (value['format'] != FORMAT or value['status'] != 'complete' or len(value['hops']) != 8
                or value['online_lora'] or (not value['full_model'] and not allow_fixture)
                or (verify_payloads and hashlib.sha256(json.dumps(self.recipe, sort_keys=True).encode()).hexdigest() != self.recipe_sha)):
            raise ValueError('Incomplete / unsupported PTD bank; fixture banks cannot be deployed')
        self.layers = int(self.recipe['layers'])
        if not 1 <= self.layers <= 50 or value['full_model'] != (self.layers == 50):
            raise ValueError('PTD full-model marker does not match the layer count')
        self.sigmas = hop_sigmas(8, 32, self.recipe['video_shift'])
        if value['sigmas'] != self.sigmas.tolist() or self.recipe['hops'] != 8:
            raise ValueError('PTD bank schedule mismatch')
        self.base = Path(self.recipe['base'])
        self.file_checks.admit(self.base)
        if verify_payloads and sha(self.base) != self.recipe['base_sha256']:
            raise ValueError('PTD base descriptor changed')
        base = BundleReader(self.base,verify_payloads=verify_payloads)
        if self.recipe['base_bundle_sha256'] != base.bundle_digest:
            raise ValueError('PTD base bundle changed')
        self.signatures = {self.path: self.signature, self.base: signature(self.base),
                           base.bundle_file: signature(base.bundle_file)}
        self.hops = []
        for hop, entry in enumerate(value['hops']):
            directory = self.path.parent / entry['directory']
            if directory.resolve().parent != self.path.parent:
                raise ValueError('PTD expert directory escaped its bank')
            receipt = directory / 'receipt.json'
            self.file_checks.admit(receipt)
            current = signature(receipt)
            raw_receipt = receipt.read_bytes()
            if verify_payloads and hashlib.sha256(raw_receipt).hexdigest() != entry['receipt_sha256']:
                raise ValueError('PTD hop receipt hash mismatch')
            self.signatures[receipt] = current
            record = json.loads(raw_receipt)
            if (record['format'] != FORMAT or record['hop'] != hop or record['status'] != 'complete'
                    or record['recipe_sha256'] != self.recipe_sha):
                raise ValueError('PTD hop belongs to another bank')
            expected = {f'dit.block{i:02d}.rank{r}' for i in range(self.layers) for r in (0,1)}
            if set(record['shards']) != expected:
                raise ValueError('PTD hop has incomplete TP2 coverage')
            if record['auxiliary'].get('format') or not record['auxiliary'].get('file'):
                raise ValueError('PTD requires the original monolithic auxiliary overlay')
            self.hops.append((directory, record))
        self.check()

    def check(self):
        self.file_checks.check()
        if any(signature(path) != expected for path, expected in self.signatures.items()):
            raise RuntimeError('PTD bank changed during sampling')

    def reader(self, hop, *, refiner=True, heads=True, base_reader=None):
        self.check()
        return HopReader(self, hop, refiner=refiner, heads=heads,
                         base_reader=base_reader)


class HopReader(BundleReader):
    """Adapt the validated Native bundle loader; no online remerge on eviction."""
    _shard = DiskBakedReader._shard

    def __init__(self, bank, hop, *, refiner=True, heads=True, base_reader=None):
        if type(hop) is not int or not 0 <= hop < 8:
            raise ValueError('PTD hop must be 0-7')
        if base_reader is None:
            super().__init__(bank.base,verify_payloads=getattr(bank,'verify_payloads',True))
        else:
            if type(base_reader) is not BundleReader or base_reader.path != bank.base:
                raise ValueError('Queued reader needs the already admitted immutable base')
            base_reader.check()
            self.__dict__.update(base_reader.__dict__)
            from patches.file_metadata import FileChecks
            self.file_checks = FileChecks()
            self.file_checks.files = dict(base_reader.file_checks.files)
            self.read_lock, self.prepare_lock = threading.RLock(), threading.Lock()
            self.verified, self.state_verified, self.shard_readers = {}, {}, {}
            self.stats = dict(base_reader.stats)
        self.bank, self.hop = bank, hop
        self.verify_payloads=getattr(bank,'verify_payloads',True)
        self.root, receipt = bank.hops[hop]
        self.recipe_sha256 = bank.recipe_sha
        self.baked_records = receipt['shards']
        self.plan = dict(self.plan, blocks=dict(self.plan['blocks']))
        for stem, row in self.baked_records.items():
            old = self.plan['blocks'][stem]
            if old['state_sha256'] != row['base_state_sha256']:
                raise ValueError('PTD expert base tensor proof mismatch')
            self.plan['blocks'][stem] = dict(old, state_sha256=row['state_sha256'])
        overlay = self.root / receipt['auxiliary']['file']
        if overlay.resolve().parent != self.root:
            raise ValueError('PTD auxiliary overlay hash mismatch')
        self.overlay = CheckpointReader(overlay)
        self.overlay.file_checks.admit(overlay, size=receipt['auxiliary']['bytes'])
        self.overlay_sha = receipt['auxiliary']['sha256']
        self.overlay_verified = False
        self.overlay_lock = threading.Lock()
        self.refiner_enabled, self.heads_enabled = refiner, heads
        self.identity = (*self.identity, FORMAT, bank.recipe_sha, bank.digest, hop, bool(refiner), bool(heads))
        if not self.verify_payloads:self.identity=(*self.identity,'metadata-checked-weights-v1')

    def _check_roots(self):
        super()._check_roots()
        if hasattr(self, 'bank'):
            self.bank.check()
        if hasattr(self, 'overlay') and signature(self.overlay.path) != self.overlay.signature:
            raise RuntimeError('PTD auxiliary overlay changed')
        if hasattr(self, 'overlay'):
            self.overlay.check()

    def read_tensor(self, name, *, rows=None, columns=None):
        self._check_roots()
        enabled = self.refiner_enabled if name.startswith('token_refiner.') else self.heads_enabled
        if enabled and name in self.overlay.header and name != '__metadata__':
            if rows or columns:
                raise ValueError('PTD overlay exposes whole native tensors')
            self._verify_overlay()
            return self.overlay.read_tensor(name)
        return super().read_tensor(name, rows=rows, columns=columns)

    def _verify_overlay(self):
        """Verify once on first consumption; retain signature guards on later reads."""
        with self.overlay_lock:
            if not self.overlay_verified:
                if self.verify_payloads and sha(self.overlay.path) != self.overlay_sha:
                    raise ValueError('PTD auxiliary overlay hash mismatch')
                if signature(self.overlay.path) != self.overlay.signature:
                    raise RuntimeError('PTD overlay changed during validation')
                self.overlay_verified = True
