"""Opt-in self-contained PTD TP2 Bank v2.

This reader never opens the original DiT base descriptor. All payload SHA256
proofs belong to the offline publisher; production admission uses metadata and
the native tensor/rank schema. The monolithic v1 Bank and its runtime admission have been retired.
"""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
import re
import threading
import time
from pathlib import Path

import torch

from patches.file_metadata import FileChecks, file_metadata
from patches.rank_local_int8 import check_state_schema, shell, state_hashes
from patches.tp2_bundle import BundleReader, sha, signature
from patches.tp2_disk_bake import SHARD
from patches.windowed_tp import CheckpointReader
from .schedule import hop_sigmas
from .http_range2 import HTTPRange2TensorReader

FORMAT_V2 = 'h3-ptd-tp2-bank-v2'
_PART_ID = re.compile(r'^[a-zA-Z0-9][a-zA-Z0-9_.-]{0,79}$')
_STEM = re.compile(r'^dit\.block(\d{2})\.rank([01])$')


def _relative(value):
    """All files inside a part must be ordinary, relative POSIX paths."""
    if not isinstance(value, str) or not value or '\\' in value:
        raise ValueError('Invalid Bank v2 part-relative path')
    path = Path(value)
    if path.is_absolute() or any(piece in ('', '.', '..') for piece in value.split('/')):
        raise ValueError('Bank v2 path escapes its immutable part')
    return path


def _checked_file(root, relative):
    path = (root / _relative(relative)).resolve()
    if not path.is_relative_to(root):
        raise ValueError('Bank v2 payload escaped its part root')
    return path


def _retire_page_cache(path):
    """Advisory Linux cache retirement after all rank reads have completed."""
    if not hasattr(os, 'posix_fadvise') or not hasattr(os, 'POSIX_FADV_DONTNEED'):
        return 'unavailable'
    try:
        fd = os.open(path, os.O_RDONLY)
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(fd)
    except OSError:
        return 'error'
    return 'hinted'


def _same_schema(first, second):
    return (first.get('dtype') == second.get('dtype')
            and first.get('shape') == second.get('shape'))


def _http4_stage_file(hop, name, expected_bytes):
    """Experiment-only: wait for an atomically published HTTP cache file.

    The Bank inventory and metadata checks remain on the immutable mounted
    part. Only the tensor payload reader uses this separate completed file.
    """
    root = os.getenv('H3_PTD_HTTP4_PIPELINE_ROOT')
    if not root or hop not in (0, 1):
        return None
    path = Path(root) / 'parts' / f'hop_{hop:02d}' / name
    ready = path.with_name(path.name + '.ready')
    metrics = os.getenv('H3_PTD_HTTP4_PIPELINE_METRICS')
    deadline = time.monotonic() + 300.0
    while time.monotonic() < deadline:
        if ready.is_file():
            stat = file_metadata(path)
            if stat[0] != expected_bytes:
                raise RuntimeError('Incomplete HTTP-staged PTD payload: ' + name)
            return path
        if metrics and Path(metrics).exists():
            try:
                report = json.loads(Path(metrics).read_text(encoding='utf-8'))
            except (OSError, ValueError):
                report = {}
            if report.get('status') in ('failed', 'blocked_insufficient_scratch'):
                raise RuntimeError('PTD HTTP4 stage failed: ' + str(report.get('error_message', report['status'])))
        time.sleep(0.1)
    raise TimeoutError('PTD HTTP4 payload readiness exceeded 300s: ' + name)


class PTDBankV2:
    """Logical hop/layer/rank mapping over independently mounted, pinned parts."""

    def __init__(self, path, *, part_roots=(), allow_fixture=False,
                 expected_sha256=None, verify_payloads=False):
        if type(verify_payloads) is not bool:
            raise ValueError('Explicit payload verification boolean required')
        self.path = Path(path).resolve()
        self.io_workers = 1
        self.retire_page_cache = False
        self._io_pool_lock = threading.Lock()
        self._io_pool = None
        self._io_pool_workers = 0
        self.verify_payloads = verify_payloads
        self.http_range2_enabled = os.getenv('H3_PTD_HTTP_RANGE2_HOSTCACHE', '0') == '1'
        # Transport timing is opt-in diagnostic work; the inference default stays quiet.
        self.http_range2_debug = os.getenv('H3_PTD_HTTP_RANGE2_DEBUG', '0') == '1'
        if self.http_range2_enabled and (verify_payloads or os.getenv('H3_PTD_HTTP4_PIPELINE_ROOT')):
            raise ValueError('HTTP Range2 host-cache experiment requires no payload rehash or HTTP4 staging')
        self._http2_lock = threading.Lock()
        self._http2_stats = dict(logical_bytes=0, transferred_bytes=0, requests=0,
                                 transport_seconds=0., tensor_ready_seconds=0., batches=0)
        self._http2_parts = {}
        self._http2_rates = {}
        self.file_checks = FileChecks()
        raw = self.file_checks.read_json(self.path)
        admitted = self.file_checks.admit(self.path)
        self.digest = (hashlib.sha256(self.path.read_bytes()).hexdigest() if verify_payloads
                       else ('metadata', str(self.path), *admitted))
        if expected_sha256 is not None and expected_sha256 != self.digest:
            raise ValueError('Bank v2 manifest identity differs from the admitted session')
        if (raw.get('format') != FORMAT_V2 or raw.get('status') != 'complete'
                or raw.get('online_lora') is not False or len(raw.get('hops', [])) != 8):
            raise ValueError('Incomplete or unsupported PTD Bank v2')
        self.recipe = raw['recipe']
        self.recipe_sha = raw['recipe_sha256']
        if (verify_payloads and hashlib.sha256(
                json.dumps(self.recipe, sort_keys=True).encode()).hexdigest() != self.recipe_sha):
            raise ValueError('Bank v2 recipe proof mismatch')
        self.layers = self.recipe['layers']
        if (type(self.layers) is not int or not 1 <= self.layers <= 50
                or raw.get('full_model') != (self.layers == 50)
                or (self.layers != 50 and not allow_fixture)
                or self.recipe.get('hops') != 8 or self.recipe.get('steps') != 32
                or (self.recipe.get('video_shift'), self.recipe.get('audio_shift')) != (12., 3.)):
            raise ValueError('Unqualified Bank v2 layer count, approximation or schedule')
        self.sigmas = hop_sigmas(8, 32, self.recipe['video_shift'])
        if raw['sigmas'] != self.sigmas.tolist():
            raise ValueError('PTD Bank v2 sigma schedule mismatch')

        mounts = dict(part_roots)
        parts = raw['parts']
        if len(parts) != len(set(parts)) or not parts or set(mounts) - set(parts):
            raise ValueError('Bank v2 part IDs are missing, duplicated or unexpected')
        self.parts = {}
        self.part_identities = {key: value['identity'] for key,value in parts.items()}
        for part_id, spec in parts.items():
            if not _PART_ID.fullmatch(part_id) or not isinstance(spec, dict):
                raise ValueError('Invalid Bank v2 part identity')
            if not isinstance(spec.get('identity'), str) or not spec['identity']:
                raise ValueError('Immutable Bank v2 part identity is required')
            if part_id in mounts:
                root = Path(mounts[part_id]).resolve()
            else:
                root = (self.path.parent / _relative(spec['root'])).resolve() if spec['root'] != '.' else self.path.parent
                if not root.is_relative_to(self.path.parent):
                    raise ValueError('Bank v2 local part root escaped the manifest directory')
            if not root.is_dir():
                raise FileNotFoundError('Bank v2 part not mounted: ' + part_id)
            self.parts[part_id] = root
        self.shared_spec = raw['shared']
        self.header_path = self.resolve(self.shared_spec['part'], self.shared_spec['native_header'])
        self.plan_path = self.resolve(self.shared_spec['part'], self.shared_spec['block_plan'])
        self.shared_path = self.resolve(self.shared_spec['part'], self.shared_spec['aux'])
        self.header = self.file_checks.read_json(self.header_path)
        block_plan = self.file_checks.read_json(self.plan_path)
        self.plan = {'blocks': block_plan['blocks']}
        self.shared = CheckpointReader(self.shared_path)
        self.file_checks.admit(self.shared_path, size=self.shared_spec['aux_bytes'])
        if verify_payloads and sha(self.shared_path) != self.shared_spec['aux_sha256']:
            raise ValueError('Bank v2 shared auxiliary proof mismatch')
        expected = {f'dit.block{i:02d}.rank{rank}' for i in range(self.layers) for rank in (0, 1)}
        if set(self.plan['blocks']) != expected:
            raise ValueError('Bank v2 base rank-state plan is incomplete')
        static_names = set(self.shared.header) - {'__metadata__'}
        all_aux = {name for name in self.header
                   if name != '__metadata__' and not name.startswith('blocks.')}
        if not static_names <= all_aux:
            raise ValueError('Bank v2 shared auxiliary contains unknown tensors')
        for name in static_names:
            if not _same_schema(self.header[name], self.shared.header[name]):
                raise ValueError('Bank v2 shared auxiliary schema mismatch: ' + name)
        for name in self.header:
            if name == '__metadata__' or not name.startswith('blocks.'):
                continue
            if int(name.split('.')[1]) >= self.layers:
                raise ValueError('Bank v2 fixture header contains undeployed layers')

        source_recipe_expected = self.recipe.get('source_bank_recipe_sha256', self.recipe_sha)
        if not isinstance(source_recipe_expected, str) or not source_recipe_expected:
            raise ValueError('Bank v2 has no immutable source recipe identity')
        self.hops = []
        self.overlay_paths = []
        self.shard_paths = []
        for hop, record in enumerate(raw['hops']):
            if (record.get('hop') != hop or record.get('status') != 'complete'
                    or record.get('source_recipe_sha256') != source_recipe_expected
                    or set(record.get('shards', {})) != expected):
                raise ValueError('Bank v2 hop coverage/identity mismatch')
            aux = record['auxiliary']
            overlay_path = self.resolve(aux['part'], aux['file'])
            self.file_checks.admit(overlay_path, size=aux['bytes'])
            overlay = CheckpointReader(overlay_path)
            names = set(overlay.header) - {'__metadata__'}
            if not names or names & static_names or names | static_names != all_aux:
                raise ValueError('Bank v2 static/overlay tensor coverage mismatch')
            for name in names:
                if not _same_schema(self.header[name], overlay.header[name]):
                    raise ValueError('Bank v2 overlay tensor schema mismatch: ' + name)
            if verify_payloads and sha(overlay_path) != aux['sha256']:
                raise ValueError('Bank v2 hop overlay proof mismatch')
            paths = {}
            owned_payloads = set()
            for stem, row in record['shards'].items():
                match = _STEM.fullmatch(stem)
                if match is None or int(match[1]) >= self.layers:
                    raise ValueError('Invalid Bank v2 logical rank key')
                plan = self.plan['blocks'][stem]
                if (row.get('base_state_sha256') != plan['state_sha256']
                        or not isinstance(row.get('state_sha256'), dict)):
                    raise ValueError('Bank v2 baked/base tensor proof mismatch')
                path = self.resolve(row['part'], row['file'])
                proof = self.resolve(row['part'], row['manifest'])
                if path == proof or path in owned_payloads or proof in owned_payloads:
                    raise ValueError('Bank v2 rank shards must have distinct immutable files')
                owned_payloads.update((path, proof))
                self._admit_inventory(path, row['bytes'])
                self._admit_inventory(proof)
                paths[stem] = (path, proof)
            self.hops.append((self.parts[aux['part']], record))
            self.overlay_paths.append(overlay_path)
            self.shard_paths.append(paths)
        self.file_checks.check()

    def _http2_transfer(self, part, name, logical, transferred, requests,
                        transport_seconds, tensor_ready_seconds):
        if not self.http_range2_debug:return
        # Times cover BOTH concurrent GETs as a batch, never sum thread durations.
        with self._http2_lock:
            row = self._http2_stats
            per_part = self._http2_parts.setdefault(part, dict(logical_bytes=0,
                transferred_bytes=0, requests=0, transport_seconds=0.,
                tensor_ready_seconds=0., batches=0, overlays=0, rank_shards=0))
            for item in (row, per_part):
                item['logical_bytes'] += logical
                item['transferred_bytes'] += transferred
                item['requests'] += requests
                item['transport_seconds'] += transport_seconds
                item['tensor_ready_seconds'] += tensor_ready_seconds
                item['batches'] += 1
            per_part['overlays' if name == 'aux_overlay.safetensors' else 'rank_shards'] += 1
            self._http2_rates.setdefault(part, []).append((
                transferred / max(transport_seconds, 1e-9) / 1e6,
                transferred / max(tensor_ready_seconds, 1e-9) / 1e6))

    def http2_state(self):
        def detailed(row, rates):
            return dict(row,
                http_download_MBps=round(row['transferred_bytes'] / max(row['transport_seconds'], 1e-9) / 1e6, 3),
                tensor_ready_MBps=round(row['transferred_bytes'] / max(row['tensor_ready_seconds'], 1e-9) / 1e6, 3),
                p50_batch_download_MBps=(round(sorted(x[0] for x in rates)[len(rates)//2], 3) if rates else 0.),
                p95_batch_download_MBps=(round(sorted(x[0] for x in rates)[min(len(rates)-1, int(.95*len(rates)))], 3)
                                          if rates else 0.))
        with self._http2_lock:
            rates = [rate for bucket in self._http2_rates.values() for rate in bucket]
            return dict(enabled=self.http_range2_enabled, debug_enabled=self.http_range2_debug,
                **detailed(self._http2_stats, rates),
                per_part={part:detailed(row, self._http2_rates.get(part, []))
                          for part,row in sorted(self._http2_parts.items())})

    def http2_reader(self, path, part, name):
        callback = (lambda logical, transferred, requests, transport_seconds, tensor_ready_seconds:
                    self._http2_transfer(part, name, logical, transferred, requests,
                                         transport_seconds, tensor_ready_seconds)) if self.http_range2_debug else None
        return HTTPRange2TensorReader(path,
            dataset_identity=self.part_identities[part], remote_filename='payload/'+name,
            on_transfer=callback)

    def configure_io(self, io_workers=1, retire_page_cache=False):
        """Opt-in v2-only host I/O; no change to Native kernels or other readers."""
        if type(io_workers) is not int or not 1 <= io_workers <= 8:
            raise ValueError('PTD v2 I/O workers per rank must be 1..8')
        if type(retire_page_cache) is not bool:
            raise ValueError('PTD v2 page-cache retirement must be boolean')
        with self._io_pool_lock:
            stale = self._io_pool if io_workers != self.io_workers else None
            if stale is not None:
                self._io_pool = None
                self._io_pool_workers = 0
            self.io_workers = io_workers
            self.retire_page_cache = retire_page_cache
        if stale is not None:stale.shutdown(wait=True)

    def _io_executor(self):
        """One bounded host-read pool shared by all hops in this rank process."""
        with self._io_pool_lock:
            if self.io_workers == 1:return None
            if self._io_pool is None:
                self._io_pool = ThreadPoolExecutor(max_workers=self.io_workers,
                    thread_name_prefix='ptd-v2-io')
                self._io_pool_workers = self.io_workers
            return self._io_pool

    def close_io(self):
        lock = getattr(self, '_io_pool_lock', None)
        if lock is None:return
        with lock:
            pool = self._io_pool
            self._io_pool = None
            self._io_pool_workers = 0
        if pool is not None:pool.shutdown(wait=True)

    def __del__(self):
        try:self.close_io()
        except Exception:pass

    def _admit_inventory(self, path, size=None):
        stat = file_metadata(path)
        if size is not None and stat[0] != size:
            raise ValueError('Bank v2 shard byte size mismatch: ' + str(path))

    def resolve(self, part, relative):
        if part not in self.parts:
            raise ValueError('Unknown Bank v2 immutable part')
        return _checked_file(self.parts[part], relative)

    def check(self):
        self.file_checks.check()

    def reader(self, hop, *, refiner=True, heads=True, base_reader=None):
        self.check()
        if refiner is not True or heads is not True:
            raise ValueError('Base-free Bank v2 requires baked hop refiners and PDD heads')
        return HopReaderV2(self, hop, base_reader=base_reader)


class HopReaderV2(BundleReader):
    """Native-facing PTD reader; deliberately does not construct BundleReader."""

    def __init__(self, bank, hop, *, base_reader=None):
        if type(hop) is not int or not 0 <= hop < 8:
            raise ValueError('PTD Bank v2 hop must be 0-7')
        if base_reader is not None and (not isinstance(base_reader, HopReaderV2)
                                      or base_reader.bank.path != bank.path):
            raise ValueError('Bank v2 queued reader belongs to another bank')
        self.bank, self.hop = bank, hop
        self.path = bank.path
        self.signature = signature(bank.path)
        self.kind = 'dit'
        self.header = bank.header
        self.root = bank.hops[hop][0]
        self.shared = bank.shared
        aux_record = bank.hops[hop][1]['auxiliary']
        self.overlay = (bank.http2_reader(bank.overlay_paths[hop], aux_record['part'], aux_record['file'])
                        if bank.http_range2_enabled else CheckpointReader(bank.overlay_paths[hop]))
        stage_overlay = _http4_stage_file(hop, bank.hops[hop][1]['auxiliary']['file'],
                                          bank.hops[hop][1]['auxiliary']['bytes'])
        if stage_overlay is not None:
            self.overlay = CheckpointReader(stage_overlay)
        self.overlay_sha = bank.hops[hop][1]['auxiliary']['sha256']
        self.verify_payloads = bank.verify_payloads
        self.baked_records = bank.hops[hop][1]['shards']
        self.plan = {'blocks': {}}
        for stem, original in bank.plan['blocks'].items():
            row = self.baked_records[stem]
            self.plan['blocks'][stem] = dict(original, state_sha256=row['state_sha256'])
        self.recipe_sha256 = bank.recipe_sha
        self.file_checks = FileChecks()
        self.file_checks.admit(self.overlay.path)
        self.read_lock, self.prepare_lock = threading.RLock(), threading.Lock()
        self._overlay_seen = set()
        self._overlay_count = len(self.overlay.header) - int('__metadata__' in self.overlay.header)
        self._overlay_retired = False
        self.verified, self.state_verified, self.shard_readers = {}, {}, {}
        self.logical_read_bytes = self.range_read_calls = 0
        self.offline_targets, self.offline_store = set(), None
        self.lora_store = None
        self.stats = dict(prepared_blocks=0, merged_matrices=0,
                          persistent_baked_bytes=0, merge_device='none', disk_baked=True,
                          io_workers=1, parallel_rank_loads=0, rank_read_ms=0.,
                          page_retire_ms=0., page_retire_hints=0,
                          page_retire_unavailable=0, page_retire_errors=0,
                          overlay_page_retire_hints=0, overlay_page_retire_unavailable=0,
                          overlay_page_retire_errors=0)
        self.identity = ('ptd-bank-v2', bank.digest, bank.recipe_sha, hop, 'metadata-checked-weights-v1')

    def _check_roots(self):
        self.bank.check()
        self.shared.check()
        self.overlay.check()

    def check(self):
        self._check_roots()
        self.file_checks.check()

    def set_offline_lora(self, stack):
        raise ValueError('Self-contained PTD Bank v2 must never receive online/offline runtime LoRA')

    def _shard(self, index, rank):
        if type(index) is not int or not 0 <= index < self.bank.layers or rank not in (0, 1):
            raise ValueError('Bank v2 rank shard outside the baked model')
        self._check_roots()
        stem = f'dit.block{index:02d}.rank{rank}'
        row = self.baked_records[stem]
        path, manifest = self.bank.shard_paths[self.hop][stem]
        stage_shard = _http4_stage_file(self.hop, row['file'], row['bytes'])
        if stage_shard is not None:
            path = stage_shard
        before = (file_metadata(path), file_metadata(manifest))
        self.file_checks.admit(path, size=row['bytes'])
        self.file_checks.admit(manifest)
        for artifact, current in zip((path, manifest), before):
            prior = self.verified.get(str(artifact))
            if prior is not None and prior != current:
                raise RuntimeError('Bank v2 shard changed after admission')
        if str(manifest) not in self.verified:
            if self.verify_payloads and sha(manifest) != row['manifest_sha256']:
                raise ValueError('Bank v2 shard manifest proof mismatch')
            proof = self.file_checks.read_json(manifest)
            if (proof.get('format') != SHARD
                    or proof.get('recipe_sha256') != self.bank.hops[self.hop][1]['source_recipe_sha256']
                    or proof.get('state_sha256') != row['state_sha256']
                    or proof.get('base_state_sha256') != row['base_state_sha256']
                    or proof.get('file_sha256') != row['file_sha256']
                    or (self.verify_payloads and sha(path) != row['file_sha256'])):
                raise ValueError('Bank v2 shard metadata/proof mismatch')
            if before != (file_metadata(path), file_metadata(manifest)):
                raise RuntimeError('Bank v2 shard mutated during admission')
            self.verified[str(path)], self.verified[str(manifest)] = before
        if stem not in self.shard_readers:
            self.shard_readers[stem] = (self.bank.http2_reader(path, row['part'], row['file'])
                                       if self.bank.http_range2_enabled else CheckpointReader(path))
        return self.shard_readers[stem]

    def _raw_tensor(self, name):
        self._check_roots()
        if self._target(name) is None:
            source = self.shared if name in self.shared.header else self.overlay
            value = source.read_tensor(name)
            retire_overlay = False
            with self.read_lock:
                self.logical_read_bytes += value.numel() * value.element_size()
                self.range_read_calls += 1
                if source is self.overlay and self.bank.retire_page_cache and not self._overlay_retired:
                    self._overlay_seen.add(name)
                    if len(self._overlay_seen) == self._overlay_count:
                        self._overlay_retired = retire_overlay = True
            if retire_overlay:
                # Only after every overlay tensor has been materialized; a hint
                # after each tensor would repeatedly evict the same shared file.
                hint = _retire_page_cache(self.overlay.path)
                with self.read_lock:
                    self.stats[{'hinted':'overlay_page_retire_hints',
                                'unavailable':'overlay_page_retire_unavailable',
                                'error':'overlay_page_retire_errors'}[hint]] += 1
            return value
        return BundleReader._raw_tensor(self, name)

    def read_tensor(self, name, *, rows=None, columns=None):
        if rows or columns:
            raise ValueError('Bank v2 exposes whole Native tensors; use rank shards for blocks')
        if name not in self.header:
            raise KeyError(name)
        return self._raw_tensor(name)

    def read_tensors(self, names):
        names = tuple(names)
        if any(name not in self.header or self._target(name) is not None for name in names):
            raise ValueError('Batch PTD HTTP2 reader accepts only full auxiliary tensors')
        shared = tuple(name for name in names if name in self.shared.header)
        overlay = tuple(name for name in names if name not in self.shared.header)
        result = {name: self.shared.read_tensor(name) for name in shared}
        if overlay:
            result.update(self.overlay.read_tensors(overlay) if self.bank.http_range2_enabled
                          else {name:self.overlay.read_tensor(name) for name in overlay})
        with self.read_lock:
            self.logical_read_bytes += sum(v.numel()*v.element_size() for v in result.values())
            self.range_read_calls += len(names)
        return result

    def workspace_bytes(self, index):
        # Existing WeightSpec preparation admission charges this extra I/O
        # transient memory in the v2 path only (8 MiB per additional reader).
        return 8 * 2**20 * (self.bank.io_workers - 1)

    def load_rank(self, index, rank):
        with self.prepare_lock:
            source = self._shard(index, rank)
            module = shell('dit', rank)
            names = tuple(module.state_dict())
            workers = self.bank.io_workers
            read_started = time.perf_counter()
            if self.bank.http_range2_enabled:
                state = source.read_tensors(names)
            elif workers == 1:
                state = {name: source.read_tensor(name) for name in names}
            else:
                # Reuse the bank-owned pool across all 50 blocks and eight hops.
                # Keep each tensor's positional read independent of CUDA/NCCL.
                pool = self.bank._io_executor()
                futures = [pool.submit(source.read_tensor, name) for name in names]
                try:
                    state = {name:future.result() for name,future in zip(names,futures)}
                except BaseException:
                    for future in futures:future.cancel()
                    for future in futures:
                        if future.running():
                            try:future.result()
                            except BaseException:pass
                    raise
            read_ms = (time.perf_counter() - read_started) * 1000.
            stem = f'dit.block{index:02d}.rank{rank}'
            expected = self.plan['blocks'][stem]
            check_state_schema(state, expected['state_sha256'])
            module.load_state_dict(state, strict=True, assign=True)
            proof = (self.verified[str(source.path)],
                     self.verified[str(self.bank.shard_paths[self.hop][stem][1])])
            if self.verify_payloads and self.state_verified.get(stem) != proof:
                if state_hashes(module) != expected['state_sha256']:
                    raise ValueError('Bank v2 rank-state proof mismatch')
                self._shard(index, rank)
                self.state_verified[stem] = proof
            module._rank_checkpoint_metadata = expected['quant_metadata']
            self.check()
            self.stats['prepared_blocks'] += 1
            self.stats['io_workers'] = workers
            self.stats['rank_read_ms'] += read_ms
            self.stats['parallel_rank_loads'] += int(workers > 1)
            if self.bank.retire_page_cache:
                # Read futures have joined; no other tensor of this rank is
                # still using its source file. The hint is never a correctness gate.
                retire_started = time.perf_counter()
                hint = _retire_page_cache(source.path)
                self.stats['page_retire_ms'] += (time.perf_counter() - retire_started) * 1000.
                self.stats[{'hinted': 'page_retire_hints',
                            'unavailable': 'page_retire_unavailable',
                            'error': 'page_retire_errors'}[hint]] += 1
            return module.eval().requires_grad_(False)
