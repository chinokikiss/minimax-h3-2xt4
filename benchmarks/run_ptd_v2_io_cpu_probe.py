"""CPU-only, opt-in PTD v2 mounted Dataset readback/parallel-I/O qualification.

Run in a fresh CPU or GPU notebook process against an already mounted Bank.
No Kaggle API, CUDA kernels, uploads, remounts, file deletion or page-cache
global drop commands are used. Source bytes remain read-only.
"""
import argparse
import gc
import hashlib
import json
import os
import math
from pathlib import Path
import statistics
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import psutil
import torch

from patches.host_memory_pressure import CgroupMemory
from patches.ptd.bank_v2 import PTDBankV2
from patches.rank_local_int8 import state_hashes
from patches.tp2_disk_bake import atomic_json
from patches.tp2_bundle import sha

FORMAT = 'h3-ptd-v2-mounted-io-cpu-probe-v1'
STANDARD_PROFILES = ((1, False), (4, False), (4, True), (8, True))


class MemorySampler:
    """Bounded 20 Hz process/cgroup snapshots during rank loading only."""
    def __init__(self):
        self.process = psutil.Process()
        self.cg = CgroupMemory()
        self.stop = threading.Event()
        self.worker = None
        self.rss_peak = 0
        self.cgroup_peak = 0
        self.file_peak = 0
        self.anon_peak = 0
        self.available = False
        self.first_cgroup = None
        self.last_cgroup = None

    def _sample(self):
        while True:
            self.rss_peak = max(self.rss_peak, self.process.memory_info().rss)
            state = self.cg.snapshot(fresh=True)
            if state.get('available'):
                self.available = True
                if self.first_cgroup is None:
                    self.first_cgroup = state
                self.last_cgroup = state
                self.cgroup_peak = max(self.cgroup_peak, state['current'])
                self.file_peak = max(self.file_peak, state['file_bytes'])
                self.anon_peak = max(self.anon_peak, state.get('anon') or 0)
            if self.stop.wait(.05):
                return

    def __enter__(self):
        self.worker = threading.Thread(target=self._sample, name='ptd-v2-cpu-memory-meter', daemon=True)
        self.worker.start()
        return self

    def __exit__(self, *_):
        self.stop.set()
        self.worker.join(timeout=2)
        if self.worker.is_alive():
            raise RuntimeError('PTD I/O memory observer did not stop')

    def summary(self):
        first, last = self.first_cgroup, self.last_cgroup
        def counter_delta(key):
            return ((last.get(key) or 0) - (first.get(key) or 0)) if first else None
        return dict(rss_peak_bytes=self.rss_peak,
                    cgroup_available=self.available,
                    cgroup_current_peak_bytes=self.cgroup_peak if self.available else None,
                    cgroup_file_peak_bytes=self.file_peak if self.available else None,
                    cgroup_anon_peak_bytes=self.anon_peak if self.available else None,
                    cgroup_file_first_bytes=first['file_bytes'] if first else None,
                    cgroup_file_last_bytes=last['file_bytes'] if first else None,
                    cgroup_file_delta_bytes=(last['file_bytes'] - first['file_bytes']) if first else None,
                    cgroup_pgscan_kswapd_delta=counter_delta('scan_kswapd'),
                    cgroup_refault_file_delta=counter_delta('refault_file'),
                    cgroup_psi_some_total_delta=(
                        last.get('psi', {}).get('some_total', 0.) -
                        first.get('psi', {}).get('some_total', 0.)) if first else None,
                    cgroup_psi_full_total_delta=(
                        last.get('psi', {}).get('full_total', 0.) -
                        first.get('psi', {}).get('full_total', 0.)) if first else None,
                    cgroup_oom_delta=((last['events'].get('oom', 0) - first['events'].get('oom', 0))
                                      if first else None))


def _validate(bank, hop, layers, ranks, profiles, repeats, max_seconds):
    if type(hop) is not int or not 0 <= hop < 8:
        raise ValueError('Hop index must be 0..7')
    if not layers or len(layers) > 4 or len(set(layers)) != len(layers):
        raise ValueError('Probe requires 1..4 distinct representative layers')
    if any(type(x) is not int or x < 0 or x >= bank.layers for x in layers):
        raise ValueError('Probe layer index exceeds admitted Bank')
    if not ranks or len(set(ranks)) != len(ranks) or any(x not in (0, 1) for x in ranks):
        raise ValueError('Probe ranks must be a nonempty subset of 0,1')
    if not profiles or len(set(profiles)) != len(profiles):
        raise ValueError('Probe I/O profiles must be distinct and nonempty')
    for workers, retire in profiles:
        bank.configure_io(workers, retire)
    if type(repeats) is not int or not 1 <= repeats <= 3:
        raise ValueError('Probe repeats must be 1..3')
    if not math.isfinite(max_seconds) or not 0 < max_seconds <= 7200:
        raise ValueError('Probe deadline must be within (0, 7200] seconds')
    if len(layers) * len(ranks) * len(profiles) * repeats > 96:
        raise ValueError('Probe exceeds the 96-rank-load safety cap')


def _summary(cases):
    by_profile = {}
    for case in cases:
        name = case['profile']
        by_profile.setdefault(name, []).append(case)
    result = {}
    for name, values in by_profile.items():
        io_ms = [row['read_ms'] for row in values]
        read_bytes = sum(row['source_read_bytes'] for row in values)
        read_seconds = sum(io_ms) / 1000.
        result[name] = dict(runs=len(values), median_read_ms=statistics.median(io_ms),
                            total_read_bytes=read_bytes,
                            aggregate_read_mib_per_sec=(read_bytes / 2**20 / read_seconds
                                                        if read_seconds else None),
                            median_load_ms=statistics.median(row['load_ms'] for row in values),
                            max_rss_peak_bytes=max(row['memory']['rss_peak_bytes'] for row in values),
                            page_retire_hints=sum(row['page_retire_hints'] for row in values),
                            page_retire_errors=sum(row['page_retire_errors'] for row in values),
                            page_retire_unavailable=sum(row['page_retire_unavailable'] for row in values))
    baseline = result.get('1-off')
    if baseline and baseline['median_read_ms'] > 0:
        for row in result.values():
            row['median_io_speedup_vs_1_off'] = baseline['median_read_ms'] / max(
                row['median_read_ms'], 1e-9)
    return result


def probe_bank(bank, *, hop=0, layers=(0,), ranks=(0, 1),
               profiles=STANDARD_PROFILES, repeats=1, max_seconds=900.,
               on_progress=None):
    """Run rank loads, hash all tensors after timing, and report bounded CPU/NFS metrics.

    This does not enforce a performance improvement: the mounted NFS/backend
    cache state is not controlled. Metrics are comparisons, not cold-read claims.
    """
    layers, ranks, profiles = tuple(layers), tuple(ranks), tuple(profiles)
    _validate(bank, hop, layers, ranks, profiles, repeats, max_seconds)
    # CUDA may already be initialized by the notebook or its test fixtures;
    # this probe itself dispatches no device operations or GPU kernels.
    report = dict(format=FORMAT, status='running', cpu_only=True,
                  cuda_preinitialized=bool(torch.cuda.is_initialized()),
                  physical_t4_qualified=False, hop=hop, layers=list(layers),
                  ranks=list(ranks), repeats=repeats,
                  profiles=[dict(io_workers=n, retire_page_cache=retire)
                            for n, retire in profiles],
                  recipe_sha256=bank.recipe_sha, manifest=str(bank.path),
                  manifest_sha256=sha(bank.path),
                  part_roots={name: str(root) for name, root in bank.parts.items()},
                  part_identities={name: spec['identity'] for name, spec in
                      json.loads(bank.path.read_text(encoding='utf-8'))['parts'].items()},
                  pread_supported=hasattr(os, 'pread'), fixture=bank.layers != 50,
                  cpu_count=psutil.cpu_count(logical=True),
                  ranks_simultaneous=False,
                  warning=('Ranks are measured sequentially, not as simultaneous dual-T4 workers; '
                           'fixed-order runs may include warm NFS/page cache'),
                  cases=[])
    started = time.monotonic()
    if on_progress is not None:
        on_progress(report)
    try:
        for repetition in range(repeats):
            for workers, retire in profiles:
                bank.configure_io(workers, retire)
                for layer in layers:
                    for rank in ranks:
                        if time.monotonic() - started > max_seconds:
                            raise TimeoutError('PTD v2 CPU probe elapsed budget exhausted')
                        reader = bank.reader(hop)
                        stem = f'dit.block{layer:02d}.rank{rank}'
                        rss_before = psutil.Process().memory_info().rss
                        with MemorySampler() as memory:
                            begin = time.perf_counter()
                            module = reader.load_rank(layer, rank)
                            load_ms = (time.perf_counter() - begin) * 1000.
                        hash_begin = time.perf_counter()
                        actual = state_hashes(module)
                        hash_ms = (time.perf_counter() - hash_begin) * 1000.
                        if actual != reader.plan['blocks'][stem]['state_sha256']:
                            raise ValueError('PTD v2 CPU probe rank bitwise state mismatch: ' + stem)
                        state_digest = hashlib.sha256(json.dumps(
                            actual, sort_keys=True).encode()).hexdigest()
                        source = reader.shard_readers[stem]
                        case = dict(profile=f'{workers}-{"on" if retire else "off"}',
                                    repetition=repetition, hop=hop, layer=layer, rank=rank,
                                    rank_shard=str(source.path),
                                    io_workers=workers, retire_page_cache=retire,
                                    status='bit_exact', state_digest_sha256=state_digest,
                                    read_ms=reader.stats['rank_read_ms'],
                                    load_ms=load_ms, state_hash_ms=hash_ms,
                                    source_read_bytes=source.source_read_bytes,
                                    read_calls=source.range_read_calls,
                                    page_retire_ms=reader.stats['page_retire_ms'],
                                    page_retire_hints=reader.stats['page_retire_hints'],
                                    page_retire_unavailable=reader.stats['page_retire_unavailable'],
                                    page_retire_errors=reader.stats['page_retire_errors'],
                                    rss_before_bytes=rss_before, memory=memory.summary())
                        report['cases'].append(case)
                        del module, reader, source
                        gc.collect()  # Outside the timed region and only in this offline probe.
                        if on_progress is not None:
                            on_progress(report)
        report['summary'] = _summary(report['cases'])
        retired = [case for case in report['cases'] if case['retire_page_cache']]
        report['page_retire_status'] = (
            'not_requested' if not retired else
            'hint_error' if any(case['page_retire_errors'] for case in retired) else
            'unavailable' if any(case['page_retire_unavailable'] for case in retired) else
            'hint_issued')
        report['cgroup_observed'] = all(case['memory']['cgroup_available']
                                        for case in report['cases'])
        report['elapsed_seconds'] = time.monotonic() - started
        report['status'] = 'passed'
        if on_progress is not None:
            on_progress(report)
        return report
    except Exception as error:
        report.update(status='failed', error_type=type(error).__name__,
                      error_message=str(error),
                      elapsed_seconds=time.monotonic() - started)
        if on_progress is not None:
            on_progress(report)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--part-roots-json', default='{}',
                        help='Explicit JSON map of immutable part id to mounted path')
    parser.add_argument('--part-roots-file', type=Path,
                        help='UTF-8 JSON file containing the immutable part mount map')
    parser.add_argument('--hop', type=int, default=0)
    parser.add_argument('--layers', nargs='+', type=int, required=True)
    parser.add_argument('--ranks', nargs='+', type=int, default=[0, 1])
    parser.add_argument('--repeats', type=int, default=1)
    parser.add_argument('--max-seconds', type=float, default=900.)
    parser.add_argument('--fixture', action='store_true',
                        help='Only for disposable one-layer CPU tests')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError('Preserve previous PTD I/O probe output')
    if not output.parent.is_dir():
        raise FileNotFoundError('PTD probe output parent must already exist')
    if args.part_roots_file is not None and args.part_roots_json != '{}':
        raise ValueError('Use either --part-roots-file or --part-roots-json')
    roots = (json.loads(args.part_roots_file.read_text(encoding='utf-8'))
             if args.part_roots_file is not None else json.loads(args.part_roots_json))
    if (not isinstance(roots, dict) or any(
            not isinstance(key, str) or not isinstance(value, str) or not value
            for key, value in roots.items())):
        raise ValueError('Part roots must be an explicit JSON map')
    bank = PTDBankV2(args.manifest, part_roots=tuple(sorted(roots.items())),
                     allow_fixture=args.fixture, verify_payloads=False)
    report = probe_bank(bank, hop=args.hop, layers=tuple(args.layers),
                        ranks=tuple(args.ranks), repeats=args.repeats,
                        max_seconds=args.max_seconds,
                        on_progress=lambda value: atomic_json(output, value))
    print(json.dumps(dict(status=report['status'], output=str(output),
                          summary=report['summary']), indent=2), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
