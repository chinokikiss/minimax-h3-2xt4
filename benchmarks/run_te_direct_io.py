"""One bounded real-TE TP2 I/O experiment; no DiT, VAE, LoRA or generation."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import asdict
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import threading
import time
import traceback
from types import SimpleNamespace

import torch
from patches import comfy_tp_runtime as rt
from benchmarks import te_io_telemetry as telemetry
from benchmarks.te_direct_io import DirectFile, IOAudit, alignment_info, install

BASE_WORKER = rt._worker
BASE_RANK = rt.RankRuntime
BASE_WINDOW = rt.ResidencyWindow
AUDIT = None
OWNER = None
PROFILE_OK = False


def save(path, value):
    path.write_text(json.dumps(value, indent=2, default=str), encoding='utf-8')


def drop_files(paths):
    results = []
    for path in paths:
        with path.open('rb', buffering=0) as file:
            os.fdatasync(file.fileno())
            os.posix_fadvise(file.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
            results.append(dict(file=path.name, **telemetry.residency(file.fileno())))
    return results


class Sampler:
    def __init__(self, path):
        self.path, self.stop = path, threading.Event()
        self.thread = threading.Thread(target=self.run, daemon=True)

    def run(self):
        with self.path.open('w', encoding='utf-8') as stream:
            while not self.stop.is_set():
                begin, cpu = time.monotonic_ns(), time.thread_time_ns()
                row = telemetry.snapshot()
                row.update(monotonic_ns=begin, sampler_cpu_ns=time.thread_time_ns()-cpu)
                stream.write(json.dumps(row, separators=(',', ':'))+'\n')
                stream.flush()
                self.stop.wait(.2)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.stop.set()
        self.thread.join(timeout=3)


def strict_probe(scratch):
    """Establish actual O_DIRECT roundtrip and observe cache residency, CPU only."""
    import fcntl
    path = scratch/'direct_alignment_probe.bin'
    raw = bytes(range(251))*17000 + b'partial-tail'
    with path.open('wb', buffering=0) as file:
        file.write(raw)
        os.fdatasync(file.fileno())
    before = drop_files([path])[0]
    evidence = dict(alignment=alignment_info(path), before=before, path=str(path))
    with DirectFile(path) as file:
        evidence['descriptor_has_o_direct'] = bool(fcntl.fcntl(file.fd, fcntl.F_GETFL) & os.O_DIRECT)
        for offset, length in [(0, 8), (8, 1777), (4093, 8*2**20), (len(raw)-37, 999)]:
            assert file.pread(length, offset) == raw[offset:offset+length]
        file.seek(0)
        assert hashlib.file_digest(file, 'sha256').hexdigest() == hashlib.sha256(raw).hexdigest()
        evidence['after'] = telemetry.residency(file.fd)
    evidence['byte_exact'] = True
    evidence['cache_bypass_observed'] = (before.get('resident_pages') == 0 and
                                        evidence['after'].get('resident_pages') == 0)
    evidence['status'] = 'passed' if evidence['descriptor_has_o_direct'] and evidence['cache_bypass_observed'] else 'blocked_cache_bypass_unverified'
    path.unlink()
    return evidence


class ProfileWindow(BASE_WINDOW):
    @contextmanager
    def use(self, key):
        owner = OWNER
        start = time.monotonic_ns()
        a, b, c = [torch.cuda.Event(enable_timing=True) for _ in range(3)]
        a.record()
        with super().use(key) as block:
            b.record()
            ready = time.monotonic_ns()
            try:
                with torch.profiler.record_function('TE/layer'+str(key[2])):
                    yield block
            finally:
                c.record()
                owner.block_events.append((dict(layer=key[2], phase=AUDIT.phase,
                    start_ns=start, ready_host_ns=ready, enqueue_end_ns=time.monotonic_ns()), a, b, c))


class ProfileRank(BASE_RANK):
    def __init__(self, *args):
        global OWNER
        OWNER = self
        self.block_events = []
        self.commands = []
        self.te_calls = 0
        start = time.monotonic_ns()
        super().__init__(*args)
        assert self.readers['te'].verify_payloads is True
        AUDIT.emit(dict(kind='rank_startup', start_ns=start, end_ns=time.monotonic_ns()))

    def dispatch(self, command, args=(), kwargs=None):
        if command != 'te':
            return super().dispatch(command, args, kwargs)
        self.te_calls += 1
        AUDIT.phase = f'te{self.te_calls}'
        profile = self.te_calls == 2 and PROFILE_OK
        trace = Path(os.environ['H3_TE_IO_CASE_DIR'])/f'rank{self.rank}_trace.json'
        profiler = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA], record_shapes=False, profile_memory=False) if profile else None
        torch.cuda.synchronize(self.device)
        torch.cuda.reset_peak_memory_stats(self.device)
        begin = time.monotonic_ns()
        if profiler:
            profiler.__enter__()
        try:
            result = super().dispatch(command, args, kwargs)
            torch.cuda.synchronize(self.device)
        finally:
            end = time.monotonic_ns()
            if profiler:
                profiler.__exit__(None, None, None)
                profiler.export_chrome_trace(str(trace))
            self.commands.append(dict(phase=AUDIT.phase, profiled=profile, start_ns=begin,
                end_ns=end, wall_seconds=(end-begin)/1e9,
                peak_allocated=torch.cuda.max_memory_allocated(self.device),
                peak_reserved=torch.cuda.max_memory_reserved(self.device)))
            self.write_evidence()
        return result

    def write_evidence(self):
        blocks = []
        for row, a, b, c in self.block_events:
            blocks.append(dict(row, wait_gpu_event_ms=a.elapsed_time(b),
                               compute_gpu_event_ms=b.elapsed_time(c)))
        save(Path(os.environ['H3_TE_IO_CASE_DIR'])/f'rank{self.rank}.json',
             dict(commands=self.commands, blocks=blocks, stats=super().dispatch('stats'),
                  io_rows=AUDIT.rows, dropped_io_rows=AUDIT.dropped))


def worker(*args):
    global AUDIT, PROFILE_OK
    rank = args[0]
    directory = Path(os.environ['H3_TE_IO_CASE_DIR'])
    AUDIT = IOAudit(directory/f'rank{rank}_io.jsonl')
    if os.environ.get('H3_TE_IO_PROFILE') == '1':
        gate = dict(status='starting')
        try:
            torch.cuda.set_device(rank)
            trace = directory/f'rank{rank}_capture_gate.json'
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA]) as profiler:
                value = torch.ones(32, device=f'cuda:{rank}').add_(1)
                torch.cuda.synchronize(rank)
            profiler.export_chrome_trace(str(trace))
            kernels = sum(e.get('cat') == 'kernel' for e in json.loads(trace.read_text()).get('traceEvents', []))
            assert kernels > 0
            PROFILE_OK = True
            gate.update(status='passed', kernel_events=kernels)
            del value
        except BaseException as error:
            gate.update(status='blocked', error=repr(error))
        save(directory/f'rank{rank}_capture_gate_check.json', gate)
    install(os.environ['H3_TE_IO_WEIGHT_ROOT'], os.environ['H3_TE_IO_MODE'], AUDIT)
    from patches import tp2_bundle as bundle, windowed_tp
    open_checkpoint = bundle.open_checkpoint
    def te_only(path, **kwargs):
        if path == 'unused_dit_for_te_only':
            return SimpleNamespace(header={}, identity=('unused-dit',), logical_read_bytes=0,
                                   range_read_calls=0)
        return open_checkpoint(path, **kwargs)
    bundle.open_checkpoint = te_only
    original_prepare = windowed_tp.prepare_tp_block
    def prepare(*args, **kwargs):
        previous = getattr(AUDIT.local, 'layer', None)
        AUDIT.local.layer = args[2]
        try:
            return AUDIT.span('block_cpu_prepare', original_prepare, *args, **kwargs)
        finally:
            AUDIT.local.layer = previous
    windowed_tp.prepare_tp_block = prepare
    rt.RankRuntime, rt.ResidencyWindow = ProfileRank, ProfileWindow
    try:
        BASE_WORKER(*args)
    finally:
        AUDIT.close()


def case(args):
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    os.environ.update(H3_TE_IO_CASE_DIR=str(out), H3_TE_IO_MODE=args.case,
        H3_TE_IO_WEIGHT_ROOT=str(args.scratch), H3_TE_IO_PROFILE=str(int(args.profile)))
    # No DiT shell, weights, cache preparation or command is admitted. The unused
    # reader only satisfies the existing runtime's two-component constructor.
    config = rt.RuntimeConfig(te_path=str(args.scratch/'te_bundle.json'),
        dit_path='unused_dit_for_te_only', cross_model_cpu=False, cross_request_te=False,
        idle_residency=False)
    rt._worker = worker
    from patches.comfy_sampler import create_clip
    from comfy_extras.nodes_minimax_h3 import MiniMaxH3ReferenceToVideo
    from PIL import Image
    import numpy as np
    plan = json.loads((args.source/'turbo_challenge_plan.json').read_text())
    images = {f'ref_image_{i+1}':torch.from_numpy(np.asarray(Image.open(
        args.source/'bench_assets/l1_current'/row['file']).convert('RGB'), dtype=np.float32)/255)[None]
        for i, row in enumerate(plan['reference_images'])}
    state = dict(status='starting', mode=args.case, runtime=asdict(config), calls=[])
    service = None
    try:
        start = time.monotonic()
        service = rt.PhysicalTPService(config)
        state['startup_seconds'] = time.monotonic()-start
        clip = create_clip(service)
        for repeat in range(2 if not args.warmup else 1):
            start = time.monotonic_ns()
            # Exact Native REF2VA resizing/tokenization; no VAE encoders, DiT,
            # sampling, decoder or LoRA. Canvas controls only reference resizing.
            output = MiniMaxH3ReferenceToVideo.execute(clip, plan['prompt'], 1216, 704,
                plan['frames'], ref_image_size='match', vae=None, ref_images=images)[0]
            end = time.monotonic_ns()
            cond = output[0][0].detach().cpu().contiguous()
            digest = hashlib.sha256(memoryview(cond.view(torch.uint8).numpy()).cast('B')).hexdigest()
            state['calls'].append(dict(repeat=repeat, wall_seconds=(end-start)/1e9,
                profiled=bool(args.profile and repeat == 1), conditioning_sha256=digest,
                shape=list(cond.shape), finite=bool(torch.isfinite(cond).all())))
            del output, cond
            save(out/'case.json', state)
        state['stats'] = service.call('stats')
        state['status'] = 'complete'
    except BaseException as error:
        state.update(status='failed', error=repr(error))
        (out/'exception.txt').write_text(traceback.format_exc())
        raise
    finally:
        if service:
            service.close()
        save(out/'case.json', state)


def prepare_assets(source, scratch, mark):
    assets = json.loads((source/'bundle_assets.json').read_text())
    needed = sum(v['bytes'] for n, v in assets['objects'].items() if n.startswith('te.')) + assets['auxiliary']['te']['bytes']
    assert shutil.disk_usage(scratch).free > needed+4*2**30, 'Insufficient stage disk headroom'
    root = scratch/'base_bundle'
    root.mkdir()
    for path in (source/'bundle_provenance').glob('*.json'):
        shutil.copyfile(path, root/path.name)
    from huggingface_hub import hf_hub_download
    from benchmarks.run_h3_l1_turbo_batch import download_with_retry
    rows = []
    lock = threading.Lock()
    def download(item):
        name, expected = item
        path = Path(download_with_retry(lambda: hf_hub_download(assets['publication']['repo'], name,
            revision=assets['publication']['commit'], token=False, local_dir=root), name,
            lambda row:print('TE_IO_DOWNLOAD_RETRY', json.dumps(row), flush=True)))
        begin = time.monotonic()
        with path.open('rb') as file:
            digest = hashlib.file_digest(file, 'sha256').hexdigest()
        assert path.stat().st_size == expected['bytes'] and digest == expected['sha256']
        with lock:
            rows.append(dict(file=name, bytes=expected['bytes'], sha256=digest,
                             download_validation_sha_seconds=time.monotonic()-begin))
            if len(rows) % 10 == 0:
                mark('download_te', completed=len(rows))
    objects = [(n, v) for n, v in assets['objects'].items() if n.startswith('te.')]
    assert len(objects) == 100
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(download, objects))
    from checkpoint_ranges import assemble
    import requests
    aux_plan, original = assets['auxiliary']['te'], assets['source_identity']['te']
    url = f"https://huggingface.co/{assets['model_repository']}/resolve/{assets['model_revision']}/{original['remote_file']}"
    def read(start, end):
        def operation():
            with requests.get(url, params={'download':'true', 'offset':str(start)},
                    headers={'Range':f'bytes={start}-{end-1}'}, stream=True, timeout=(15, 90)) as response:
                assert response.status_code == 206 and response.headers.get('Content-Range') == f'bytes {start}-{end-1}/{original["bytes"]}'
                value = bytearray()
                for chunk in response.iter_content(2**20):
                    value.extend(chunk)
                    assert len(value) <= end-start
                return value
        return download_with_retry(operation, 'te_aux', lambda row:print('TE_IO_DOWNLOAD_RETRY', json.dumps(row), flush=True))
    mark('assemble_te_aux')
    auxiliary = scratch/'te_aux.safetensors'
    assemble(aux_plan, auxiliary, read)
    from patches.tp2_bundle import DESCRIPTOR
    save(scratch/'te_bundle.json', dict(format=DESCRIPTOR, kind='te', bundle_root=root.name,
        bundle_sha256=assets['publication']['bundle_sha256'], source={k:original[k] for k in ('bytes','sha256')},
        native_header=aux_plan['native_header'], native_header_sha256=aux_plan['native_header_sha256'],
        auxiliary_file=auxiliary.name, auxiliary_sha256=aux_plan['sha256']))
    return dict(publication=assets['publication'], shards=rows, auxiliary_sha256=aux_plan['sha256'])


def raw_io(paths, mode):
    def one(path):
        file = DirectFile(path) if mode == 'direct' else path.open('rb', buffering=0)
        buffer = bytearray(8*2**20)
        total = 0
        begin = time.monotonic_ns()
        with file:
            while (count := file.readinto(buffer)):
                total += count
        return dict(file=path.name, bytes=total, seconds=(time.monotonic_ns()-begin)/1e9)
    begin = time.monotonic_ns()
    with ThreadPoolExecutor(max_workers=2) as pool:
        rows = list(pool.map(one, paths))
    elapsed = (time.monotonic_ns()-begin)/1e9
    return dict(mode=mode, readers=2, rows=rows, seconds=elapsed,
                aggregate_mib_s=sum(row['bytes'] for row in rows)/2**20/elapsed,
                semantics='Read/copy wall throughput, no hashing; includes bounce copy for direct')


def summarize_io(path):
    """Durations are nested/overlapping work totals, never additive critical-path costs."""
    groups = {}
    with path.open(encoding='utf-8') as stream:
        for line in stream:
            row = json.loads(line)
            key = (row['phase'], row['kind'], row['category'])
            item = groups.setdefault(key, dict(phase=key[0], kind=key[1], category=key[2],
                calls=0, work_wall_ns=0, thread_cpu_ns=0, returned_bytes=0, hashed_bytes=0,
                aligned_requested_bytes=0))
            item['calls'] += 1
            item['work_wall_ns'] += row['end_ns']-row['start_ns']
            item['thread_cpu_ns'] += row.get('thread_cpu_ns', 0)
            item['returned_bytes'] += row.get('returned_bytes', 0)
            item['hashed_bytes'] += row.get('bytes', 0)
            if row['kind'] == 'direct_syscall':
                item['aligned_requested_bytes'] += row.get('requested_bytes', 0)
    return list(groups.values())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--scratch', type=Path, required=True)
    parser.add_argument('--case', choices=('buffered','direct'))
    parser.add_argument('--profile', action='store_true')
    parser.add_argument('--warmup', action='store_true')
    args = parser.parse_args()
    if args.case:
        return case(args)
    args.output.mkdir(parents=True, exist_ok=True)
    args.scratch.mkdir(parents=True, exist_ok=False)
    state = dict(status='starting', full_h3_pipeline=False, te_only=True, cases=[], raw_io=[],
        no_global_cache_drop=True, te_hash_verification=True, production_operators_unchanged=True)
    def mark(stage, **fields):
        state.update(status=stage, **fields)
        save(args.output/'result.json', state)
        print('TE_IO_STAGE', stage, json.dumps(fields), flush=True)
    stopped = threading.Event()
    def heartbeat():
        while not stopped.wait(60):
            print('TE_IO_HEARTBEAT', state['status'], flush=True)
    threading.Thread(target=heartbeat, daemon=True).start()
    try:
        from patches.legacy_runtime import validate_packages, validate_source
        validate_packages()
        validate_source(args.source/'ComfyUI')
        # Adapter tests and actual CPU/direct-I/O gate precede GPU execution.
        tests = subprocess.run([sys.executable, '-m', 'unittest', 'benchmarks.test_te_direct_io', '-v'],
                               capture_output=True, text=True, timeout=30)
        (args.output/'adapter_cpu_tests.log').write_text(tests.stdout+tests.stderr)
        assert tests.returncode == 0
        state['direct_support'] = strict_probe(args.scratch)
        mark('direct_support_gate')
        if state['direct_support']['status'] != 'passed':
            raise RuntimeError('O_DIRECT cache bypass could not be verified on this filesystem')
        devices = [dict(name=torch.cuda.get_device_name(i), capability=torch.cuda.get_device_capability(i),
            uuid=str(torch.cuda.get_device_properties(i).uuid)) for i in range(torch.cuda.device_count())]
        assert len(devices) == 2 and len({d['uuid'] for d in devices}) == 2
        assert all('T4' in d['name'] and d['capability'] == (7,5) for d in devices)
        state['devices'] = devices
        import importlib.metadata as md
        state['software'] = dict(torch=torch.__version__, cuda=torch.version.cuda,
            kitchen=md.version('comfy-kitchen'), tilelang=md.version('tilelang'), python=sys.version)
        state['mountinfo'] = Path('/proc/self/mountinfo').read_text()
        os.environ.update(HF_HUB_DISABLE_XET='1', HF_HUB_DISABLE_PROGRESS_BARS='1', HF_HUB_DOWNLOAD_TIMEOUT='90')
        mark('download_te')
        state['assets'] = prepare_assets(args.source, args.scratch, mark)
        files = sorted(args.scratch.rglob('*.safetensors'))
        state['actual_weight_alignment'] = alignment_info(files[0])
        # Match-check representative real header/tensor ranges before trials.
        with files[0].open('rb', buffering=0) as buffered, DirectFile(files[0]) as direct:
            for offset in (0, 8, 4093, files[0].stat().st_size-1777):
                buffered.seek(offset)
                assert direct.pread(1777, offset) == buffered.read(1777)
        subset = [args.scratch/'base_bundle'/f'te.block{block:02d}.rank{rank}.safetensors'
                  for block in (0,25) for rank in (0,1)]
        for mode in ('buffered','direct','direct','buffered'):
            drop_files(subset)
            state['raw_io'].append(raw_io(subset, mode))
            mark('raw_io_complete')
        nv = subprocess.Popen(['nvidia-smi','--query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.used,power.draw',
            '--format=csv','-l','1'], stdout=(args.output/'gpu_telemetry.csv').open('w'), stderr=subprocess.DEVNULL)
        try:
            for label, mode, profile, warmup in [('warmup','buffered',False,True),
                    ('a1','buffered',True,False), ('b1','direct',False,False),
                    ('b2','direct',True,False), ('a2','buffered',False,False)]:
                directory = args.output/label
                directory.mkdir()
                mark('te_'+label)
                save(directory/'residency_before.json', drop_files(files))
                before = telemetry.snapshot()
                command = [sys.executable, '-u', str(Path(__file__).resolve()), '--source', str(args.source),
                    '--output', str(directory), '--scratch', str(args.scratch), '--case', mode]
                if profile:
                    command += ['--profile']
                if warmup:
                    command += ['--warmup']
                with (directory/'worker.log').open('w') as log, Sampler(directory/'os_samples.jsonl'):
                    process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                    save(args.output.parent/'te_direct_supervision/active_process_group.json', dict(pid=process.pid))
                    try:
                        process.wait(timeout=720)
                    finally:
                        if process.poll() is None or process.returncode:
                            try:
                                os.killpg(process.pid, signal.SIGKILL)
                            except ProcessLookupError:
                                pass
                            process.wait(timeout=10)
                after = telemetry.snapshot()
                save(directory/'os_epoch.json', dict(before=before, after=after, counters=telemetry.delta(before,after)))
                record = json.loads((directory/'case.json').read_text()) if (directory/'case.json').exists() else {}
                state['cases'].append(dict(label=label, mode=mode, warmup=warmup, returncode=process.returncode, result=record))
                save(args.output/'result.json', state)
                if process.returncode or record.get('status') != 'complete':
                    raise RuntimeError('TE case failed: '+label)
                # Validate capture and explicit flush after worker shutdown. It
                # remains a diagnostic timing, excluded from unprofiled medians.
                for trace in directory.glob('rank*_trace.json'):
                    data = json.loads(trace.read_text())
                    kernels = sum(e.get('cat') == 'kernel' for e in data.get('traceEvents', []))
                    save(trace.with_suffix('.check.json'), dict(kernel_events=kernels,
                        complete=kernels > 0, worker_exited=True))
                    with trace.open('rb') as src, gzip.open(str(trace)+'.gz', 'wb') as dst:
                        shutil.copyfileobj(src, dst)
                    trace.unlink()
                for io in directory.glob('rank*_io.jsonl'):
                    save(io.with_suffix('.summary.json'), dict(groups=summarize_io(io),
                        semantics='Nested spans and cross-thread work totals; do not sum them as TE critical-path attribution'))
                    with io.open('rb') as src, gzip.open(str(io)+'.gz', 'wb', compresslevel=1) as dst:
                        shutil.copyfileobj(src, dst)
                    io.unlink()
        finally:
            nv.terminate()
            nv.wait(timeout=10)
        hashes = {call['conditioning_sha256'] for item in state['cases'] for call in item['result']['calls']}
        assert len(hashes) == 1, 'I/O modes changed TE conditioning'
        assert all(call['finite'] for item in state['cases'] for call in item['result']['calls'])
        state['conditioning_bit_exact'] = True
        state['profile_checks'] = [dict(file=str(p.relative_to(args.output)), **json.loads(p.read_text()))
                                  for p in args.output.glob('*/rank*_trace.check.json')]
        state['complete_gpu_profiles'] = len(state['profile_checks']) == 4 and all(
            item['complete'] for item in state['profile_checks'])
        mark('complete')
    except BaseException as error:
        state.update(status='blocked_or_failed', error=repr(error))
        (args.output/'exception.txt').write_text(traceback.format_exc())
        raise
    finally:
        stopped.set()
        save(args.output/'result.json', state)


if __name__ == '__main__':
    main()
