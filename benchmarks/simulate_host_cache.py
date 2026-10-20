"""Two consecutive trained CUDA requests with persistent disk/RAM scheduling.

--physical executes the same rank function on two real GPUs using NCCL.
This is a correctness/admission gate, not an end-to-end latency benchmark.
"""
import argparse
from dataclasses import replace
from datetime import timedelta
import hashlib
import json
import os
from pathlib import Path
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, Path(os.environ.get('COMFY_ROOT', ROOT/'ComfyUI')), ROOT/'.retest-deps', ROOT/'benchmarks'):
    sys.path.insert(0, str(path))

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
import retest
from fixture_inputs import block_input
from logical_dual_gpu import LogicalDualGPU, isolate_cublas_workspaces
from patches.host_weight_cache import HostBudget, HostWeightCache, GiB
from patches.weight_residency import ResidencyWindow, WeightSpec
from patches.windowed_tp import CheckpointReader, tp_specs
from patches.optimized_tp import enable
import patches.tp_qwen3vl as te
import patches.tp_dit as dit
from te_dit_study import inputs, invoke, GATE


def setup(local):
    enable('scalemax')
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    if local:
        import comfy_kitchen as ck
        ck.backends.cuda._max_dynamic_shared_memory_per_block = lambda x: 49152
        isolate_cublas_workspaces()


def data_for(kind, path, round_index, device):
    if kind == 'te':
        x, args = inputs('te', path, 512, 1, torch.device(device))
        return x + round_index*.03125, args
    curve = retest.state(path, 'adaln_t_table')['']
    data = block_input('dit', 8192, 3100+round_index, curve)
    return data['input'].to(device), (data['t_emb'].to(device), data['segments'], data['rope'].to(device))


def goldens(paths, output):
    result = {}
    with torch.inference_mode():
        for kind in ('te', 'dit'):
            native = retest.load_native_block(kind, paths[kind], 0, 'cuda:0')
            for round_index in range(2):
                x, args = data_for(kind, paths[kind], round_index, 'cuda:0')
                if kind == 'te':
                    out = invoke(kind, native, x, args, True)
                else:
                    out = native(x.clone(), *args)
                    out = native(out.clone(), *args)
                result[f'{kind}_{round_index}'] = out.cpu()
                del x, args, out
            del native
            retest.cleanup()
        with torch.inference_mode(False), torch.no_grad():
            video, state = retest.load_video_native(paths['video'], 'cuda:0')
        del state
        if not hasattr(video.decoder.transformer_blocks[0].attn.to_qkv.weight, '_qdata'):
            raise RuntimeError('Native video comparator lost INT8 dispatch')
        for i in range(2):
            z = torch.full((1,24,2,4,4), .03125*(i+1), device='cuda:0', dtype=torch.float16)
            result[f'video_{i}'] = video.decode(z).cpu()
        del video, z
    retest.cleanup()
    torch.save(result, output/'goldens.pt')


def budget_child(budget, ready, done):
    if not budget.acquire(20*2**20):
        raise RuntimeError('Child reservation unexpectedly rejected')
    ready.set()
    if not done.wait(30):
        raise TimeoutError('Parent did not finish shared budget probe')
    budget.release(20*2**20)


def cache_policy_checks():
    budget = HostBudget(32*2**20, reserve_bytes=0)
    cache = HostWeightCache(budget, 8*2**20)
    def linear(value):
        m = nn.Linear(4, 4, bias=False)
        m.weight.data.fill_(value)
        return m
    a = WeightSpec(('checkpoint-a', 'fp32'), lambda: linear(1), 64)
    b = WeightSpec(('checkpoint-b', 'fp32'), lambda: linear(2), 64)
    try:
        cache.prefetch([a]); cache.wait_idle()
        cache.begin_request([b])
        # Two independent rank windows share the same physical memory limit.
        for spec, expected in [(b,8.), (a,4.)]:
            w = ResidencyWindow([spec], [spec.key], 'cuda:0', max_weight_bytes=64,
                                pinned_bytes=128, host_cache=cache, reserve_bytes=0)
            try:
                with w.use(spec.key) as model:
                    actual = model(torch.ones(1,4,device='cuda:0')).cpu()
                assert torch.equal(actual, torch.full((1,4), expected))
                w.finish()
            finally:
                w.close()
        assert not budget.acquire(33*2**20)
        # Cancel queued stale work and discard a stale in-flight preparation
        # at its module boundary, without retaining its CPU tensors.
        started, release = threading.Event(), threading.Event()
        def delayed():
            started.set()
            if not release.wait(10):
                raise TimeoutError('Cancellation probe did not release preparation')
            return linear(3)
        stale = WeightSpec('stale-running', delayed, 64)
        queued = WeightSpec('stale-queued', lambda: linear(4), 64)
        cache.prefetch([stale, queued])
        assert started.wait(10)
        cache.begin_request([b])
        release.set(); cache.wait_idle()
        assert cache.stats['discarded_preparations']==1
        assert cache.stats['cancelled_prefetches']==1
        snapshot = cache.snapshot()
    finally:
        cache.close()
    assert budget.snapshot()['reserved_bytes'] == 0
    context = mp.get_context('spawn')
    ready, done = context.Event(), context.Event()
    child = context.Process(target=budget_child, args=(budget,ready,done))
    child.start()
    try:
        assert ready.wait(30)
        assert budget.snapshot()['reserved_bytes']==20*2**20
        assert not budget.acquire(20*2**20)
    finally:
        done.set(); child.join(30)
        if child.is_alive():
            child.terminate(); child.join()
    assert child.exitcode==0 and budget.snapshot()['reserved_bytes']==0
    return dict(passed=True, configuration_change=True, shared_budget_denial=True,
                process_shared_admission=True, stale_prefetch_cancelled=True,
                ownership_drained=True, cache=snapshot)


def rank_run(rank, paths, output, budget, physical):
    device = f'cuda:{rank}' if physical else 'cuda:0'
    torch.cuda.set_device(device)
    cache = HostWeightCache(budget, 4*GiB)
    readers = {k: CheckpointReader(paths[k]) for k in ('te','dit')}
    text = tp_specs(readers['te'], 'te', rank, 1)
    denoise = tp_specs(readers['dit'], 'dit', rank, 1)
    specs = text + denoise
    if rank == 0:
        def prepare_video():
            model, state = retest.load_video_native(paths['video'], 'cpu')
            del state
            return model
        video_spec = WeightSpec(('video', str(paths['video'])), prepare_video,
                                Path(paths['video']).stat().st_size+64*2**20,
                                # Native mixed-precision Linear creates no
                                # dense weight before loading. CPU INT8 wraps
                                # checkpoint storage; reserve source + final +
                                # one full group of conversion headroom.
                                prepare_peak_bytes=3*(Path(paths['video']).stat().st_size+64*2**20))
        specs += [video_spec]
    references = torch.load(output/'goldens.pt', weights_only=True)
    rows, rounds = [], []
    before_warm = None
    def save(passed=False):
        retest.write_json(output/f'rank{rank}.json', dict(passed=passed, cases=rows,
                          rounds=rounds, host_cache=cache.snapshot(), physical=physical))
    def check(label, out, reference):
        score = retest.metrics(out.cpu(), reference)
        score.update(label=label, passed=retest.passes(score, GATE))
        rows.append(score); save()
        print('CACHE',rank,label,score['passed'],score['is_bit_exact'],flush=True)
        if not score['passed']:
            raise RuntimeError('Native output mismatch: '+label)
    try:
        for round_index in range(2):
            video_completed_ns = None
            cache.begin_request(specs)
            uses = [text[0].key] + [denoise[0].key]*2
            if rank == 0:
                uses += [video_spec.key]
            window = ResidencyWindow(specs, uses, device, max_weight_bytes=4*GiB,
                pinned_bytes=3*GiB, lookahead=1, reserve_bytes=256*2**20,
                adaptive=False, host_cache=cache)
            try:
                x, args = data_for('te', paths['te'], round_index, device)
                with window.use(text[0].key) as block:
                    out = block(x, args[0], args[1])
                check(f'te_s512_request{round_index}', out, references[f'te_{round_index}'])
                del block, x, args, out
                if round_index:
                    assert cache.stats['preparation_counts'][str(text[0].key)] == before_warm
                x, args = data_for('dit', paths['dit'], round_index, device)
                for _ in range(2):
                    with window.use(denoise[0].key) as block:
                        x = block(x.clone(), *args)
                check(f'dit_s8192_repeated_request{round_index}', x, references[f'dit_{round_index}'])
                del block, x, args
                if rank == 0:
                    with window.use(video_spec.key) as video:
                        z = torch.full((1,24,2,4,4), .03125*(round_index+1), device=device, dtype=torch.float16)
                        out = video.decode(z)
                    check(f'video36_public_request{round_index}', out, references[f'video_{round_index}'])
                    video_completed_ns = time.monotonic_ns()
                    del video, z, out
                stats = window.finish()
                assert stats['loads'][str(denoise[0].key)] == 1
                rounds.append(dict(request=round_index, window=stats,
                                   video_completed_ns=video_completed_ns))
            finally:
                window.close()
            # Both rank windows have ended before the next-request warmup.
            flag = torch.ones(1, device=device)
            te.dist.all_reduce(flag)
            # NCCL completion is ordered on the GPU stream; CPU disk workers
            # need an explicit host observation at this request boundary.
            if flag.item() != 2:
                raise RuntimeError('Both ranks did not complete the request')
            del flag
            if round_index == 0:
                cache.trim()  # Force an actual post-decode disk reload.
                rounds[-1]['idle_prefetch_started_ns'] = time.monotonic_ns()
                begin = time.perf_counter()
                cache.prefetch(text)
                enqueue_ms = (time.perf_counter()-begin)*1000
                cache.wait_idle()  # Diagnostic only: idle interval is reported.
                rounds[-1]['idle_prefetch_ms'] = (time.perf_counter()-begin)*1000
                rounds[-1]['prefetch_enqueue_ms'] = enqueue_ms
                before_warm = cache.stats['preparation_counts'][str(text[0].key)]
                rounds[-1]['host_after_idle_prefetch'] = cache.snapshot()
                print('CACHE',rank,'TE idle preload ready',flush=True)
            save()
        save(passed=True)
    finally:
        cache.close()


def physical_worker(rank, paths, output, budget, port):
    torch.cuda.set_device(rank)
    setup(False)
    store = dist.TCPStore('127.0.0.1', port, 2, False, timeout=timedelta(seconds=180))
    dist.init_process_group('nccl', store=store, rank=rank, world_size=2,
                            timeout=timedelta(seconds=180), device_id=torch.device('cuda',rank))
    try:
        with torch.inference_mode():
            rank_run(rank, paths, Path(output), budget, True)
    finally:
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--physical', action='store_true')
    parser.add_argument('--checkpoints', type=Path)
    parser.add_argument('--output', type=Path, default=ROOT/'semantic_results/host_cache')
    parser.add_argument('--policy-only', action='store_true')
    options = parser.parse_args()
    output = options.output; output.mkdir(parents=True, exist_ok=True)
    paths = json.loads(options.checkpoints.read_text()) if options.checkpoints else {
        'te':'D:/minimaxH3/qwen3vl_32b_minimax_h3_int8_convrot.safetensors',
        'dit':'D:/minimaxH3/MiniMax_H3_Ref2VA_pruned_int8_convrot.safetensors',
        'video':'D:/minimaxH3/minimax_h3_video_vae_int8_convrot.safetensors'}
    files = list((ROOT/'patches').glob('*.py')) + [Path(__file__),ROOT/'benchmarks/logical_dual_gpu.py']
    hashes = {p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    result = dict(passed=False, physical=options.physical, source_sha256=hashes,
                  scope='Two consecutive requests: trained TE/DiT block zero, repeated DiT use, full compact public video decode; no sampler or performance claim')
    retest.write_json(output/'result.json',result)
    setup(not options.physical)
    result['policy'] = cache_policy_checks()
    if not options.policy_only:
        print('CACHE computing Native references',flush=True)
        goldens(paths, output)
        budget = HostBudget()
        if options.physical:
            if torch.cuda.device_count()!=2 or any('T4' not in torch.cuda.get_device_name(i) for i in range(2)):
                raise RuntimeError('Physical gate requires exactly two Tesla T4 GPUs')
            # Parent owns the listener continuously; no free-port handoff race.
            store = dist.TCPStore('127.0.0.1',0,2,True,timeout=timedelta(seconds=180),wait_for_workers=False)
            mp.spawn(physical_worker,args=(paths,str(output),budget,store.port),nprocs=2,join=True)
        else:
            comm = LogicalDualGPU(timeout=180)
            te.dist = dit.dist = comm
            comm.run([lambda rank=rank:rank_run(rank,paths,output,budget,False) for rank in range(2)])
        result['ranks'] = [json.loads((output/f'rank{rank}.json').read_text()) for rank in range(2)]
        completed = result['ranks'][0]['rounds'][0]['video_completed_ns']
        assert all(r['rounds'][0]['idle_prefetch_started_ns'] >= completed for r in result['ranks'])
        result['post_decode_ordering_passed'] = True
        result['budget'] = budget.snapshot()
        assert result['budget']['reserved_bytes']==0
        assert all(r['passed'] for r in result['ranks'])
    result['passed'] = True
    retest.write_json(output/'result.json',result)
    print('CACHE COMPLETE',flush=True)


if __name__=='__main__':
    main()
