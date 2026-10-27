"""Bounded column-layout ablation on trained TE shards; no full H3 run."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from datetime import timedelta
import hashlib
import importlib.metadata as md
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time
import traceback
from unittest.mock import patch

import torch

from benchmarks.te_column_layout import (
    VARIANTS, accumulator, balanced_order, column_layout,
)

LENGTHS = (128, 353, 1306, 4096)


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')


def digest(value):
    return hashlib.sha256(value.detach().contiguous().cpu().view(torch.uint8).numpy()).hexdigest()


def paired_measure(run, barrier, warmups=5, repeats=9):
    """Balance variant order; context setup and CPU barriers are outside timing."""
    for _ in range(warmups):
        for name in VARIANTS:
            with column_layout(name):
                barrier()
                value = run(name)
                del value
    torch.cuda.synchronize()
    result = {name: dict(samples=[], incremental_peak_bytes=0) for name in VARIANTS}
    for repetition in range(repeats):
        for name in balanced_order(repetition):
            with column_layout(name):
                barrier()
                torch.cuda.synchronize()
                initial = torch.cuda.memory_allocated()
                torch.cuda.reset_peak_memory_stats()
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                begin = time.perf_counter()
                start.record()
                value = run(name)
                end.record()
                end.synchronize()
                wall_ms = 1000 * (time.perf_counter() - begin)
                result[name]['samples'].append(dict(wall_ms=wall_ms, cuda_ms=start.elapsed_time(end)))
                result[name]['incremental_peak_bytes'] = max(
                    result[name]['incremental_peak_bytes'], torch.cuda.max_memory_allocated() - initial)
                del value
    for value in result.values():
        value['median_wall_ms'] = statistics.median(r['wall_ms'] for r in value['samples'])
        value['median_cuda_ms'] = statistics.median(r['cuda_ms'] for r in value['samples'])
    return result


def profile_variants(run, x, barrier, output, rank):
    from torch.profiler import ProfilerActivity, profile, record_function
    activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
    with profile(activities=activities) as probe:
        probe_value = x + 1
        torch.cuda.synchronize()
    probe_path = output / f'rank{rank}_probe_trace.json'
    probe.export_chrome_trace(str(probe_path))
    del probe_value

    def kernels(path):
        return [e for e in json.loads(path.read_text())['traceEvents']
                if e.get('cat') == 'kernel' and e.get('ph') == 'X']

    if not kernels(probe_path):
        raise RuntimeError('Minimal CUDA probe lacks kernel coverage')
    barrier()
    with profile(activities=activities) as trace:
        for name in VARIANTS:
            with column_layout(name):
                barrier()
                with record_function('TE_column_layout/' + name):
                    value = run(name)
                    torch.cuda.synchronize()
                    del value
    path = output / f'rank{rank}_layout_trace.json'
    trace.export_chrome_trace(str(path))
    assert kernels(path), 'Full layout trace lacks kernel coverage'
    return dict(status='cuda_trace_saved', file=path.name, probe_file=probe_path.name)


def _worker(args):
    import torch.distributed as dist
    from safetensors.torch import load_file
    from comfy.text_encoders.llama import precompute_freqs_cis
    from benchmarks.te_parallel_matrix import make_block, merge_shards
    from patches import optimized_tp
    from patches.h3_operator_policy import H3OperatorPolicy, configure_block
    from patches.dit_column_linear import shared_column_linears

    local = args.local
    rank = 0 if local else int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(rank)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    if local:
        barrier = lambda: None
    else:
        dist.init_process_group('nccl', timeout=timedelta(seconds=180))
        group = dist.new_group(backend='gloo', timeout=timedelta(seconds=180))
        barrier = lambda: dist.barrier(group=group)
    optimized_tp.enable()
    record = dict(status='running', rank=rank, device=torch.cuda.get_device_name(rank),
                  capability=list(torch.cuda.get_device_capability(rank)),
                  uuid=str(torch.cuda.get_device_properties(rank).uuid),
                  physical_t4_qualification=not local, full_te_encoder=False, full_h3_pipeline=False,
                  input_distribution='Seeded residuals through trained block0; not captured H3 inputs',
                  topology='unsharded local smoke' if local else 'current TP2 with actual NCCL',
                  attention='native' if local else 'kitchen_causal',
                  column_selection='forced pinned eager for local API preview' if local else 'unchanged production selection',
                  warmups=5, repeats=9, variants=VARIANTS, cases=[], gemm_controls=[])
    path = args.output / f'rank{rank}.json'
    save(path, record)
    try:
        if not local:
            assert record['capability'] == [7, 5] and 'T4' in record['device']
        states = [load_file(str(args.assets / f'te.block00.rank{r}.safetensors')) for r in (0, 1)]
        block = make_block(merge_shards(*states) if local else states[rank], rank,
                           1 if local else 2, torch.device('cuda', rank))
        configure_block(block, 'te', H3OperatorPolicy(te_attention='native' if local else 'kitchen_causal'))
        # Keep trained TP2 projection weights for the operator sweep on either
        # machine. Local full-block control is unsharded and labeled separately.
        micro_weights = {name: states[rank][name + '.weight'].cuda(rank)
                         for name in ('q_proj', 'k_proj', 'gate_proj')}
        del states
        for length in args.lengths:
            gen = torch.Generator().manual_seed(20261027 + length)
            x = (torch.randn(1, length, 5120, generator=gen) / 64).half().cuda(rank)
            mask = torch.empty((length, length), device=x.device, dtype=x.dtype)
            mask.fill_(torch.finfo(x.dtype).min / 4).triu_(1)
            freqs = precompute_freqs_cis(128, torch.arange(length, device=x.device).unsqueeze(0),
                                       1000000., device=x.device)
            run = lambda name: block(x, attention_mask=mask, freqs_cis=freqs)
            with column_layout('legacy_copy'):
                reference = run('legacy_copy')
            checks = {}
            for name in VARIANTS:
                with column_layout(name):
                    from patches import dit_column_linear as column
                    with patch.object(column, '_column_accumulate', wraps=column._column_accumulate) as calls:
                        actual = run(name)
                    assert calls.call_count == 5, 'Experiment did not reach all five TE column GEMMs'
                    checks[name] = dict(byte_exact=torch.equal(actual, reference),
                                        finite=bool(torch.isfinite(actual).all()), output_sha256=digest(actual),
                                        column_gemm_calls=calls.call_count)
                    assert checks[name]['byte_exact'] and checks[name]['finite'], name
                    del actual
            del reference
            block_times = paired_measure(run, barrier)
            normed = torch.nn.functional.rms_norm(x.float(), (5120,),
                block.input_layernorm.weight.float(), block.input_layernorm.eps).half()
            projection_times = {}
            projection_checks = {}
            for label, names in (('qkv', ('q_proj', 'k_proj', 'v_proj')),
                                 ('gate_up', ('gate_proj', 'up_proj'))):
                weights = [(getattr(block, name).weight, getattr(block, name).weight_scale) for name in names]
                project = lambda name: shared_column_linears(normed, weights)
                with column_layout('legacy_copy'):
                    reference = project('legacy_copy')
                assert reference is not None, 'Current column dispatch is not pinned eager; no layout attribution'
                for name in VARIANTS:
                    with column_layout(name):
                        actual = project(name)
                        assert actual is not None, 'Column dispatch changed during layout ablation'
                        assert all(torch.equal(a, b) for a, b in zip(actual, reference)), (label, name)
                        projection_checks[label + '/' + name] = True
                        del actual
                del reference
                projection_times[label] = paired_measure(project, barrier)
            case = dict(tokens=length, full_block=block_times, byte_exact=checks,
                        grouped_columns=projection_times, projection_byte_exact=projection_checks)
            record['cases'].append(case)
            save(path, record)
            print('TE_LAYOUT_CASE', rank, length,
                  json.dumps({name: round(t['median_wall_ms'], 3) for name, t in block_times.items()}), flush=True)
            # INT32 equality on trained TP2 weights, including row tails. No
            # dequantization or attention can hide accumulator differences.
            q = torch.randint(-128, 128, (length, 5120), device=x.device, dtype=torch.int8)
            for module, weight in micro_weights.items():
                expected = accumulator('legacy_copy')(q, weight)
                for name in VARIANTS:
                    actual = accumulator(name)(q, weight)
                    assert torch.equal(actual, expected), (module, length, name)
                    del actual
                timing = paired_measure(lambda name: accumulator(name)(q, weight), barrier)
                record['gemm_controls'].append(dict(tokens=length, output_channels=weight.shape[0],
                                                   int32_exact_all=True, timing=timing))
                del expected
            del q
            save(path, record)
            if length == 353:
                record['profile'] = profile_variants(run, x, barrier, args.output, rank)
                save(path, record)
            del x, mask, freqs, normed
        # Signed extremes exercise the complete INT8 range, not just trained
        # weight statistics. Maximum absolute sum is 83,886,080 < INT32_MAX.
        codes = torch.full((33, 5120), -128, device='cuda', dtype=torch.int8)
        weight = torch.full((32, 5120), -128, device='cuda', dtype=torch.int8)
        expected = torch.full((33, 32), 5120 * 128 * 128, device='cuda', dtype=torch.int32)
        for name in VARIANTS:
            assert torch.equal(accumulator(name)(codes, weight), expected), name
        record.update(status='complete', signed_extremes_int32_exact=True)
    except BaseException as exc:
        record.update(status='failed', error=repr(exc))
        raise
    finally:
        save(path, record)
        if not local:
            dist.destroy_process_group()


@torch.no_grad()
def worker(args):
    import comfy_kitchen as ck
    # SM120 selects a different Kitchen column path. Force the pinned eager
    # columns only for the labeled local preview; T4 keeps real dispatch.
    with ck.registry.use_backend('eager') if args.local else nullcontext():
        return _worker(args)


def batch(args):
    """Validate frozen APIs, fetch only two public shards, supervise TP owners."""
    from patches.legacy_runtime import validate_packages, validate_source
    validate_packages()
    validate_source(args.source / 'ComfyUI')
    state = dict(status='starting', full_te_encoder=False, full_h3_pipeline=False,
                 torch=torch.__version__, cuda=torch.version.cuda,
                 kitchen=md.version('comfy-kitchen'), tilelang=md.version('tilelang'),
                 profiler_tools={name: shutil.which(name) for name in ('nsys', 'ncu')})
    path = args.output / 'result.json'
    save(path, state)
    try:
        hardware = [dict(name=torch.cuda.get_device_name(i),
                         capability=list(torch.cuda.get_device_capability(i)),
                         uuid=str(torch.cuda.get_device_properties(i).uuid))
                    for i in range(torch.cuda.device_count())]
        assert len(hardware) == 2 and len({d['uuid'] for d in hardware}) == 2
        assert all(d['capability'] == [7, 5] and 'T4' in d['name'] for d in hardware)
        state['hardware'] = hardware
        nvcc = Path(shutil.which('nvcc') or '')
        assert nvcc.is_file() and 'release 12.8' in subprocess.check_output([str(nvcc), '--version'], text=True)
        os.environ.update(CUDACXX=str(nvcc.resolve()), CUDA_HOME=str(nvcc.resolve().parent.parent),
                          CUDA_PATH=str(nvcc.resolve().parent.parent), OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
                          NCCL_P2P_DISABLE='1', NCCL_MIN_NCHANNELS='4', NCCL_MAX_NCHANNELS='4', NCCL_DEBUG='INFO',
                          HF_HUB_DISABLE_XET='1', HF_HUB_DISABLE_PROGRESS_BARS='1', HF_HUB_DOWNLOAD_TIMEOUT='90')
        os.environ.pop('NCCL_PROTO', None)
        state['transport_environment'] = {name: os.environ.get(name) for name in
            ('NCCL_P2P_DISABLE', 'NCCL_MIN_NCHANNELS', 'NCCL_MAX_NCHANNELS', 'NCCL_PROTO', 'NCCL_IB_DISABLE')}
        plan = json.loads((args.source / 'benchmarks/review_t4_assets.json').read_text())
        assert plan['revision'] == 'd3e6a4b131056d2c626a01acf6ba48123a885783' and not plan['token_required']
        assets = Path('/kaggle/temp/h3_te_layout_assets')
        assets.mkdir(exist_ok=True)
        from huggingface_hub import hf_hub_download

        def download(rank):
            name = f'te.block00.rank{rank}.safetensors'
            expected = plan['objects'][name]
            file = Path(hf_hub_download(plan['repository'], name, revision=plan['revision'],
                                       token=False, local_dir=assets))
            with file.open('rb') as stream:
                actual = hashlib.file_digest(stream, 'sha256').hexdigest()
            assert file.stat().st_size == expected['bytes'] and actual == expected['sha256']
            return dict(name=name, **expected)

        state['status'] = 'download_two_public_shards'
        save(path, state)
        start = time.monotonic()
        with ThreadPoolExecutor(max_workers=2) as pool:
            objects = list(pool.map(download, (0, 1)))
        state['assets'] = dict(repository=plan['repository'], revision=plan['revision'],
                               objects=objects, seconds=time.monotonic() - start)
        state['status'] = 'operator_ablation'
        save(path, state)
        command = [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=2',
                   str(Path(__file__)), '--worker', '--assets', str(assets), '--output', str(args.output),
                   '--lengths', *map(str, args.lengths)]
        with (args.output / 'workers.log').open('w') as log:
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            for line in process.stdout:
                log.write(line)
                log.flush()
                if line.startswith(('TE_LAYOUT', 'Traceback')):
                    print(line, end='', flush=True)
            code = process.wait()
        state['worker_exit_code'] = code
        ranks = [json.loads((args.output / f'rank{rank}.json').read_text()) for rank in (0, 1)]
        state['ranks'] = ranks
        assert code == 0 and all(r['status'] == 'complete' and len(r['cases']) == len(args.lengths) for r in ranks)
        for left, right in zip(ranks[0]['cases'], ranks[1]['cases']):
            assert left['tokens'] == right['tokens']
            for name in VARIANTS:
                assert left['byte_exact'][name]['output_sha256'] == right['byte_exact'][name]['output_sha256']
        state['status'] = 'complete'
    except BaseException as exc:
        state.update(status='failed', error=repr(exc))
        (args.output / 'exception.txt').write_text(traceback.format_exc())
        raise
    finally:
        save(path, state)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path)
    parser.add_argument('--assets', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--local', action='store_true')
    parser.add_argument('--lengths', type=int, nargs='+', choices=LENGTHS, default=LENGTHS)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.worker or args.local:
        worker(args)
    else:
        batch(args)
