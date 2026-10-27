"""Real-latent decoder-only Query versus Temporal comparison on two SM75 T4s."""
import argparse
import gc
import gzip
import hashlib
import importlib.metadata as md
import json
from pathlib import Path
import statistics
import sys
import threading
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'ComfyUI'), str(ROOT/'benchmarks')]
import torch


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def pixels_check(actual, reference=None):
    digest, finite, maxima = hashlib.sha256(), True, []
    for frames in actual[0].split(8):
        finite &= bool(torch.isfinite(frames).all())
        digest.update(frames.contiguous().view(torch.uint8).numpy())
    if reference is not None:
        for a, b in zip(actual[0], reference[0]):
            maxima.append(float((a-b).abs().max()))
    return dict(sha256=digest.hexdigest(), finite=finite, per_frame_max_abs=maxima,
                bit_exact=torch.equal(actual, reference) if reference is not None else None,
                max_abs=max(maxima, default=0.))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', type=Path, required=True)
    parser.add_argument('--latents', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 2:
        raise ValueError('Repeated steady-state measurements required')
    args.output.mkdir(parents=True, exist_ok=True)
    state = dict(status='starting', scope='Video decoder only; no TE, DiT, LoRA, Audio VAE or full H3 sampling',
        weights_retained_between_repeats=True, cases=[], cold_loads=[], warmup=[], requests=[],
        controls=[], defaults_changed=False, source_sha256={n: sha(ROOT/n) for n in (
            'patches/temporal_vae.py','patches/parallel_vae.py','patches/video_decode_planner.py',
            'patches/comfy_sampler.py','patches/comfy_tp_runtime.py',
            'benchmarks/vae_decoder_service.py','benchmarks/run_vae_temporal_compare.py')})
    def save():
        (args.output/'result.json').write_text(json.dumps(state, indent=2))
    save()
    stopped, memory = threading.Event(), []
    service = vae = None
    try:
        from patches.legacy_runtime import validate_source, validate_packages
        validate_source(ROOT/'ComfyUI');validate_packages()
        devices = [dict(name=torch.cuda.get_device_name(i), capability=list(torch.cuda.get_device_capability(i)),
                        uuid=str(torch.cuda.get_device_properties(i).uuid)) for i in range(torch.cuda.device_count())]
        assert len(devices) == 2 and len({d['uuid'] for d in devices}) == 2
        assert all('T4' in d['name'] and d['capability'] == [7,5] for d in devices)
        state['hardware'] = dict(devices=devices, torch=torch.__version__, cuda=torch.version.cuda,
                                  kitchen=md.version('comfy-kitchen'), tilelang=md.version('tilelang'))
        torch.set_grad_enabled(False)
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = torch.backends.cudnn.deterministic = False
        z = torch.load(args.latents, map_location='cpu', weights_only=True)['video']
        state['input'] = dict(shape=list(z.shape), dtype=str(z.dtype), file_sha256=sha(args.latents),
            tensor_sha256=hashlib.sha256(z.contiguous().view(torch.uint8).numpy()).hexdigest())
        assert list(z.shape) == [1,24,107,34,60]
        assert state['input']['tensor_sha256'] == '8fb3296f9dfc00641698e1a1aa26acb7f0fa98675b89b597dc95f10e64a9fa64'
        assert sha(args.video) == '52a2c8c73583c86e4f41cdcce3a6ad0ea562987bc0bf3d60a0cef5f5c8e60c0e'
        import pynvml as nv
        nv.nvmlInit()
        handles = [nv.nvmlDeviceGetHandleByIndex(i) for i in (0,1)]
        def sample():
            while not stopped.wait(.2):
                memory.append(dict(time_ns=time.monotonic_ns(), devices=[dict(device=i,
                    used=nv.nvmlDeviceGetMemoryInfo(h).used, total=nv.nvmlDeviceGetMemoryInfo(h).total,
                    utilization=nv.nvmlDeviceGetUtilizationRates(h).gpu) for i,h in enumerate(handles)]))
        threading.Thread(target=sample, daemon=True).start()
        from vae_decoder_service import DecoderService
        from patches.staged_vae import StagedVAE
        from patches.comfy_sampler import ParallelDecodeHandoff
        service = DecoderService()
        vae = StagedVAE(args.video, service)
        wrappers = {mode: ParallelDecodeHandoff(vae, service, args.video, mode) for mode in ('query','temporal')}
        # Same-hardware Native reference on a real cropped clip, not FP32 weights.
        control = z[:,:,:7,:16,:16].contiguous()
        native = vae.decode(control).cpu()
        for mode, wrapper in wrappers.items():
            out = wrapper.decode(control).cpu()
            check = pixels_check(out, native)
            state['controls'].append(dict(mode=mode, check=check))
            assert check['finite'], check
            save()
        del native, out, control
        references = {}
        # Both paths warm the actual full-resolution tile GEMM/attention shapes.
        for mode in ('query','temporal'):
            start = time.monotonic()
            out = wrappers[mode].decode(z[:,:,:12]).cpu()
            torch.cuda.synchronize()
            check = pixels_check(out, references.get('warmup'))
            if mode == 'query':
                references['warmup'] = out
            assert check['finite'], check
            state['warmup'].append(dict(mode=mode, seconds=time.monotonic()-start, check=check))
            save();print('VAE_COMPARE_WARMUP', mode, flush=True)
        references.clear();gc.collect()
        # Alternate the starting backend between repeated pairs. Cold replica
        # loads are separate; each timed request uses resident decoder weights.
        for repeat in range(args.repeats):
            order = ('query','temporal') if repeat % 2 == 0 else ('temporal','query')
            for mode in order:
                wrapper = wrappers[mode]
                if service.resident_mode != mode:
                    service.release()
                    template = vae.prepare_remote_decode()
                    memory_estimate = vae.memory_used_decode(z.shape, torch.float16)
                    numeric = dict(matmul_tf32=False,cudnn_tf32=False,cudnn_benchmark=False,cudnn_deterministic=False)
                    begin = time.monotonic()
                    prepared = service.call('vae_prepare',str(args.video),'video',int(memory_estimate),numeric,mode)
                    if not prepared['accepted']:
                        raise RuntimeError('Decoder admission refused: ' + prepared['reason'])
                    state['cold_loads'].append(dict(mode=mode, seconds=time.monotonic()-begin, admission=prepared.get('admission')))
                    del template
                start_ns = time.monotonic_ns()
                output = wrapper.decode(z).cpu()
                torch.cuda.synchronize()
                end_ns = time.monotonic_ns()
                row = dict(mode=mode, repeat=repeat, start_ns=start_ns, end_ns=end_ns,
                    seconds=(end_ns-start_ns)/1e9, strategy=wrapper.last_strategy, fallback=wrapper.last_fallback,
                    shape=list(output.shape), check=pixels_check(output,references.get('full')))
                assert wrapper.last_fallback is None
                assert row['strategy'] == ('native_aligned_query' if mode=='query' else 'native_temporal_pairs')
                assert row['check']['finite'], row
                if 'full' not in references:
                    references['full'] = output
                state['cases'].append(row)
                state['worker_stats'] = service.call('stats')
                state['parent_commands'] = service.parent_commands
                save();print('VAE_COMPARE_CASE', mode, repeat, round(row['seconds'],3), flush=True)
                del output;gc.collect()
        state['summary'] = {mode:dict(samples=[r['seconds'] for r in state['cases'] if r['mode']==mode],
            median_seconds=statistics.median(r['seconds'] for r in state['cases'] if r['mode']==mode)) for mode in wrappers}
        state['temporal_speedup_over_query'] = state['summary']['query']['median_seconds']/state['summary']['temporal']['median_seconds']
        state['all_outputs_byte_exact'] = all(r['check']['bit_exact'] is not False for r in state['controls']+state['warmup']+state['cases'])
        state['worker_stats'] = service.call('stats')
        state['parent_commands'] = service.parent_commands
        state['status'] = 'complete'
    except BaseException as error:
        state.update(status='failed',error=repr(error))
        (args.output/'exception.txt').write_text(traceback.format_exc())
        raise
    finally:
        try:
            if vae is not None:
                vae.release()
            if service is not None:
                service.release();service.close()
        finally:
            stopped.set()
            (args.output/'memory_samples.json.gz').write_bytes(gzip.compress(json.dumps(memory).encode()))
            save()


if __name__ == '__main__':
    main()
