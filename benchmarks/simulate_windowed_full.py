"""Full 50-layer Native entry points with a shared TE/DiT residency window.

Two logical ranks on the local GPU; compact Ref2VA fixture, no performance or
vision-tower/sampler claims. Uses the same attach_window API as integration.
"""
import argparse
import hashlib
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT/'ComfyUI', ROOT/'.retest-deps', ROOT/'benchmarks'):
    sys.path.insert(0, str(path))

import torch
import retest
from full_models import load_full, skeleton, forward, stage_dit_aux
from ref2va_workloads import request
from logical_dual_gpu import LogicalDualGPU, isolate_cublas_workspaces
from patches.optimized_tp import enable
from patches.windowed_tp import CheckpointReader, tp_specs, attach_window, DiTPreprocessing
from patches.weight_residency import ResidencyWindow, WeightSpec, module_tensors
import patches.tp_qwen3vl as te
import patches.tp_dit as dit

PATHS = {'te': Path('D:/minimaxH3/qwen3vl_32b_minimax_h3_int8_convrot.safetensors'),
         'dit': Path('D:/minimaxH3/MiniMax_H3_Ref2VA_pruned_int8_convrot.safetensors')}
GATE = {'rel_l2_max': 1e-5, 'cosine_min': .999999, 'max_abs_max': .005}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--layers', type=int, default=50)
    parser.add_argument('--adaptive', action='store_true')
    parser.add_argument('--include-video', action='store_true')
    options = parser.parse_args()
    if not 1 <= options.layers <= 50:
        raise ValueError('Expected 1–50 trained layers')
    target = ROOT/f'semantic_results/windowed_full_{options.layers}{"_video" if options.include_video else ""}.json'
    paths = list((ROOT/'patches').glob('*.py')) + [Path(__file__), ROOT/'benchmarks/full_models.py',
        ROOT/'benchmarks/logical_dual_gpu.py', ROOT/'benchmarks/ref2va_workloads.py']
    fingerprints = {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    rows = []
    def save(passed=False, **extra):
        retest.write_json(target, {'passed': passed, 'cases': rows, 'source_sha256': fingerprints,
            'scope': f'Local logical TP2, {options.layers} trained TE/DiT layers, Native outer forward, shared request-wide weight window; no physical performance evidence', **extra})
    save(status='running')
    enable('scalemax')
    import comfy_kitchen as ck
    ck.backends.cuda._max_dynamic_shared_memory_per_block = lambda x: 49152
    torch.backends.cuda.matmul.allow_tf32 = False
    isolate_cublas_workspaces()
    torch.set_grad_enabled(False)
    text_data = request('te', 512, 1700, 'cuda:0')
    with torch.inference_mode():
        print('FULL reference TE', options.layers, flush=True)
        native = load_full('te', PATHS['te'], 'streamed', layers=options.layers)
        text_ref = forward('te', native, text_data).cpu()
        del native
        retest.cleanup()
        data = request('dit', 68, 1700, 'cuda:0')
        args, kwargs = data
        args = (args[0], args[1], text_ref[:, :32].float().cuda())
        print('FULL reference DiT', options.layers, flush=True)
        native = load_full('dit', PATHS['dit'], 'streamed', layers=options.layers)
        # Production Native refines fixed text once per sampling run.
        refined = native.preprocess_text_embeds(args[2]).cpu()
        data = ((args[0], args[1], refined.cuda()), kwargs)
        denoiser_ref = forward('dit', native, data).cpu()
        del native
        retest.cleanup()
        video_path = Path('D:/minimaxH3/minimax_h3_video_vae_int8_convrot.safetensors')
        video_ref = None
        if options.include_video:
            template = data[0][0][0]
            # A deterministic latent update composes the entry points. This
            # is a test fixture, not a claim to execute a production sampler.
            decode_latent = (template.cpu() - .1*denoiser_ref[:template.numel()].reshape(template.shape)).half()
            with torch.inference_mode(False), torch.no_grad():
                native, sd = retest.load_video_native(video_path, 'cuda:0')
            del sd
            if not hasattr(native.decoder.transformer_blocks[0].attn.to_qkv.weight, '_qdata'):
                raise RuntimeError('VAE comparator changed native INT8 dispatch')
            video_ref = native.decode(decode_latent.cuda()).cpu()
            del native
            retest.cleanup()
        comm = LogicalDualGPU(timeout=180)
        te.dist = dit.dist = comm
        windows, text_models, denoisers = [], [], []
        for rank in range(2):
            text_specs = tp_specs(CheckpointReader(PATHS['te']), 'te', rank, options.layers)
            dit_specs = tp_specs(CheckpointReader(PATHS['dit']), 'dit', rank, options.layers)
            model = skeleton('dit', PATHS['dit'], 'cuda:0')
            preprocess = DiTPreprocessing(model, rank, 'cuda:0', comm, str(PATHS['dit']))
            aux_specs = preprocess.specs()
            video_specs = []
            if options.include_video:
                def prepare_video():
                    vae, sd = retest.load_video_native(video_path, 'cpu')
                    del sd
                    return vae
                video_specs = [WeightSpec(('video', rank), prepare_video,
                                         video_path.stat().st_size + 64*2**20)]
            specs = text_specs + aux_specs + dit_specs + video_specs
            keys = [s.key for s in specs]
            window = ResidencyWindow(specs, keys, 'cuda:0',
                max_weight_bytes=4096*2**20, pinned_bytes=6144*2**20,
                lookahead=2, reserve_bytes=256*2**20, adaptive=options.adaptive)
            preprocess.window = window
            windows.append(window)
            text_models.append(attach_window(skeleton('te', PATHS['te'], 'cuda:0'),
                                              'te', window, [s.key for s in text_specs]))
            denoisers.append(attach_window(model, 'dit', window, [s.key for s in dit_specs]))
        def run(rank):
            for t in module_tensors(text_models[rank]):
                if t.is_cuda:
                    t.record_stream(torch.cuda.current_stream())
            out = forward('te', text_models[rank], text_data)
            text = out.cpu()
            text_models[rank] = None  # Backed by checkpoint; no D2H weight copy.
            conditioning = denoisers[rank].preprocess_text_embeds(out[:, :32].float())
            del out
            print('FULL window TE rank', rank, 'done', flush=True)
            native_args, native_options = data
            step_data = ((native_args[0], native_args[1], conditioning), native_options)
            denoised = forward('dit', denoisers[rank], step_data).cpu()
            print('FULL window DiT rank', rank, 'done', flush=True)
            pixels = None
            if options.include_video:
                template = step_data[0][0][0]
                latent = (template - .1*denoised[:template.numel()].reshape(template.shape).cuda()).half()
                with windows[rank].use(('video', rank)) as vae:
                    pixels = vae.decode(latent).cpu()
                print('FULL window Video rank', rank, 'done', flush=True)
            return text, denoised, pixels
        try:
            outputs = comm.run([lambda r=r: run(r) for r in range(2)])
            for rank, pair in enumerate(outputs):
                comparisons = [('te', pair[0], text_ref), ('dit', pair[1], denoiser_ref)]
                if options.include_video:
                    comparisons.append(('video', pair[2], video_ref))
                for kind, output, reference in comparisons:
                    score = retest.metrics(output, reference)
                    score.update(kind=kind, rank=rank, passed=retest.passes(score, GATE))
                    rows.append(score)
                    save()
                    print('FULL', kind, rank, score['passed'], score['is_bit_exact'], flush=True)
                    if not score['passed']:
                        raise RuntimeError('Native full entry mismatch')
            stats = [window.finish() for window in windows]
            save(passed=True, status='complete', gate=GATE, residency=stats,
                 limitations=['Compact S68 Ref2VA with precomputed fixture reference conditioning',
                              'Both comparators execute Native once-per-request text refinement',
                              'Vision tower, production sampler loop and physical transport excluded',
                              'Optional video uses full 36-layer public Native decode with tiling disabled at compact geometry'])
        finally:
            for window in windows:
                window.close()


if __name__ == '__main__':
    main()
