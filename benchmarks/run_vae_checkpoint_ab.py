"""Two checkpoint decodes of one immutable latent; no TE/DiT sampling."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'ComfyUI'), str(ROOT/'benchmarks')]


def export_video(pixels, output):
    import av
    import numpy as np
    from fractions import Fraction
    from PIL import Image
    frames = pixels[0] if pixels.ndim == 5 else pixels
    snapshots = {0, 48, 96, 144, 192, 240, 288, 336, len(frames)-1}
    output.mkdir(exist_ok=True)
    with av.open(str(output/'video.mp4'), 'w') as container:
        stream = container.add_stream('libx264', rate=24)
        stream.width, stream.height = frames.shape[2], frames.shape[1]
        stream.pix_fmt = 'yuv420p'
        stream.options = {'crf': '12', 'preset': 'fast'}
        for i, frame in enumerate(frames):
            rgb = np.round(frame.float().clamp(0, 1).numpy()*255).astype(np.uint8)
            if i in snapshots:
                Image.fromarray(rgb).save(output/f'frame_{i:04d}.png')
            image = av.VideoFrame.from_ndarray(rgb, format='rgb24')
            image.pts, image.time_base = i, Fraction(1, 24)
            for packet in stream.encode(image):container.mux(packet)
        for packet in stream.encode():container.mux(packet)
    return dict(frames=len(frames), fps=24, video_bytes=(output/'video.mp4').stat().st_size,
                snapshots=sorted(snapshots), encoder='libx264 crf12 yuv420p; matched lossless PNG snapshots')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--latents', type=Path, required=True)
    parser.add_argument('--models', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False)
    import torch
    import comfy.cli_args
    comfy.cli_args.args.cpu = False
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = torch.backends.cudnn.deterministic = False
    from vae_decoder_service import DecoderService
    from patches.staged_vae import StagedVAE
    from patches.comfy_sampler import ParallelDecodeHandoff
    from patches.host_weight_cache import GiB
    z = torch.load(args.latents, map_location='cpu', weights_only=True)['video']
    digest = hashlib.sha256(z.contiguous().view(torch.uint8).numpy()).hexdigest()
    assert digest == '8fb3296f9dfc00641698e1a1aa26acb7f0fa98675b89b597dc95f10e64a9fa64'
    assert list(z.shape) == [1, 24, 107, 34, 60] and torch.isfinite(z).all()
    manifest = json.loads(args.models.read_text())
    state = dict(status='starting', input_tensor_sha256=digest, cases=[],
                 scope='Native checkpoint-level VAE A/B; weight and native attention dispatch may differ',
                 full_pipeline=False, host_budget_bytes=24*GiB)
    def save(): (args.output/'result.json').write_text(json.dumps(state, indent=2))
    save()
    try:
        for label in ('int8_convrot', 'fp16'):
            service = vae = None
            try:
                service = DecoderService(host_bytes=24*GiB, timeout_seconds=900)
                vae = StagedVAE(Path(manifest[label]['local_path']), service)
                wrapper = ParallelDecodeHandoff(vae, service, Path(manifest[label]['local_path']), 'temporal')
                start = time.monotonic()
                pixels = wrapper.decode(z).cpu()
                torch.cuda.synchronize()
                seconds = time.monotonic()-start
                assert wrapper.last_strategy == 'native_temporal_pairs' and wrapper.last_fallback is None
                assert list(pixels.shape) == [1, 362, 544, 960, 3]
                assert all(bool(torch.isfinite(part).all()) for part in pixels[0].split(8))
                row = dict(checkpoint=label, decode_seconds=seconds, shape=list(pixels.shape), finite=True,
                           strategy=wrapper.last_strategy, fallback=wrapper.last_fallback,
                           worker_stats=service.call('stats'), events=vae.events)
                row['constructed_attention'] = [command['attention'] for worker in row['worker_stats']
                    for command in worker['commands'] if command.get('attention')]
                assert len(row['constructed_attention']) == 2
                expected_unquantized = [] if label == 'int8_convrot' else list(range(36))
                assert all(a['unquantized_blocks'] == expected_unquantized for a in row['constructed_attention'])
                row['media'] = export_video(pixels, args.output/label)
                state['cases'].append(row); save()
                print('VAE_AB_CASE', label, seconds, flush=True)
                del pixels
            finally:
                if vae is not None:vae.release()
                if service is not None:service.release();service.close()
                del vae, service
                gc.collect();torch.cuda.empty_cache()
        assert len(state['cases']) == 2
        def tiles(case):
            return [[t for c in worker['commands'] for t in c['native_tiles']][-1000:]
                    for worker in case['worker_stats']]
        state['identical_realized_tiles'] = tiles(state['cases'][0]) == tiles(state['cases'][1])
        state['tiling_comparison_note'] = 'Inspect any OOM/retry or geometry difference before attributing quality to checkpoint'
        state['status'] = 'complete'
    except BaseException as error:
        state.update(status='failed', error=repr(error))
        (args.output/'exception.txt').write_text(traceback.format_exc())
        raise
    finally:save()


if __name__ == '__main__': main()
