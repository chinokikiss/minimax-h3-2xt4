"""Run the actual two-stream VAE wrappers with independent coordinate decoders."""
import json
from pathlib import Path
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "ComfyUI"))
sys.path.insert(0, str(ROOT / ".retest-deps"))
import torch


def main():
    checks = []
    from patches.multi_gpu_video_vae import MultiGPUVideoVAEDecoder
    from patches.multi_gpu_audio_vae import MultiGPUAudioVAEDecoder
    from comfy.ldm.minimax.vae import MiniMaxH3VideoVAE
    class VideoOracle(torch.nn.Module):
        vae_ratio = 16
        blend = MiniMaxH3VideoVAE.blend
        def _decode_pixels(self, z):
            return z[:, :3].repeat_interleave(4, 2).repeat_interleave(16, 3).repeat_interleave(16, 4)
    class AudioOracle(torch.nn.Module):
        samples_per_latent = 2
        def decode(self, z):
            return z[:, 0].repeat_interleave(2, -1)
    # CPU geometry checks do not pretend to exercise physical CUDA behavior.
    with patch("torch.cuda.is_available", return_value=False):
        video = MultiGPUVideoVAEDecoder(VideoOracle(), dev0="cpu", dev1="cpu", tile_size=32,
                                       tile_overlap_min=16, dtype=torch.float32)
        audio = MultiGPUAudioVAEDecoder(AudioOracle(), dev0="cpu", dev1="cpu", dtype=torch.float32)
        for ts, ov in [(0, 16), (32, 0), (32, 32), (32, 48), (33, 16), (32, -16)]:
            try:
                video.split_tiles(80, ts, ov)
            except ValueError:
                checks.append(f"video_invalid_geometry_{ts}_{ov}")
            else:
                raise AssertionError("Invalid spatial tile plan accepted")
        audio_z = torch.arange(2 * 32 * 2 * 21).reshape(2, 32, 2, 21).float()
        for chunk, overlap in [(0, 1), (10, 0), (10, 10), (10, 11), (10, -1)]:
            for method in [audio.decode_tiled_single, audio.decode_tiled_multi]:
                try:
                    method(audio_z, chunk, overlap)
                except ValueError:
                    checks.append(f"{method.__name__}_invalid_{chunk}_{overlap}")
                else:
                    raise AssertionError("Invalid temporal chunk plan accepted")
        wav, _ = audio.decode_tiled_single(audio_z, 10, 3)
        assert torch.allclose(wav, audio.vae_dev0.decode(audio_z), atol=.00025, rtol=0)
        checks.append("audio_batch_stereo_tail_coordinates")
        for height, width in [(3, 5), (4, 6), (1, 5), (3, 1)]:
            z = torch.arange(2 * 3 * 2 * height * width).reshape(2, 3, 2, height, width).float()
            out, _ = video.decode_tiled_single(z)
            assert torch.equal(out, video.vae_dev0._decode_pixels(z))
            checks.append(f"video_cpu_coordinate_oracle_{height}_{width}")

    # Two logical ranks on cuda:0 exercise the actual thread/queue wrappers.
    video = MultiGPUVideoVAEDecoder(VideoOracle(), dev1="cuda:0", tile_size=32,
                                   tile_overlap_min=16, dtype=torch.float32)
    audio = MultiGPUAudioVAEDecoder(AudioOracle(), dev1="cuda:0", dtype=torch.float32)
    for wrapper, replica in [(video, VideoOracle()), (audio, AudioOracle())]:
        wrapper.vae_dev1 = replica.cuda()
        wrapper.is_multi_gpu = True
        wrapper.stream0 = torch.cuda.Stream(device="cuda:0")
        wrapper.stream1 = torch.cuda.Stream(device="cuda:0")
    z = torch.arange(2 * 3 * 2 * 3 * 5).reshape(2, 3, 2, 3, 5).float()
    sequential, _ = video.decode_tiled_single(z)
    parallel, _ = video.decode_tiled_multi(z)
    assert torch.equal(sequential, parallel)
    checks.append("video_two_logical_rank_tile_coordinates")
    audio_z = torch.arange(2 * 32 * 2 * 21).reshape(2, 32, 2, 21).float()
    channels, _ = audio.decode_channel_parallel(audio_z)
    assert torch.equal(channels, audio.vae_dev0.decode(audio_z.cuda()))
    checks.append("audio_two_logical_rank_stereo_order")
    sequential, _ = audio.decode_tiled_single(audio_z, 10, 3)
    parallel, _ = audio.decode_tiled_multi(audio_z, 10, 3)
    assert torch.equal(sequential, parallel)
    checks.append("audio_two_logical_rank_temporal_coordinates")
    output = ROOT / "semantic_results/vae_scheduling.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"passed": True, "cases": len(checks), "checks": checks,
                                  "scope": "coordinate-oracle wrapper simulation; one physical GPU, no trained VAE weights"}, indent=2))
    print(f"PASS {len(checks)} VAE scheduling simulations; saved {output}")


if __name__ == "__main__":
    main()
