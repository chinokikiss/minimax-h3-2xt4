"""Validate saved T4 E2E evidence without changing it or running a GPU job."""
import argparse
import hashlib
import json
from pathlib import Path

from benchmarks.anime_e2e_contract import validate
from benchmarks.e2e_attention_validation import denoiser_attention_counts


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def finite(summary):
    if isinstance(summary, list):
        return bool(summary) and all(finite(item) for item in summary)
    return isinstance(summary, dict) and summary.get('finite') is True


def verify(outputs, media_check):
    generation = outputs / 'turbo_e2e/generation'
    result_path = generation / 'result.json'
    result = read(result_path)
    batch = read(outputs / 'turbo_e2e/batch_results.json')
    setup = read(outputs / 'light_setup/setup_receipt.json')
    delivery = read(outputs / 'light_setup/delivery_binding.json')
    assert setup['capsule_sha256'] == delivery['capsule_sha256']
    assert setup['status'] == 'ready_for_device_gate'
    config = result['config']
    validate(config, config['reference_images'])
    devices = batch['hardware']['devices']
    assert len(devices) == 2 and len({d['uuid'] for d in devices}) == 2
    assert all(d['capability'] == [7, 5] and 'T4' in d['name'] for d in devices)
    assert batch['hardware']['kitchen'] == '0.2.37' and batch['hardware']['tilelang'] == '0.1.13'
    assert not batch['online_lora'] and batch['request_count'] == 1
    assert batch['persistent_bake']['verified_native_targets'] == 208
    assert not batch['persistent_bake']['inference_remerge']
    assert not batch['persistent_bake']['online_lora']
    assert any(c['stage'] == 'offline_disk_bake' and c['exit_code'] == 0 for c in batch['commands'])
    assert all(stage['status'] == 'complete' for stage in result['stages'].values())
    assert set(result['stages']) == {'vae_metadata', 'reference_and_TE', 'sampler',
                                    'video_decode', 'audio_decode', 'file_export'}
    assert len(result['denoiser_outputs']) == len(result['sampler_steps']) == config['steps']
    assert all(finite(row) for row in result['denoiser_outputs'])
    assert all(finite(row['denoised']) and finite(row['latent']) for row in result['sampler_steps'])
    assert all(row['finite'] for row in result['outputs'].values())
    assert result['decoded_shapes']['pixels'] == [1, 243, 672, 1152, 3]
    assert result['decoded_shapes']['waveform'] == [1, 324000, 2]
    assert result['decode']['video'] == dict(strategy='native_temporal_pairs', fallback=None)
    assert result['decode']['audio'] == dict(strategy='native', fallback=None)
    assert result['vae_before_sampling']['audio']['read_bytes'] == 0
    assert len(result['ranks']) == 2 and {rank['rank'] for rank in result['ranks']} == {0, 1}
    ranks = []
    for rank in result['ranks']:
        benchmark = rank['benchmark']
        calls = benchmark['executed_backends']
        counts = denoiser_attention_counts(benchmark)
        assert counts['total'] == 50 * config['steps']
        assert len(counts['per_step']) == config['steps']
        assert all(count == 50 for count in counts['per_step'].values())
        assert calls['TE/column_view'] == 250 and calls['TE/kitchen_causal'] == 50
        assert calls['current/fused_swiglu_quant'] and calls['current/scaled_fp16_epilogue']
        assert not any(value for name, value in calls.items() if name.startswith('LoRA/'))
        assert not benchmark['online_lora_attached']
        assert rank['bundle_preparation']['dit']['disk_baked']
        assert rank['bundle_preparation']['dit']['merged_matrices'] == 0
        peak = max(c['peak_allocated'] for c in benchmark['commands'] if c['command'] == 'denoise')
        assert peak < 14.5 * 2**30
        assert benchmark['layouts']
        ranks.append(dict(rank=rank['rank'], attention=counts, executed_backends=calls,
                          layouts=benchmark['layouts'], denoise_peak_allocated_bytes=peak))
    if result['status'] != 'complete':
        # Recover only this demonstrated post-export counter assertion. Other
        # failures retain their original failure status and are not accepted.
        error = (generation / 'exception.txt').read_text(encoding='utf-8')
        assert result['error'] == 'AssertionError()'
        assert 'assert kitchen_denoiser_calls(bench)' in error
        assert all(r['attention']['outside_command'] == 2016 for r in ranks)
    for name, key in [('video.mp4', 'video_bytes'), ('audio.wav', 'audio_bytes')]:
        assert (generation / name).stat().st_size == result['media'][key]
    assert media_check['video']['full_ffmpeg_decode_exit_code'] == 0
    assert media_check['video']['decoded_frames'] == 243
    assert media_check['video']['fps'] == 24
    assert media_check['audio']['frames'] == 324000 and media_check['audio']['sample_rate'] == 32000
    assert media_check['audio']['channels'] == 2
    assert media_check['audio']['peak'] > 0 and media_check['audio']['exact_pcm_clip_fraction'] == 0
    hashes = {name: hashlib.sha256((generation / name).read_bytes()).hexdigest()
              for name in ['result.json', 'video.mp4', 'audio.wav']}
    return dict(status='saved_e2e_checks_passed_observer_assertion_reconciled',
                original_remote_status=result['status'], original_remote_error=result.get('error'),
                original_files_modified=False, gpu_rerun=False, hardware=batch['hardware'],
                completed_steps=config['steps'], decoded_shapes=result['decoded_shapes'],
                decode=result['decode'], ranks=ranks, hashes=hashes,
                stage_seconds={k: (v['end_ns'] - v['start_ns']) / 1e9 for k, v in result['stages'].items()},
                foreground_seconds=result['foreground_seconds'],
                offline_bake_seconds=next(c['seconds'] for c in batch['commands'] if c['stage'] == 'offline_disk_bake'),
                media_check=media_check,
                quality_acceptance='Pending full video/audio review; no baseline equivalence or speedup claim')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--outputs', type=Path, required=True)
    parser.add_argument('--media-check', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.resolve().is_relative_to(args.outputs.resolve()), 'Preserve original collected evidence'
    result = verify(args.outputs, read(args.media_check))
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
