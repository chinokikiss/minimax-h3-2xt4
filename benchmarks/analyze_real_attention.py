"""Real-distribution error, fixed-noise pipeline and temporal comparison report."""
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import sys
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
OUT=Path('D:/attention_eval/tokyo_ref2va_20261013')
sys.path.insert(0,json.loads((ROOT/'bench_assets/tokyo_ref2va.json').read_text())['comfy_root'])


def main():
    errors=json.loads((OUT/'attention_errors.json').read_text())
    video=json.loads((OUT/'attention_errors_video.json').read_text())
    assert errors['status']==video['status']=='complete'
    rows=[r for r in errors['records'] if r['capture']['component']!='video']+video['records']
    groups=defaultdict(list)
    for row in rows:
        for mode,parts in row['modes'].items():
            for part in parts:groups[(row['capture']['component'],mode)].append(part)
    summary=[]
    for (component,mode),parts in groups.items():
        head_values=np.array([v for p in parts for v in p['error']['relative_l2']])
        squared=sum(sum(p['error']['squared_error']) for p in parts)
        reference=sum(sum(p['error']['squared_reference']) for p in parts)
        summary.append({'component':component,'mode':mode,'head_instances':len(head_values),
                        'median_head_relative_l2':float(np.median(head_values)),
                        'p95_head_relative_l2':float(np.percentile(head_values,95)),
                        'max_head_relative_l2':float(head_values.max()),
                        'energy_weighted_relative_l2':math.sqrt(squared/reference)})
    pipelines=[]
    original_noise=torch.load(OUT/'initial_noise.pt',map_location='cpu',weights_only=False)
    original_sigmas=torch.load(OUT/'sigmas.pt',map_location='cpu',weights_only=True)
    original_conditioning=torch.load(OUT/'conditioning.pt',map_location='cpu',weights_only=False)
    for mode,folder in [('native',OUT),('dense',OUT/'runs/dense'),('sparse',OUT/'runs/sparse')]:
        state=json.loads((folder/'run_state.json').read_text());assert state['status']=='complete'
        assert len(list(folder.glob('denoiser_call*.pt')))==20
        noise=torch.load(folder/'initial_noise.pt',map_location='cpu',weights_only=False)
        assert all(torch.equal(a,b) for a,b in zip(original_noise.unbind(),noise.unbind()))
        assert torch.equal(original_sigmas,torch.load(folder/'sigmas.pt',map_location='cpu',weights_only=True))
        conditioning=torch.load(folder/'conditioning.pt',map_location='cpu',weights_only=False)
        assert all(torch.equal(a,b) for a,b in zip(original_conditioning['latent'].unbind(),conditioning['latent'].unbind()))
        refs=conditioning['positive'][0][1]['minimax_refs']
        original_refs=original_conditioning['positive'][0][1]['minimax_refs']
        assert len(refs)==len(original_refs)==1
        assert torch.equal(refs[0]['latent'],original_refs[0]['latent'])
        profile=json.loads((folder/'attention_profile.json').read_text())
        item={'mode':mode,'stages':state['stages'],'initial_noise_equal':True,'sigma_schedule_equal':True,
              'initial_latent_equal':True,'reference_vae_latent_equal':True,
              'attention_ms':{c:sum(r['attention_ms'] for r in profile if r['component']==c) for c in ('te','dit','video')},
              'attention_calls':{c:sum(r['component']==c for r in profile) for c in ('te','dit','video')}}
        if mode!='native':
            frame=json.loads((folder/'frame_errors.json').read_text())
            assert len(frame)==90
            item['temporal']={name:{'mean_frame_relative_l2':float(np.mean([r['relative_l2'] for r in samples])),
                                    'mean_frame_psnr_db':float(np.mean([r['psnr_db'] for r in samples])),
                                    'mean_temporal_difference_rmse':float(np.mean([r['temporal_difference_rmse'] for r in samples if 'temporal_difference_rmse' in r]))}
                              for name,samples in [('before_2s',frame[:48]),('after_2s',frame[48:])]}
            # Conditioning drift isolates TE before the denoiser starts.
            cond=torch.load(folder/'conditioning.pt',map_location='cpu',weights_only=False)['positive'][0][0]
            native_cond=torch.load(OUT/'conditioning.pt',map_location='cpu',weights_only=False)['positive'][0][0]
            item['conditioning_relative_l2']=((cond-native_cond).float().norm()/native_cond.float().norm()).item()
        pipelines.append(item)
        route_file=folder/'route_profile.json'
        if route_file.exists():
            route_rows=json.loads(route_file.read_text())
            dit_density=[r['selected_density'] for r in route_rows if r['component']=='dit']
            if dit_density:item['mean_dit_selected_density']=float(np.mean(dit_density))
    result={'date':'2026-10-13','gpu':errors['gpu'],'captures':len(rows),'components':summary,'pipelines':pipelines,
            'video_recapture':json.loads((OUT/'video_chunks/validation.json').read_text()),
            'kernel_validation':json.loads((OUT/'real_kernel_validation.json').read_text()),
            'audio':json.loads((OUT/'audio_comparison.json').read_text())}
    (OUT/'summary.json').write_text(json.dumps(result,indent=2))
    lines=['# MiniMax H3 real attention distribution and pipeline study','',
           f"Completed on {errors['gpu']} using `C:/Users/Administrator/Documents/project/ComfyUI-master` and the trained checkpoints in `D:/minimaxH3`.",'',
           'Fixed workload: Tokyo neon reference; 640×352, 90 frames at 24 fps; 20 res_multistep/simple steps; CFG 1; seed 20261013; denoise 1; no negative conditioning or Turbo LoRA. Exact reference bytes, initial noise and sigma schedule are retained. All tensor artifacts are on D:.','',
           '## Real activation coverage','',
           'Native inference supplies every captured activation; operator replay never changes downstream inputs. TE layers 0/10/25/40/49, DiT layers 0/10/25/40/49 at denoiser calls 0/9/19, and VAE decoder layers 0/9/18/27/35 across every public decoder chunk are sampled. Every head and every query row in each capture is evaluated. This is stratified layer coverage, not every model layer. The remaining native layers still execute during capture and all layers execute in the full pipeline comparisons.','',
           f"Captures in the final evaluation: {len(rows)}. TE: 64 query/8 KV heads, 353 tokens, D128. DiT: 56 heads, 6814 tokens, D128. Decoder captures use 32 heads, D64. Recapture produced bit-identical native pixels: {result['video_recapture']['bit_equal']}.",'',
           '## Attention output error against native','',
           'Relative L2 below compares the attention output before its output projection. Median/p95/max are per-head errors over the captured layer/call/chunk instances. Energy-weighted relative L2 combines squared errors and reference energy over those instances. FP16 cast isolates input/output representation error; dense INT8 adds Q/K quantization; INT8 PV additionally quantizes probabilities and V. Sparse routes retain the complete text/reference/audio prefix for queries and keys.','',
           '| Component | Approach | Median head L2 | P95 head L2 | Worst head L2 | Energy-weighted L2 |',
           '|---|---|---:|---:|---:|---:|']
    for s in summary:
        lines.append(f"| {s['component']} | {s['mode']} | {100*s['median_head_relative_l2']:.3f}% | {100*s['p95_head_relative_l2']:.3f}% | {100*s['max_head_relative_l2']:.3f}% | {100*s['energy_weighted_relative_l2']:.3f}% |")
    lines+=['','## Full pipeline comparison','',
            'Native keeps floating-point attention. Dense changes TE, DiT and the Video VAE decoder to QK INT8 / PV FP16. Sparse uses the same dense INT8 TE/VAE and Sol pooled QK INT8 / PV FP16 with tau=0 in every DiT layer. These are experimental fixed-input comparisons, not quality-certified deployment defaults. The audio VAE is unchanged.','',
            '| Mode | Reference+TE s | Sampling s | Video decode s | TE backend s | DiT backend s | VAE backend s |',
            '|---|---:|---:|---:|---:|---:|---:|']
    for p in pipelines:
        vals=[p['stages'][s]['wall_seconds'] for s in ('reference_and_TE','native_sampling','native_video_decode')]
        vals += [p['attention_ms'][c]/1000 for c in ('te','dit','video')]
        lines.append('| '+p['mode']+' | '+' | '.join(f'{v:.3f}' for v in vals)+' |')
    lines+=['','These are single measured runs. Model streaming/loading and first kernel compilation remain in stage wall times; native capture also incurs disk-copy/hash overhead. Backend event totals include preparation and routing for TileLang runs, while the kernel profiler separates the fused attention kernel. Cold/warm cache effects and capture overhead prevent treating the wall ratios as accepted speedups.','',
            '## Deviation after two seconds','',
            'Frame 48 starts the >=2-second interval. Pixel L2/PSNR and temporal-difference RMSE compare raw decoded floating-point frames, not compressed MP4 files. Temporal-difference RMSE measures perturbation of adjacent-frame changes; it is not an optical-flow or semantic consistency guarantee.','',
            '| Mode | 0–2s mean frame L2 | >=2s mean frame L2 | 0–2s mean PSNR | >=2s mean PSNR | >=2s temporal-difference RMSE |',
            '|---|---:|---:|---:|---:|---:|']
    for p in pipelines[1:]:
        before,after=p['temporal']['before_2s'],p['temporal']['after_2s']
        lines.append(f"| {p['mode']} | {before['mean_frame_relative_l2']*100:.3f}% | {after['mean_frame_relative_l2']*100:.3f}% | {before['mean_frame_psnr_db']:.2f} dB | {after['mean_frame_psnr_db']:.2f} dB | {after['mean_temporal_difference_rmse']:.6f} |")
    lines+=['','## Evidence and limits','',
            '- Native/candidate videos, raw pixels, initial noise, conditioning, denoiser outputs and Q/K/V captures are retained in this directory.',
            '- `attention_errors.json` contains TE/DiT and first-chunk VAE replay; `attention_errors_video.json` expands the VAE to every decoder chunk. Input hashes are verified before replay.',
            '- `real_kernel_validation.json` checks both PV modes against a matching quantized reference on real difficult heads, separating implementation error from approximation error.',
            '- `kernel_profiles/` contains exported SM120 PTX and profiler traces/summaries. A profiling error file, if present, identifies a limitation rather than a successful trace.',
            '- `audio_comparison.json` reports unchanged-audio-VAE decoding of each pipeline’s generated audio latent; WAVs and waveform tensors are retained.',
            '- No synthetic Q/K/V substitutes or reduced-depth models are used in this study. No T4 performance or transfer claim follows from the local results.',
            '- Replay samples native trajectories. Full candidate trajectories accumulate errors and can leave that distribution; the separate end-to-end comparison measures this effect.',
            '- Head tail errors and growing late-frame deviations matter more than a low median alone. INT8 PV and sparsity require component/head/step-specific qualification before adoption.','']
    lines += ['## Visual interpretation','',
              'The contact sheet shows dense INT8 retaining the native camera framing much more closely. The uniform sparse run visibly changes the camera perspective and scene composition, with the difference growing after two seconds. This is a fidelity observation from the sampled frames; pixel and temporal-difference metrics do not establish an aesthetic quality score or prove the presence/absence of every flicker or scene cut.','',
              '## Audio and density','']
    for audio in result['audio']:
        lines.append(f"- {audio['mode']} audio waveform relative L2: {100*audio['relative_l2']:.3f}%; decode {audio['decode_seconds']:.3f} s. The raw stereo output has 120000 samples at 32000 Hz (3.75 s).")
    for p in pipelines[1:]:
        if 'mean_dit_selected_density' in p:
            lines.append(f"- {p['mode']} mean token-level DiT block density: {100*p['mean_dit_selected_density']:.2f}%.")
    lines+=['','## Files','',
            f"[Frame comparison]({OUT.as_posix()}/frame_comparison.png), [temporal error plot]({OUT.as_posix()}/temporal_errors.png), [native video]({OUT.as_posix()}/native_video.mp4), [dense video]({OUT.as_posix()}/runs/dense/native_video.mp4), [sparse video]({OUT.as_posix()}/runs/sparse/native_video.mp4). Audio WAVs are beside each video.",'',
            'Reference-image credit: [Neon-street-in-Tokyo-Japan.jpg](https://commons.wikimedia.org/wiki/File:Neon-street-in-Tokyo-Japan.jpg), Acediscovery, [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Native model preprocessing resizes the image; generated videos transform the reference.','']
    (OUT/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,1,figsize=(9,6),sharex=True)
    for p in pipelines[1:]:
        data=json.loads((OUT/'runs'/p['mode']/'frame_errors.json').read_text())
        seconds=[r['seconds'] for r in data]
        axes[0].plot(seconds,[r['relative_l2']*100 for r in data],label=p['mode'])
        axes[1].plot(seconds,[r.get('temporal_difference_rmse',np.nan) for r in data],label=p['mode'])
    for ax in axes:ax.axvline(2,color='black',linestyle='--',alpha=.5);ax.legend();ax.grid(alpha=.2)
    axes[0].set_ylabel('Frame relative L2 (%)');axes[1].set_ylabel('Temporal-difference RMSE');axes[1].set_xlabel('Time (s)')
    fig.tight_layout();fig.savefig(OUT/'temporal_errors.png',dpi=160);plt.close(fig)
    frame_ids=[0,48,72,89]
    fig,axes=plt.subplots(len(frame_ids),3,figsize=(15,11))
    for col,(mode,folder) in enumerate([('native',OUT),('dense',OUT/'runs/dense'),('sparse',OUT/'runs/sparse')]):
        pixels=torch.load(folder/'native_pixels.pt',map_location='cpu',weights_only=True)[0]
        for row,idx in enumerate(frame_ids):
            axes[row,col].imshow(pixels[idx].clamp(0,1).numpy());axes[row,col].set_axis_off()
            axes[row,col].set_title(f'{mode}, {idx/24:.2f}s')
        del pixels
    fig.tight_layout();fig.savefig(OUT/'frame_comparison.png',dpi=130);plt.close(fig)
    print('REPORT',OUT/'REPORT.md',flush=True)


if __name__=='__main__':main()
