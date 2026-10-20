"""Validate collected AV artifacts and assemble paired reviews without ratings."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
PRIOR=Path('D:/attention_eval/quality_first_20261014/generation_v4/generation_screen')


def sha(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def link(source,destination):
    destination.parent.mkdir(parents=True,exist_ok=True)
    if destination.exists():
        if sha(source)!=sha(destination):raise RuntimeError('Existing review artifact changed: '+str(destination))
        return
    try:os.link(source,destination)
    except OSError:shutil.copy2(source,destination)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--runs',type=Path,required=True);args=ap.parse_args();runs=args.runs
    state=json.loads((runs/'status.json').read_text());reference=ROOT/'bench_assets/tokyo_neon.jpg'
    link(reference,runs/'reference.jpg')
    previous_metric=json.loads((PRIOR/'native_seed20261013/embedding_metrics.json').read_text())
    (runs/'embedding_revisions.json').write_text(json.dumps(previous_metric['embedding_versions'],indent=2))
    result={'source_status':state['status'],'expected_new_runs':10,'validated_runs':[],
            'scope':'CPU artifact validation and GPU-event analysis of physical-T4 outputs; not SM120 benchmarking',
            'quality_acceptance':'Unrated; proxy diagnostics and complete media do not establish perceptual noninferiority'}
    paired={};timings=[]
    import av
    for record in state.get('runs',[]):
        name=record['name'];directory=runs/name;cfg=json.loads((runs/(name+'.json')).read_text())
        if not (directory/'quality_screen.json').exists():continue
        with av.open(str(directory/'native_video.mp4')) as container:
            stream=container.streams.video[0];fps=float(stream.average_rate)
            decoded=list(container.decode(stream));video={'frames':len(decoded),'fps':fps,
                'width':decoded[0].width,'height':decoded[0].height}
        with av.open(str(directory/'audio.wav')) as container:
            decoded=list(container.decode(audio=0));audio={'samples':sum(f.samples for f in decoded),
                'sample_rate':decoded[0].sample_rate,'channels':len(decoded[0].layout.channels)}
        if video!={'frames':90,'fps':24.,'width':cfg['width'],'height':cfg['height']} or audio!={'samples':120000,'sample_rate':32000,'channels':2}:
            raise RuntimeError('Unexpected video/audio export '+name)
        used=json.loads((directory/'reused_inputs.json').read_text())
        pair=(record['workload'],record['seed'])
        if pair in paired and paired[pair]!=used:raise RuntimeError('Candidates used different conditioning/noise/sigmas')
        paired[pair]=used
        source=(json.loads((runs/'paired_input_provenance.json').read_text())['native_inputs'][str(record['seed'])]['sha256']
                if record['workload']=='fixed' else {f:sha(runs/f'challenge_inputs_seed{record["seed"]}'/f)
                    for f in ['conditioning.pt','initial_noise.pt','sigmas.pt']})
        if source!=used:raise RuntimeError('Paired input provenance mismatch '+name)
        result['validated_runs'].append(record|{'video':video,'audio':audio,
            'video_sha256':sha(directory/'native_video.mp4'),'audio_sha256':sha(directory/'audio.wav'),'paired_input_sha256':used})
        events=json.loads((directory/'attention_profile.json').read_text())
        dit=[e for e in events if e['component']=='dit' and e['layer']>=0]
        excludes={0}
        if record['profiled_step'] is not None:excludes.add(record['profiled_step'])
        if cfg.get('cpu_profile_step') is not None:excludes.add(cfg['cpu_profile_step'])
        usable=[e for e in dit if e['denoise_call'] not in excludes]
        if not all(math.isfinite(e['attention_ms']) and e['attention_ms']>=0 for e in dit):raise RuntimeError('Invalid CUDA event timing')
        calls=json.loads((directory/'denoiser_timings.json').read_text())
        timings.append({'name':name,'scope':'Instrumented within-trajectory calls, not independent steady requests',
            'excluded_calls':sorted(excludes),'DiT_event_count':len(dit),'per_step':[
                {'call':call,'attention_sum_ms':sum(e['attention_ms'] for e in dit if e['denoise_call']==call)} for call in sorted({e['denoise_call'] for e in dit})],
            'per_layer_usable_API_median_ms':{str(layer):statistics.median(e['attention_ms'] for e in usable if e['layer']==layer)
                for layer in sorted({e['layer'] for e in usable})},
            'denoiser_call_wall_seconds':calls,
            'usable_denoiser_wall_median_seconds':statistics.median(c['wall_seconds'] for c in calls if c['call'] not in excludes),
            'actual_QKV_shapes_strides':[dict((k,e[k]) for k in ['device','q_shape','k_shape','q_stride','k_stride','v_stride','dtype'])
                for e in dit if e['denoise_call']==0 and e['layer'] in [0,25]],
            'compilation':json.loads((directory/'compilation.json').read_text())})
    result['complete_new_run_set']=len(result['validated_runs'])==10
    (runs/'media_and_pairing_validation.json').write_text(json.dumps(result,indent=2))
    (runs/'instrumented_timing_summary.json').write_text(json.dumps(timings,indent=2))
    for family in ['fixed','challenge']:
        inputs=runs/'review_inputs'/family;inputs.mkdir(parents=True,exist_ok=True);link(reference,inputs/'reference.jpg')
        for seed in [20261013,20261014]:
            modes=['native','sage32_dit','group32_dit','drop_tau0_dit']
            for mode in modes:
                prior=family=='fixed' and mode!='group32_dit'
                name=f'{mode}_seed{seed}';source=PRIOR/name if prior else runs/f'{family}_{name}'
                config=PRIOR/(name+'.json') if prior else runs/(family+'_'+name+'.json')
                if not config.exists() or not (source/'quality_screen.json').exists():continue
                for filename in ['native_video.mp4','audio.wav','quality_screen.json']:
                    link(source/filename,inputs/name/filename)
                link(config,inputs/(name+'.json'))
        review=runs/'blinded_review'/family
        if not (review/'review.html').exists():
            review.mkdir(parents=True,exist_ok=True)
            subprocess.run([sys.executable,str(ROOT/'benchmarks/quality_first_review.py'),'--runs',str(inputs),'--output',str(review),
                '--candidates','group32_dit' if family=='fixed' else 'sage32_dit,group32_dit,drop_tau0_dit',
                '--key-output',str(runs/(family+'_private_review_key.json'))],check=True,timeout=120)
    print('Validated',len(result['validated_runs']),'new AV runs; paired input identity checked; review ratings remain blank')


if __name__=='__main__':main()
