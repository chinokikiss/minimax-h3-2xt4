"""Offline artifact analysis; no model execution, network or GPU use."""
import argparse
from collections import Counter
import csv
import gzip
import hashlib
import json
from pathlib import Path
import statistics as stats
import subprocess
import wave


def analyze(artifacts, submission, output):
    read=lambda p:json.loads(p.read_text(encoding='utf-8'))
    state=read(submission);batch=read(artifacts/'turbo_085mp_e2e/batch_results.json')
    result=read(artifacts/'turbo_085mp_e2e/generation/result.json')
    setup=read(artifacts/'light_setup/setup_receipt.json')
    overlay=read(artifacts/'turbo_085mp_e2e/source_overlay.json')
    assert batch['status']==result['status']=='complete'
    assert len(result['sampler_steps'])==len(result['denoiser_outputs'])==8
    assert setup['capsule_sha256']==state['source_binding']['publication']['capsule_sha256']
    assert overlay['binding']==state['source_binding']['benchmark_overlay']
    effective=dict(setup['source_hashes']);binding=overlay['binding']
    effective[binding['path']]=binding['sha256']
    for row in binding['additional_files']:effective[row['path']]=row['sha256']
    assert effective==overlay['effective_source_hashes']
    assert (result['config']['width'],result['config']['height'])==(1216,704)
    assert result['decode']==dict(video=dict(strategy='native_temporal_pairs',fallback=None),audio=dict(strategy='native',fallback=None))
    assert all(value['finite'] for value in result['outputs'].values())
    assert len({d['uuid'] for d in batch['hardware']['devices']})==2
    assert all(d['capability']==[7,5] and 'T4' in d['name'] for d in batch['hardware']['devices'])
    output.mkdir(parents=True,exist_ok=True)
    durations={k:(v['end_ns']-v['start_ns'])/1e9 for k,v in result['stages'].items()}
    foreground=result['foreground_seconds']
    calls=[r|dict(seconds=(r['end_ns']-r['start_ns'])/1e9) for r in result['service_calls'] if r['command']=='denoise']
    assert len(calls)==8
    with gzip.open(artifacts/'turbo_085mp_e2e/generation/memory_samples.json.gz','rt') as stream:memory=json.load(stream)
    phases={}
    for name,times in result['stages'].items():
        samples=[m for m in memory if times['start_ns']<=m['time_ns']<=times['end_ns']]
        phases[name]=dict(sample_count=len(samples),devices=[dict(device=i,
            sampled_used_gib_peak=max((m['devices'][i]['used']/2**30 for m in samples),default=None),
            sampled_utilization_percent_mean=stats.mean(m['devices'][i]['utilization'] for m in samples) if samples else None,
            sampled_utilization_percent_median=stats.median(m['devices'][i]['utilization'] for m in samples) if samples else None)
            for i in range(2)])
    ranks=[]
    for rank in result['ranks']:
        b=rank['benchmark'];commands=[r for r in b['commands'] if r['command']=='denoise']
        assert len(commands)==8 and not b['online_lora_attached'] and b['verify_dit_weights'] is False
        assert b['dit_payload_verification'] is False
        attention=Counter(r['nfe'] for r in b['gpu'] if r.get('kind')=='kitchen/int8_attention' and r.get('command')=='denoise')
        assert set(attention.values())=={50} and len(attention)==8
        assert rank['bundle_preparation']['dit']['disk_baked'] and rank['bundle_preparation']['dit']['merged_matrices']==0
        rows=[r for r in b['gpu'] if r.get('kind')=='command_gpu_span' and r.get('command')=='denoise']
        ranks.append(dict(rank=rank['rank'],denoise_allocated_gib=[r['peak_allocated']/2**30 for r in commands],
            denoise_reserved_gib=[r['peak_reserved']/2**30 for r in commands],
            denoise_cuda_interval_seconds=[r['cuda_ms']/1000 for r in rows],
            te_allocated_gib=next(r['peak_allocated']/2**30 for r in b['commands'] if r['command']=='te'),
            temporal_allocated_gib=max(r['peak_allocated']/2**30 for r in b['commands'] if r['command']=='video_temporal_pair'),
            temporal_pair_calls=sum(r['command']=='video_temporal_pair' for r in b['commands']),
            denoise_attention_calls=sum(attention.values()),executed_backends=b['executed_backends'],
            host_cache={k:rank['host_cache'][k] for k in ('preparations','hits','misses','evictions','prepare_ms','wait_ms','peak_cache_bytes')},
            layouts=b['layouts']))
    sampler=result['stages']['sampler']
    preparation=[dict(command=r['command'],seconds=(r['end_ns']-r['start_ns'])/1e9)
        for r in result['service_calls'] if sampler['start_ns']<=r['start_ns'] and r['end_ns']<=sampler['end_ns'] and r['command']!='denoise']
    media_root=artifacts/'turbo_085mp_e2e/generation'
    probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_entries',
        'format=duration,size:stream=codec_type,codec_name,width,height,avg_frame_rate,nb_frames,sample_rate,channels,duration',
        '-of','json',str(media_root/'video.mp4')],text=True))
    video=next(s for s in probe['streams'] if s['codec_type']=='video')
    assert (video['width'],video['height'],video['nb_frames'],video['avg_frame_rate'])==(1216,704,'240','24/1')
    assert float(probe['format']['duration'])==10.
    with wave.open(str(media_root/'audio.wav')) as stream:
        wav=dict(channels=stream.getnchannels(),sample_rate=stream.getframerate(),samples=stream.getnframes(),sample_bytes=stream.getsampwidth())
    assert wav==dict(channels=2,sample_rate=32000,samples=320000,sample_bytes=2)
    files={name:dict(bytes=(media_root/name).stat().st_size,sha256=hashlib.sha256((media_root/name).read_bytes()).hexdigest())
        for name in ('video.mp4','audio.wav','decode_inputs.pt')}
    summary=dict(status='complete',physical_hardware=batch['hardware'],completed_steps=8,
        input_config=result['config'],runtime=result['runtime'],layout=ranks[0]['layouts'][0],
        setup_seconds=result['setup_seconds'],foreground_seconds=foreground,
        stage_seconds=durations,stage_percent_foreground={k:100*v/foreground for k,v in durations.items()},
        batch_command_seconds={r['stage']:r['seconds'] for r in batch['commands']},
        execution_worker_seconds=read(artifacts/'turbo_085mp_supervision/supervision.json')['seconds'],
        environment_setup_seconds=setup['seconds'],persistent_bake=batch['persistent_bake'],
        denoise_service_seconds=[r['seconds'] for r in calls],denoise_service_sum_seconds=sum(r['seconds'] for r in calls),
        later_denoise_service_median_seconds=stats.median(r['seconds'] for r in calls[1:]),
        sampler_non_denoise_calls=preparation,ranks=ranks,sampled_phases=phases,memory_samples=len(memory),
        cgroup_sampled_peak_gib=max(r['cgroup']['current']/2**30 for r in memory if r['cgroup']['available']),
        cgroup_limit_gib=next(r['cgroup']['limit']/2**30 for r in memory if r['cgroup']['available']),
        all_final_outputs_finite=True,output_tensor_digests=result['outputs'],decoded_shapes=result['decoded_shapes'],
        export_shapes=result['export_shapes'],decode=result['decode'],media=files,media_probe=probe,wav_probe=wav,
        download_retries=batch.get('download_retries',[]),source_binding_verified=True,
        quality_acceptance=result['quality_acceptance'],
        timing_scope='One cold E2E request; CUDA intervals include waits; NVML values are sampled. No exclusive kernel profiling or steady-state speedup claim.')
    (output/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    with (output/'denoiser_calls.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.writer(stream);writer.writerow(['call','service_seconds','rank0_cuda_interval_seconds','rank1_cuda_interval_seconds','rank0_allocated_gib','rank1_allocated_gib','rank0_reserved_gib','rank1_reserved_gib'])
        for i,r in enumerate(calls):writer.writerow([i+1,r['seconds'],*[rank['denoise_cuda_interval_seconds'][i] for rank in ranks],*[rank['denoise_allocated_gib'][i] for rank in ranks],*[rank['denoise_reserved_gib'][i] for rank in ranks]])
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'svg.fonttype':'none'})
    fig,(left,right)=plt.subplots(1,2,figsize=(12,4.3),layout='constrained')
    labels=['Reference + TE','Sampler','Video decode','Audio decode','Export']
    values=[durations[k] for k in ('reference_and_TE','sampler','video_decode','audio_decode','file_export')]
    left.barh(labels[::-1],values[::-1],color=['#7c9ab8','#5c9f9a','#6389ba','#b89c57','#7b6ba6'][::-1])
    left.set_xlabel('Wall time (seconds)');left.set_title('One cold request: foreground stages')
    for i,value in enumerate(values[::-1]):left.text(value+10,i,f'{value:.2f}',va='center')
    left.set_xlim(0,max(values)*1.2)
    right.plot(range(1,9),[r['seconds'] for r in calls],marker='o',color='#326fa3')
    right.axhline(summary['later_denoise_service_median_seconds'],linestyle='--',color='#b17724',label='Calls 2-8 median')
    right.set_xticks(range(1,9));right.set_xlabel('Denoiser call');right.set_ylabel('Service wall time (seconds)')
    right.set_title('Same trajectory, different timestep inputs');right.legend()
    svg=output/'timings.svg';fig.savefig(svg)
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text(encoding='utf-8').splitlines())+'\n',encoding='utf-8')
    fig.savefig(output/'timings.png',dpi=150);plt.close(fig)
    print(json.dumps({k:summary[k] for k in ('foreground_seconds','stage_seconds','denoise_service_seconds','cgroup_sampled_peak_gib','source_binding_verified')},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--artifacts',type=Path,required=True)
    p.add_argument('--submission',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();analyze(a.artifacts,a.submission,a.output)
