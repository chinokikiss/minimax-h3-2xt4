"""One-shot collection of one submitted Kaggle version; never launches GPU jobs."""
import argparse
from collections import defaultdict
import gzip
import json
from pathlib import Path
import subprocess
import sys
import time


def summarize(output):
    base=output/'ref2va_e2e_results'
    rows=[]
    for name in ('S1','S2','L1'):
        path=base/name/'result.json'
        if not path.exists():continue
        result=json.loads(path.read_text())
        finite=(all(item.get('finite') is True for item in result['outputs'].values())
                if result.get('outputs') else None)
        ranks=result.get('ranks',[])
        stages=result.get('stages',{})
        preload=result.get('preload',{})
        preload_seconds=((preload['end_ns']-preload['start_ns'])/1e9
            if 'end_ns' in preload else preload.get('seconds',0))
        rank_stats=[]
        for rank in ranks:
            bench=rank['benchmark']
            denoise=[c for c in bench['commands'] if c['command']=='denoise']
            gpu=defaultdict(float)
            for span in bench['gpu']:gpu[span['kind']]+=span['cuda_ms']
            windows=[e for e in rank.get('events',[]) if 'window' in e]
            rank_stats.append(dict(rank=rank['rank'],**{k:bench[k] for k in (
                'peak_allocated','peak_reserved','layouts','executed_backends')},
                nfe=len(denoise),denoise=denoise,
                denoise_peak_allocated=max((c['peak_allocated'] for c in denoise),default=0),
                denoise_peak_reserved=max((c['peak_reserved'] for c in denoise),default=0),
                gpu_span_ms=dict(gpu),host_cache=rank.get('host_cache'),
                windows=[dict(phase=e['phase'],metrics={k:v for k,v in e['window'].items()
                    if k in ('hits','h2d_bytes','evictions','host_wait_ms','idle_gpu_hits',
                             'idle_avoided_h2d_bytes','peak_live_weight_bytes','peak_pinned_admitted_bytes')},
                    h2d_cuda_ms=sum(c['cuda_ms'] for c in e['window'].get('copies',[]))) for e in windows]))
        memory=base/name/'memory_samples.json.gz'
        board_peaks={}
        if memory.exists():
            with gzip.open(memory,'rt') as stream:
                for sample in json.load(stream):
                    for device in sample['devices']:
                        key=str(device['device']);board_peaks[key]=max(board_peaks.get(key,0),device['used'])
        rows.append(dict(run=name,status=result['status'],outputs_finite=finite,
            foreground_seconds=result.get('foreground_seconds'),
            preload_seconds=preload_seconds,
            preload_plus_foreground_seconds=(result['foreground_seconds']+preload_seconds)
                if 'foreground_seconds' in result else None,
            stages_seconds={k:(v['end_ns']-v['start_ns'])/1e9 for k,v in stages.items()},
            service_wall_ms=[dict(command=c['command'],wall_ms=(c['end_ns']-c['start_ns'])/1e6)
                for c in result.get('service_calls',[]) if c['command'] in ('te','refine','denoise')],
            nvml_sampled_peak_used_bytes=board_peaks,ranks=rank_stats))
    (output/'compact_summary.json').write_text(json.dumps(rows,indent=2),encoding='utf8')
    lines=['# REF2VA generation results','',
        'Physical results only where a run completed. Download/service setup and idle preload are separate from foreground latency.',
        '', '| Run | Status | Foreground (s) | Preload (s) | Preload + foreground (s) |',
        '|---|---|---:|---:|---:|']
    def number(value):return '-' if value is None else f'{value:.3f}'
    for row in rows:
        lines.append('| '+ ' | '.join([row['run'],row['status'],number(row['foreground_seconds']),
            number(row['preload_seconds']),number(row['preload_plus_foreground_seconds'])])+' |')
    for row in rows:
        directory=base/row['run']
        finite_text='unavailable (no output tensors produced)' if row['outputs_finite'] is None else str(row['outputs_finite'])
        lines+=['',f"## {row['run']}",'',f"All saved output tensors finite: {finite_text}."]
        for filename in ('video.mp4','audio.wav'):
            media=directory/filename
            if media.exists():lines.append(f'- [{filename}](<{media.resolve().as_posix()}>)')
        if row['stages_seconds']:
            lines+=['', '| Stage | Wall time (s) |','|---|---:|']
            lines += [f'| {stage} | {seconds:.3f} |' for stage,seconds in row['stages_seconds'].items()]
        for rank in row['ranks']:
            lines+=['', f"Rank {rank['rank']}: allocated peak {rank['peak_allocated']/2**30:.3f} GiB; reserved peak {rank['peak_reserved']/2**30:.3f} GiB.",
                f"Denoising ({rank['nfe']} actual evaluations): allocated peak {rank['denoise_peak_allocated']/2**30:.3f} GiB; reserved peak {rank['denoise_peak_reserved']/2**30:.3f} GiB.",
                'Actual packed layouts: `'+json.dumps(rank['layouts'])+'`.',
                'Executed entrypoints: `'+json.dumps(rank['executed_backends'])+'`.']
    checks=base/'s1_s2_correctness.json'
    if checks.exists():lines+=['','S1/S2 tensor checks: `'+checks.read_text().strip()+'`.']
    lines+=['','Interpretation limits: one cold request per configuration; first-use compilation may be included. '
        'CUDA event spans can contain idle/dependency time. LoRA bypass spans measure attributed work, not a causal no-LoRA ablation. '
        'S1/S2 equality is an idle-policy comparison, not a new dense Native accuracy reference. '
        'L1 preserves 864x480/362 frames as explicitly confirmed by the user.']
    (output/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf8')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--kernel',default='ocnsai/h3-ref2va-end-to-end')
    p.add_argument('--version',required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--wait',action='store_true')
    p.add_argument('--tls12',action='store_true')
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    from kaggle.api.kaggle_api_extended import KaggleApi
    api=KaggleApi();api.authenticate()
    deadline=time.monotonic()+9*3600
    previous=None
    while True:
        try:
            status=str(api.kernels_status(args.kernel).status)
            state=dict(kernel=args.kernel,version=args.version,status=status,checked_ns=time.time_ns(),
                collection='waiting_for_terminal_status',launches_gpu_jobs=False)
            (args.output/'collector_status.json').write_text(json.dumps(state,indent=2),encoding='utf8')
            if status!=previous:print('COLLECTOR_STATUS',status,flush=True);previous=status
            terminal=any(word in status.upper() for word in ('COMPLETE','ERROR','CANCEL','FAIL'))
            if terminal or not args.wait:break
        except Exception as error:
            (args.output/'collector_status.json').write_text(json.dumps(dict(
                status='status_lookup_failed',error_type=type(error).__name__,checked_ns=time.time_ns())),encoding='utf8')
            if not args.wait:raise
        if time.monotonic()>deadline:raise TimeoutError('One-shot collection deadline exhausted')
        time.sleep(60)
    command=[sys.executable,str(Path(__file__).with_name('download_kaggle_evidence.py')),args.kernel,
        '--version',args.version,'--output',str(args.output),
        '--pattern',r'^ref2va_e2e_results/.*\.(json|gz|txt|log|mp4|wav|pt|md|svg|html|csv)$']
    if args.tls12:command.insert(1,str(Path(__file__).with_name('kaggle_tls12_runner.py')))
    # Completed sessions can take a little time to expose their output listing.
    for attempt in range(6):
        subprocess.run(command,check=True,timeout=900)
        manifest=json.loads((args.output/'download_manifest.json').read_text())
        if manifest['files']:break
        if not args.wait:break
        time.sleep(20)
    summarize(args.output)
    memory=args.output/'ref2va_e2e_results/L1/gpu_memory'
    if memory.exists():
        subprocess.run([sys.executable,str(Path(__file__).with_name('analyze_gpu_memory_overhead.py')),
            '--directory',str(memory)],check=True,timeout=300)
    state.update(collection='downloaded',files=len(manifest['files']),completed_ns=time.time_ns())
    (args.output/'collector_status.json').write_text(json.dumps(state,indent=2),encoding='utf8')
    print('COLLECTOR_COMPLETE',len(manifest['files']),str(args.output.resolve()),flush=True)


if __name__=='__main__':main()
