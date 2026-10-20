"""Accept physical P0/P1/P2 ablations and keep evidence limitations explicit."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

from analyze_pipeline_audit import overlap


CASES = {
    'direct_sync': ('sync', 'direct', False),
    'direct_p1': ('adaptive', 'direct', False),
    'direct_p1_p2': ('adaptive', 'direct', True),
    'offline_p1_p2': ('adaptive', 'offline', True),
}


def startup_during_decode(rank, previous):
    events = rank['events'][len(previous.get('events', [])):]
    begin = [e['monotonic'] for e in events if e['phase'] == 'decode_start']
    end = [e['monotonic'] for e in events if e['phase'] == 'decode_complete']
    intervals = list(zip(begin, end))
    cache = rank['startup_cache']
    old_cache = previous.get('startup_cache', {})
    pending = {}
    preparations = []
    old_count = len(old_cache.get('events', []))
    for index, e in enumerate(cache['events']):
        if e['event'] == 'prepare_started':
            pending[e['key']] = e
        elif e['event'] == 'prepare_finished' and e['key'] in pending:
            started = pending.pop(e['key'])
            if index >= old_count:
                preparations.append((started, e))
    return dict(rank=rank['rank'], decode_intervals_monotonic=intervals,
        background_cpu_prepare_overlap_seconds=sum(
            overlap([(a['monotonic'], b['monotonic'])], intervals)
            for a, b in preparations if not a['foreground']),
        background_items_completed_inside_decode=sum(
            not a['foreground'] and any(lo <= b['monotonic'] <= hi for lo, hi in intervals)
            for a, b in preparations),
        pending_at_snapshot=len(pending),
        startup_cache_delta={k:cache[k]-old_cache.get(k, 0)
                            for k in ('hits', 'misses', 'preparations', 'prepare_ms', 'wait_ms', 'evictions')})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    root = args.directory.resolve()
    hardware = json.loads((root/'hardware.json').read_text())
    if len(hardware['devices']) != 2 or not all('T4' in d['name'] for d in hardware['devices']):
        raise ValueError('Two physical T4 devices required')
    native = json.loads((root/'native_result.json').read_text())
    if not (native['passed'] and native['physical'] and native['layers'] == 50):
        raise ValueError('Fresh completed full-depth Native reference required')
    accepted = {}
    pending = []
    for label, (strategy, loader, p2) in CASES.items():
        folder = root/label
        path = folder/f'{strategy}_result.json'
        if not path.exists():
            pending.append(label)
            continue
        result = json.loads(path.read_text())
        if not result.get('passed'):
            pending.append(label)
            continue
        sources_match=result['source_sha256']==native['source_sha256']
        if not sources_match and (root/'loading_launcher_repair.json').exists():
            repair=json.loads((root/'loading_launcher_repair.json').read_text())
            old,new=native['source_sha256'],result['source_sha256']
            changed={k for k in set(old)|set(new) if old.get(k)!=new.get(k)}
            name='benchmarks/run_loading_audit.py'
            sources_match=(changed=={name} and repair['passed'] and
                           repair['before']==old.get(name) and repair['after']==new.get(name))
        if not (result['physical'] and sources_match and all(result[k] == native[k]
                for k in ('layers', 'steps', 'width', 'height'))):
            raise ValueError('Unmatched physical workload/source: '+label)
        policy = result['loading_policy']
        if policy['loader'] != loader or policy['cross_request_te'] != p2:
            raise ValueError('Unexpected loading policy: '+label)
        if len(result['requests']) != 2:
            raise ValueError('Cold and warm requests required: '+label)
        for request, temperature in zip(result['requests'], ('cold', 'warm')):
            if request['temperature'] != temperature or set(request['correctness']) != {
                    'conditioning', 'video', 'audio', 'pixels', 'waveform'}:
                raise ValueError('Incomplete output validation: '+label)
            if not all(c['passed'] and c['is_bit_exact'] for c in request['correctness'].values()):
                raise ValueError('Incorrect performance result rejected: '+label)
            if any(request[k] != native['requests'][0][k] for k in (
                    'video_shape', 'audio_shape', 'video_latent_shape',
                    'audio_latent_shape', 'conditioning_shape')):
                raise ValueError('Unmatched output shapes: '+label)
        subprocess.run([sys.executable, str(Path(__file__).with_name('analyze_pipeline_audit.py')),
                        str(folder), '--strategies', strategy], check=True, stdout=subprocess.DEVNULL)
        analysis = json.loads((folder/'analysis.json').read_text())
        previous = {}
        p2_details = []
        for request in result['requests']:
            for rank in request['runtime']:
                p2_details.append(dict(temperature=request['temperature'],
                    **startup_during_decode(rank, previous.get(rank['rank'], {}))))
                previous[rank['rank']] = rank
        accepted[label] = dict(analysis=analysis, p2_cpu_evidence=p2_details,
            runtime_correctness='all five outputs Native bit-exact, cold and warm',
            memory_scope='whole case including construction, validation and profiling; rank peaks cumulative')
    comparisons = []
    for name, before, after in (
        ('P1 independent CPU/GPU scheduling', 'direct_sync', 'direct_p1'),
        ('P2 next-request TE CPU preparation', 'direct_p1', 'direct_p1_p2'),
        ('P0 offline versus direct rank-local reads', 'direct_p1_p2', 'offline_p1_p2')):
        if before not in accepted or after not in accepted:
            continue
        for temperature in ('cold', 'warm'):
            a = next(t for t in accepted[before]['analysis']['timings'] if t['temperature'] == temperature)
            b = next(t for t in accepted[after]['analysis']['timings'] if t['temperature'] == temperature)
            comparisons.append(dict(ablation=name, temperature=temperature, before=before, after=after,
                before_ms=a['end_to_end_ms'], after_ms=b['end_to_end_ms'],
                latency_reduction_ms=a['end_to_end_ms']-b['end_to_end_ms'],
                latency_reduction_percent=100*(1-b['end_to_end_ms']/a['end_to_end_ms']),
                stage_reduction_ms={k:a['stage_ms'][k]-b['stage_ms'][k]
                                    for k in a['stage_ms'] if k in b['stage_ms']}))
    evidence = dict(status='complete' if not pending else 'partial', hardware=hardware,
        accepted=accepted, pending=pending, comparisons=comparisons, limitations=[
            'One cold/warm pair per configuration; differences are descriptive, not statistically established.',
            'Native end-to-end includes reference artifact IO and uses different startup boundaries; only TP ablations are compared.',
            'Profiler covers one extra full-depth DiT evaluation, not TE or VAE; it is outside request timing.',
            'Runtime counter snapshots include output validation and background preparation; they are not exact timed-request counters.',
            'NVML PCIe values are sampled estimates of all traffic, not exact weight or NCCL wire bytes.',
            'P0 comparison isolates offline versus direct; historical full-Native-block loading is not rerun.',
            'P2 reduces next-request readiness stalls; decode timing differences can indicate contention, not decode acceleration.',
            'Cold runtime cache does not prove physically cold filesystem cache.'])
    manifest_path=root/'manifest.json'
    if manifest_path.exists():
        provenance=json.loads(manifest_path.read_text()).get('reference_provenance')
        if provenance:
            evidence['reference_provenance']=provenance
            evidence['limitations'].append('Native numerical tensors are reused from the stated prior physical session after operator-source verification; Native timing is not a paired same-session comparison.')
    (root/'p012_analysis.json').write_text(json.dumps(evidence, indent=2))
    lines = ['# Physical P0/P1/P2 audit', '', 'Status: '+evidence['status']+'. Native correctness gates apply to every accepted row.', '',
        '| Ablation | Request | Before (s) | After (s) | Reduction (%) |',
        '|---|---|---:|---:|---:|']
    for c in comparisons:
        lines.append(f"| {c['ablation']} | {c['temperature']} | {c['before_ms']/1000:.3f} | {c['after_ms']/1000:.3f} | {c['latency_reduction_percent']:.2f} |")
    lines += ['', *['- '+s for s in evidence['limitations']]]
    if pending:
        lines += ['', 'Pending/failed configurations: '+', '.join(pending)+'.']
    (root/'P012_RESULTS.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(status=evidence['status'], accepted=list(accepted), pending=pending, comparisons=comparisons), indent=2))


if __name__ == '__main__':
    main()
