"""Locate full-request device peaks using existing, clock-aligned telemetry."""
import argparse
from collections import defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import statistics


def analyze(path):
    data = json.loads(path.read_text(encoding='utf8'))
    if not data.get('physical') or not data.get('passed'):
        raise ValueError('Only completed physical full-request captures are accepted: ' + str(path))
    samples = data['host_monitor']['samples']
    requests = [('measured', r) for r in data['requests']]
    requests += [('warmup', r) for r in data.get('warmup', [])]
    stages = {}
    global_peaks = {}
    times = [s['monotonic_ns'] for s in samples]
    for sample in samples:
        t = sample['monotonic_ns']
        request = next(((kind, r) for kind, r in requests if r['start_ns'] <= t <= r['end_ns']), None)
        stage = None
        if request:
            stage = next((name for name, span in request[1]['stages'].items()
                          if span['start_ns'] <= t <= span['end_ns']), 'request_handoff')
        for device in sample['devices']:
            rank, used = device['device'], device['used']
            point = dict(bytes=used, gib=used / 2**30, monotonic_ns=t,
                         request=request[1]['request'] if request else None,
                         request_kind=request[0] if request else 'outside_request', stage=stage,
                         request_relative_s=(t - request[1]['start_ns']) / 1e9 if request else None)
            if used > global_peaks.get(rank, {}).get('bytes', -1):
                global_peaks[rank] = point
            if request:
                key = (request[0], stage, rank)
                if used > stages.get(key, {}).get('bytes', -1):
                    stages[key] = point
    correlations = []
    for rank, point in global_peaks.items():
        trace = path.parent / ('batch_r' + str(rank) + '_causal.json.gz')
        if not trace.is_file():
            continue
        with gzip.open(trace, 'rt', encoding='utf8') as stream:
            causal = json.load(stream)
        t = point['monotonic_ns']
        active = [e for e in causal['gpu_intervals'] if e['name'] == 'block_compute_stream'
                  and e['start_ns'] <= t <= e['end_ns']]
        ready = [e for e in causal['events'] if e['name'] == 'weight_compute_ready' and e['start_ns'] <= t]
        nearest = max(ready, key=lambda e: e['start_ns']) if ready else None
        correlations.append(dict(rank=rank, trace=str(trace), peak_sample=point,
                                 active_block_stream_intervals=active, preceding_ready_observation=nearest,
                                 caution='CPU enqueue can run ahead of GPU; sampled NVML plateau does not identify the allocating operator'))
    return dict(source=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                case=data['case'], capture=data['capture'], workload=data['workload'],
                measured_requests=len(data['requests']), warmup_requests=len(data.get('warmup', [])),
                observed_sample_interval_ms=statistics.median(b - a for a, b in zip(times, times[1:])) / 1e6,
                global_sampled_driver_peaks={str(k): v for k, v in global_peaks.items()},
                stage_sampled_driver_peaks=[dict(kind=kind, stage=stage, rank=rank, peak=point)
                                            for (kind, stage, rank), point in sorted(stages.items())],
                worker_batch_allocator_peaks=[dict(rank=r['rank'], **r['gpu']) for r in data['runtime']],
                parent_batch_allocator_peaks=data.get('parent_gpu'), correlations=correlations)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('inputs', nargs='+', type=Path, help='Physical result.json files or directories containing them')
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    paths = sorted({p for source in args.inputs for p in
                    (source.rglob('result.json') if source.is_dir() else [source])})
    records = [analyze(path) for path in paths]
    summary = dict(scope='Reanalysis only; no inference changes or new GPU performance claims',
                   units='GiB = 2**30 bytes; NVML board use and process-local allocator peaks are distinct',
                   limitations=['Sampling can miss sub-sample transients',
                                'NVML includes inactive allocator cache, all processes and non-PyTorch allocations',
                                'Do not sum independent stage/process/rank maxima',
                                'Batch allocator counters do not identify the allocating operator',
                                'Historical full-request source predates current Online LoRA/operator experiments'],
                   captures=records)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2), encoding='utf8')
    maxima = defaultdict(float)
    for record in records:
        for stage in record['stage_sampled_driver_peaks']:
            if record['capture'] == 'timing' and stage['kind'] == 'measured':
                key = (stage['stage'], stage['rank'])
                maxima[key] = max(maxima[key], stage['peak']['gib'])
    print('CAPTURES', len(records), 'MEASURED_REQUESTS', sum(r['measured_requests'] for r in records))
    for key, value in sorted(maxima.items()):
        print(key, round(value, 6), 'GiB')


if __name__ == '__main__':
    main()
