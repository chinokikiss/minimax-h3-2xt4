"""Offline analysis of retained Query/Temporal worker traces; no Kaggle calls."""
import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import statistics


def analyze(comparison):
    result_path = comparison/'result.json'
    if not result_path.exists():
        result_path = comparison/'gpu_result.json'
    result = json.loads(result_path.read_text())
    memory_path = comparison/'memory_samples.json.gz'
    if not memory_path.exists():
        memory_path = comparison/'gpu_memory_samples.json.gz'
    memory = json.loads(gzip.decompress(memory_path.read_bytes()))
    assert result['status'] == 'complete' and len(result['cases']) == 6
    devices = result['hardware']['devices']
    assert len({d['uuid'] for d in devices}) == 2
    assert all(d['capability'] == [7, 5] and 'T4' in d['name'] for d in devices)
    cases = []
    for case in result['cases']:
        low, high = case['start_ns'], case['end_ns']
        samples = [s for s in memory if low <= s['time_ns'] <= high]
        assert samples and case['fallback'] is None and case['check']['finite']
        workers = []
        for worker in result['worker_stats']:
            commands = [c for c in worker['commands']
                        if low <= c['start_ns'] and c['end_ns'] <= high]
            collectives = defaultdict(lambda: dict(calls=0, operand_bytes=0))
            for command in commands:
                for name, value in command['collectives'].items():
                    for key in ('calls', 'operand_bytes'):
                        collectives[name][key] += value[key]
            workers.append(dict(rank=worker['rank'], commands=dict(Counter(c['command'] for c in commands)),
                cuda_event_span_seconds=sum((c['gpu_span_ms'] or 0)/1000 for c in commands),
                thread_cpu_seconds=sum(c['thread_cpu_ns']/1e9 for c in commands),
                command_wall_seconds=sum((c['end_ns']-c['start_ns'])/1e9 for c in commands),
                cpu_reply_bytes=sum(c['reply_bytes'] for c in commands), collectives=dict(collectives),
                peak_allocated_bytes=max(c['peak_allocated'] for c in commands),
                peak_reserved_bytes=max(c['peak_reserved'] for c in commands)))
        cases.append(dict(mode=case['mode'], repeat=case['repeat'], seconds=case['seconds'],
            workers=workers, nvml_samples=len(samples),
            nvml_peak_device_bytes=[max(s['devices'][i]['used'] for s in samples) for i in (0, 1)],
            nvml_mean_utilization=[statistics.mean(s['devices'][i]['utilization'] for s in samples) for i in (0, 1)]))
    summary = {}
    for mode in ('query', 'temporal'):
        rows = [c for c in cases if c['mode'] == mode]
        checks = [c['check'] for c in result['cases'] if c['mode'] == mode]
        assert len(rows) == 3
        summary[mode] = dict(samples_seconds=[c['seconds'] for c in rows],
            median_seconds=statistics.median(c['seconds'] for c in rows),
            repeatable_output_sha256=len({c['sha256'] for c in checks}) == 1,
            peak_device_bytes=[max(c['nvml_peak_device_bytes'][i] for c in rows) for i in (0, 1)],
            median_worker_cuda_span_seconds=[statistics.median(c['workers'][i]['cuda_event_span_seconds'] for c in rows) for i in (0, 1)],
            mean_sampled_utilization=[statistics.mean(c['nvml_mean_utilization'][i] for c in rows) for i in (0, 1)])
        assert summary[mode]['median_seconds'] == result['summary'][mode]['median_seconds']
    per_frame = next(c['check']['per_frame_max_abs'] for c in result['cases'] if c['mode'] == 'temporal')
    speedup = summary['query']['median_seconds']/summary['temporal']['median_seconds']
    return dict(hardware=result['hardware'], summary=summary, cases=cases,
        speedup=speedup, latency_reduction_percent=100*(1-1/speedup),
        accuracy=dict(reference='First full Query output; Native control is cropped only',
            controls=result['controls'], all_finite=True,
            full_cross_method_max_abs=max(per_frame), nonzero_frames=sum(v > 0 for v in per_frame),
            frames=len(per_frame), first_48_frames_max_abs=max(per_frame[:48]),
            remaining_frames_max_abs=max(per_frame[48:])),
        evidence_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (result_path, memory_path)},
        limits=['CUDA event spans include host gaps and are not exclusive kernel times.',
                'Collective operand bytes are not physical PCIe traffic.',
                'Allocator peaks persist from replica preparation until the next preparation.',
                'NVML reports sampled whole-device usage, including parent context/allocations.',
                'No full Native output, media review or generation-quality experiment.'])


def plot(result, comparison, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.6), layout='constrained')
    colors = ['#3379ae', '#dd7830']
    for i, mode in enumerate(('query', 'temporal')):
        row = result['summary'][mode]
        axes[0].bar(i, row['median_seconds'], color=colors[i], alpha=.8)
        axes[0].scatter([i]*3, row['samples_seconds'], c='black', s=18, zorder=3)
        axes[0].text(i, row['median_seconds']+6, f"{row['median_seconds']:.2f}s", ha='center')
        bars = axes[1].bar([i-.17, i+.17], [x/2**30 for x in row['peak_device_bytes']],
                           width=.3, color=colors[i])
        for bar, alpha in zip(bars, (.9, .5)):
            bar.set_alpha(alpha)
    for axis in axes[:2]:
        axis.set_xticks([0, 1], ['Query', 'Temporal'])
        axis.spines[['top', 'right']].set_visible(False)
    axes[0].set(title='Warm decode: median and all 3 runs', ylabel='Seconds', ylim=(0, 285))
    axes[1].set(title='Sampled device peak: GPU 0 / GPU 1', ylabel='GiB', ylim=(0, 6.3))
    path = comparison/'result.json'
    if not path.exists():
        path = comparison/'gpu_result.json'
    raw = json.loads(path.read_text())
    frames = next(c['check']['per_frame_max_abs'] for c in raw['cases'] if c['mode'] == 'temporal')
    axes[2].plot([i/24 for i in range(len(frames))], frames, linewidth=1, color=colors[1])
    axes[2].axvline(2, linestyle='--', color='gray', linewidth=1)
    axes[2].set(title='Temporal vs Query: per-frame maximum', xlabel='Video time (seconds)', ylabel='Maximum absolute pixel difference')
    axes[2].ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    axes[2].spines[['top', 'right']].set_visible(False)
    fig.suptitle('Two physical Tesla T4s | saved H3 latent [1,24,107,34,60] | 362 frames at 960 x 544')
    fig.savefig(output, dpi=150)
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--comparison', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--plot', type=Path)
    args = parser.parse_args()
    result = analyze(args.comparison)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    if args.plot:
        plot(result, args.comparison, args.plot)
    print(json.dumps(dict(speedup=result['speedup'], latency_reduction_percent=result['latency_reduction_percent'])))
