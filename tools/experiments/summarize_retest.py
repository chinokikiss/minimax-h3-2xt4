"""Summarize measured reruns without assigning speedups to failed outputs."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics

def rows_for(suite, data):
    if suite in ('te','dit'):
        native = data.get('native', data.get('baseline', {}))
        for name, value in native.items():
            yield name,'native',value,None
        for mode, workloads in data.get('proposals',{}).items():
            if 'status' in workloads:
                yield '*',mode,workloads,None
            else:
                for name,value in workloads.items():
                    yield name,mode,value,native.get(name)
    elif suite == 'video':
        native = data.get('strategies',{}).get('native',{})
        for mode,workloads in data.get('strategies',{}).items():
            if 'status' in workloads:
                yield '*',mode,workloads,None
            else:
                for name,value in workloads.items():
                    yield name,mode,value,native.get(name)
    else:
        for name,workload in data.get('workloads',{}).items():
            if suite.endswith('_stack'):
                yield name,'PP=2 50 blocks',workload,None
            else:
                native = workload.get('strategies',{}).get('native')
                for mode,value in workload.get('strategies',{}).items():
                    value = {**value, 'input_sha256':workload.get('input_sha256')}
                    if 'tiled_2gpu' in mode:
                        other = workload['strategies'].get(mode.replace('2gpu','1gpu'),{})
                        value = {**value,'matching_tiles_reference_median_ms':other.get('median_ms')}
                    yield name,mode,value,native

def summarize(root, expected_rounds=3):
    grouped = defaultdict(list)
    selection_path = root/'selection.json'
    selection = json.loads(selection_path.read_text()) if selection_path.exists() else None
    if selection and selection.get('expected_rounds', expected_rounds) != expected_rounds:
        raise ValueError('Requested round count differs from the recorded selection')
    if selection:
        paths = [root/allocation['folder']/'retest_results'/f'round_{round_index}'/suite/'results.json'
                 for allocation in selection['allocations'] for round_index in range(expected_rounds)
                 for suite in allocation['suites'] if suite != 'video_profile']
    else:
        paths = sorted(root.glob('**/results.json'))
    for path in paths:
        if not path.exists():
            continue
        suite = path.parent.name
        for name,mode,value,native in rows_for(suite,json.loads(path.read_text())):
            grouped[suite,name,mode].append((str(path.relative_to(root) if path.is_relative_to(root) else path),value,native))
    lines = ['# Measured benchmark rerun results','',
             f'Times below are medians of per-process medians. Each successful round has 3 warmups and 10 synchronized measurements; {expected_rounds} fresh-process rounds are expected per suite. Each suite uses one Kaggle allocation, recorded in selection.json when results come from multiple allocations. Speedups are accepted only when all expected rounds pass the declared numerical gate. Tiling remains an approximation to full decoding.','',
             '| Suite | Workload | Strategy | Rounds | Median ms | Round range ms | Max rel L2 | Min cosine | Assessment |',
             '|---|---|---|---:|---:|---:|---:|---:|---|']
    detailed = []
    for (suite,name,mode),entries in sorted(grouped.items()):
        good = [(r,v,b) for r,v,b in entries if 'median_ms' in v]
        times = [v['median_ms'] for _,v,_ in good]
        accuracies = [v.get('accuracy',v.get('accuracy_vs_normal',{})) for _,v,_ in good]
        valid = [v for v in accuracies if v.get('valid')]
        l2 = max((a['rel_l2'] for a in valid),default=None)
        cosine = min((a['cosine'] for a in valid),default=None)
        max_abs = max((a['max_abs'] for a in valid),default=None)
        gated = [v['correctness_passed'] for _,v,_ in good if 'correctness_passed' in v]
        if not times:
            status = 'EXECUTION FAILED'
        elif len(good) < expected_rounds or len(good) < len(entries):
            status = 'PARTIAL / EXECUTION FAILURE'
        elif gated:
            status = 'PASS' if all(gated) else 'CORRECTNESS FAILED'
        elif mode == 'native':
            status = 'Reference'
        else:
            agreement = [v.get('matching_tiles_passed') for _,v,_ in good if 'matching_tiles_passed' in v]
            status = 'Approximate tiles' + ('; matching paths PASS' if agreement and all(agreement) else '; matching paths FAIL' if agreement else '')
        speedups = [b['median_ms']/v['median_ms'] for _,v,b in good if b and 'median_ms' in b]
        accepted = statistics.median(speedups) if len(speedups)==expected_rounds and len(gated)==expected_rounds and all(gated) and len(good)==len(entries)==expected_rounds else None
        if accepted is not None:
            status += f'; {accepted:.3f}x'
        matching = [v['matching_tiles_reference_median_ms']/v['median_ms'] for _,v,_ in good
                    if v.get('matching_tiles_passed') and v.get('matching_tiles_reference_median_ms')]
        accepted_matching = statistics.median(matching) if len(matching)==expected_rounds and len(good)==len(entries)==expected_rounds else None
        if accepted_matching is not None:
            status += f'; same tiles {accepted_matching:.3f}x'
        median = f'{statistics.median(times):.2f}' if times else '—'
        span = f'{min(times):.2f}–{max(times):.2f}' if times else '—'
        lines.append(f'| {suite} | {name} | {mode} | {len(good)}/{expected_rounds} | {median} | {span} | {l2:.6g}' if l2 is not None else f'| {suite} | {name} | {mode} | {len(good)}/{expected_rounds} | {median} | {span} | —')
        lines[-1] += f' | {cosine:.9f} | {status} |' if cosine is not None else f' | — | {status} |'
        detailed.append({'suite':suite,'workload':name,'strategy':mode,'successful_rounds':len(good),
                         'available_rounds':len(entries),'expected_rounds':expected_rounds,'median_ms':statistics.median(times) if times else None,
                         'round_medians_ms':times,'max_rel_l2':l2,'min_cosine':cosine,
                         'max_abs_error':max_abs,'round_accuracy':accuracies,
                         'source_results':[r for r,_,_ in entries],
                         'input_sha256':sorted({v['input_sha256'] for _,v,_ in entries if v.get('input_sha256')}),
                         'assessment':status,'accepted_speedup':accepted,
                         'accepted_matching_tile_speedup':accepted_matching})
    lines += ['', 'Execution failures are retained in each suite’s execution log and results JSON. A passing block or stack comparison does not establish full text-to-video generation quality or latency.','']
    (root/'MEASURED_RESULTS.md').write_text('\n'.join(lines),encoding='utf-8')
    (root/'aggregate_results.json').write_text(json.dumps(detailed,indent=2,allow_nan=False),encoding='utf-8')
    print('\n'.join(lines))

if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('root',type=Path)
    parser.add_argument('--expected-rounds',type=int,default=3)
    args=parser.parse_args()
    summarize(args.root,args.expected_rounds)
