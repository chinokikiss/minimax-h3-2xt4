"""Summarize printed medians without inventing raw samples or correctness passes."""
from collections import defaultdict
import json
from pathlib import Path
import re
import statistics

ROOT = Path(__file__).resolve().parent / 'kaggle_retest_output' / 'partial'
SELECTED_MAIN = {'audio','video','video_tiling','te'}

def parse(path, allocation):
    records, pending, jobs = [], [], []
    profile = None
    for line in path.read_text(encoding='utf-8-sig').splitlines():
        match = re.match(r'\[(audio|video|te|dit)/(.+?)\] (\S+): ([\d.]+) ms',line)
        if match:
            kind, mode, workload, milliseconds = match.groups()
            suite = kind+'_stack' if mode == 'PP full 50 blocks' else kind
            if kind == 'video' and 'tiled_' in mode:
                suite = 'video_tiling'
            pending.append({'kind':kind,'inferred_suite':suite,'strategy':mode.replace('native block','native'),
                            'workload':workload,'median_ms_printed':float(milliseconds),'allocation':allocation})
        if line.startswith('BLOCK_ACCURACY '):
            match = re.match(r'BLOCK_ACCURACY (\S+) (\S+) (\S+) (\{.*\}) passed (True|False)',line)
            if match:
                kind, mode, workload, value, passed = match.groups()
                for row in reversed(pending):
                    if (row['kind'],row['strategy'],row['workload']) == (kind,mode,workload):
                        row['accuracy_printed'] = json.loads(value)
                        row['correctness_passed_printed'] = passed == 'True'
                        break
        if line.startswith('PROFILE_ACCURACY '):
            profile = json.loads(line[len('PROFILE_ACCURACY '):])
        if line.startswith('JOB_RESULT '):
            job = json.loads(line[len('JOB_RESULT '):])
            jobs.append(job)
            for row in pending:
                row.update(suite=job['suite'],round=job['round'],completed_suite=True)
                records.append(row)
            pending.clear()
    for row in pending:
        row.update(suite=row['inferred_suite'], round=jobs[-1]['round'] if jobs else 0, completed_suite=False)
        records.append(row)
    return records,jobs,profile

def main():
    rows, jobs, profile = [], {}, None
    for allocation in ('main','corrected'):
        data, completed, comparison = parse(ROOT/f'{allocation}_live.log',allocation)
        jobs[allocation] = completed
        rows.extend(row for row in data if (allocation == 'main' and row['suite'] in SELECTED_MAIN)
                    or (allocation == 'corrected' and row['suite'] in {'dit','te_stack','dit_stack'}))
        profile = comparison or profile
    grouped = defaultdict(list)
    for row in rows:
        grouped[row['suite'],row['workload'],row['strategy']].append(row)
    lines = ['# Available benchmark evidence at stop request','',
             'These are medians printed in captured live logs, rounded to 0.01 ms. The table reports the median of available round medians. Raw timing samples, output tensors, and unprinted correctness metrics have not been retrieved; no passing speedup is asserted from timing alone. Incomplete suites are marked.','',
             '| Suite | Workload | Strategy | Recorded rounds | Median ms | Timing ratio to native | Correctness evidence |',
             '|---|---|---|---:|---:|---:|---|']
    aggregate = []
    for key, entries in sorted(grouped.items()):
        suite, workload, mode = key
        times = [r['median_ms_printed'] for r in entries]
        reference = grouped.get((suite,workload,'native'),[])
        by_round = {r['round']:r['median_ms_printed'] for r in reference}
        ratios = [by_round[r['round']]/r['median_ms_printed'] for r in entries if r['round'] in by_round]
        ratio = statistics.median(ratios) if ratios and mode != 'native' else None
        metrics = [r['accuracy_printed'] for r in entries if 'accuracy_printed' in r]
        status = 'Reference' if mode == 'native' else 'Not available in log'
        if metrics:
            status = 'FAIL' if any(not r.get('correctness_passed_printed') for r in entries) else 'Printed gate PASS'
        if any(not r['completed_suite'] for r in entries):
            status += '; incomplete suite'
        lines.append(f'| {suite} | {workload} | {mode} | {len(entries)} | {statistics.median(times):.2f} | '+
                     (f'{ratio:.3f}x' if ratio is not None else '—')+f' | {status} |')
        aggregate.append({'suite':suite,'workload':workload,'strategy':mode,'recorded_rounds':len(entries),
                          'median_of_printed_medians_ms':statistics.median(times),'timing_ratio_to_native':ratio,
                          'correctness_evidence':status,'records':entries})
    record = {'evidence':'printed medians; no raw samples inferred','jobs':jobs,'selected_timing_rows':len(rows),
              'profile_accuracy':profile,'aggregate':aggregate}
    (ROOT/'available_results.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
    (ROOT/'AVAILABLE_RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('Selected timing rows:',len(rows))
    for suite in ['audio','video','video_tiling','te','dit','te_stack','dit_stack']:
        selected=[r for r in rows if r['suite']==suite]
        print(suite,'rows',len(selected),'rounds',sorted({r['round'] for r in selected}))
    print('Video profile accuracy:',json.dumps(profile))

if __name__ == '__main__':
    main()
