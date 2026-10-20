"""Compact evidence from real TE reads and controlled file-cache probes."""
import argparse
from collections import Counter,defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import statistics


def open_text(path):
    path=Path(path)
    return path.open() if path.exists() else gzip.open(path.with_suffix(path.suffix+'.gz'),'rt')


def read_json(path):
    with open_text(path) as stream:return json.load(stream)


def union_ns(intervals):
    total=0;end=None
    for a,b in sorted(intervals):
        if b<=a:continue
        if end is None or a>end:total+=b-a;end=b
        elif b>end:total+=b-end;end=b
    return total


def aggregate(reads):
    result=dict(reads=len(reads),logical_bytes=sum(r.get('returned_bytes',0) or 0 for r in reads),
        read_wall_s=sum(r['wall_ns'] for r in reads)/1e9,
        read_cpu_s=sum(r['cpu_ns'] for r in reads)/1e9,
        nonrunning_s=sum(max(0,r['wall_ns']-r['cpu_ns']) for r in reads)/1e9,
        observer_outside_read_s=sum(r['observer_outside_read_ns'] for r in reads)/1e9,
        sample_states=dict(sum((Counter(r['samples']['states']) for r in reads),Counter())),
        sample_wchans=dict(sum((Counter(r['samples']['wchans']) for r in reads),Counter())))
    for key in ('usage','io'):
        result[key]=dict(sum((Counter(r['counters'][key]) for r in reads),Counter()))
    for label in ('before','after'):
        pages=sum(r[f'resident_{label}'].get('pages',0) for r in reads)
        resident=sum(r[f'resident_{label}'].get('resident_pages',0) for r in reads)
        result[f'resident_{label}_fraction']=resident/pages if pages else None
    result['read_union_s']=union_ns([(r['start_ns'],r['end_ns']) for r in reads])/1e9
    return result


def analyze(root):
    report=dict(status='partial',full_requests=[],probes=[],analysis_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        limits=['Thread CPU and wall are inclusive; aggregate by rank, never sum rank waits for critical path.',
            'mincore is page residency, not proof that an in-flight folio is uptodate or unlocked.',
            'Global/cgroup/disk counters are not exclusive attribution to this rank or reader.',
            'Samples are observation counts; no exact blocked duration or GIL time inferred.',
            'Probe performance is I/O/preparation evidence, not full-inference speedup.',
            'Full request diagnostics retain instrumentation overhead; prior allocations are not matched timing baselines.'])
    folder=root/'pipeline';result=read_json(folder/'result.json')
    if not result['passed'] or not result['physical']:raise ValueError('Passed physical request evidence required')
    all_reads={}
    for rank in (0,1):
        counts=json.loads((folder/f'io_counts_r{rank}.json').read_text())
        if counts['dropped']:raise ValueError('Incomplete read capture')
        with open_text(folder/f'io_reads_r{rank}.jsonl') as stream:all_reads[rank]=[json.loads(line) for line in stream]
    for request in result['requests']:
        stage=request['stages']['reference_and_TE'];a,b=stage['start_ns'],stage['end_ns']
        row=dict(request=request['request'],te_s=stage['host_wall_ms']/1000,
            e2e_s=request['latency_ms']/1000,ranks={})
        for rank,reads in all_reads.items():
            selected=[r for r in reads if a<=r['start_ns'] and r['end_ns']<=b]
            crossing=[r for r in reads if r['start_ns']<b and r['end_ns']>a and r not in selected]
            blocks=defaultdict(list)
            for read in selected:blocks[Path(read['source']).stem].append(read)
            row['ranks'][rank]=dict(summary=aggregate(selected),blocks={k:aggregate(v) for k,v in blocks.items()},
                boundary_crossing_reads=crossing,
                counter_scope='Fully contained reads; boundary-crossing reads retained separately without proportional CPU/counter attribution',
                slow_reads=sorted(selected,key=lambda r:r['wall_ns'],reverse=True)[:12])
        # Common host monotonic clock: union across ranks for overlap, not sum.
        intervals=[(max(a,r['start_ns']),min(b,r['end_ns'])) for reads in all_reads.values() for r in reads if r['start_ns']<b and r['end_ns']>a]
        row['any_rank_read_union_s']=union_ns(intervals)/1e9
        report['full_requests'].append(row)
    probe_file=root/'probe/io_probe.json'
    if probe_file.exists() or probe_file.with_suffix('.json.gz').exists():
        probe=read_json(probe_file);report['probe_status']=probe['status'];report['pressure']=probe['pressure']
        for case in probe['cases']:
            rows=case['results'];reads=[r for row in rows for block in row['blocks'] for r in block['reads']]
            record=dict(case=case['case'],ranks=case['ranks'],summary=aggregate(reads),
                blocks=[{k:v for k,v in block.items() if k!='reads'} for row in rows for block in row['blocks']],
                before=case['before'],after=case['after'])
            report['probes'].append(record)
        if probe['status']=='complete':report['status']='complete'
    (root/'io_analysis.json').write_text(json.dumps(report,separators=(',',':')))
    print('AUDIT IO ANALYSIS',report['status'],'requests',len(report['full_requests']),'probe_cases',len(report['probes']))
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path);analyze(parser.parse_args().root)
