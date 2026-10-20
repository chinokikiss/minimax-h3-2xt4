"""Compact physical residency comparison; no device/rank overlap double count."""
import argparse
import gzip
import json
from pathlib import Path


def read(path):
    if path.exists():return json.loads(path.read_text())
    with gzip.open(str(path)+'.gz','rt') as stream:return json.load(stream)


def analyze(root):
    manifest=read(root/'residency_ablation.json')
    output=dict(status=manifest['status'],failures=manifest['failures'],physical=True,
        timing_requests=0,profile_requests=sum(v['requests'] for v in manifest['profiles'].values()),
        cases={},profiles={},limitations=[
            'One allocation, two/three requests per timing policy; no statistical/general workload claim.',
            'Fresh managed caches, uncontrolled OS/provider cache; counterbalanced order.',
            'sda whole-device reads are shared endpoint counters, not exclusive model I/O. Do not sum partitions or dm/loop layers.',
            'Cache eviction-before-use counts CPU borrowing, not GPU stall causality.',
            'Warm TE profile is a separate diagnostic pass; no GPU idle measurement outside TE; category unions overlap.'])
    for phase in ('short','continuous'):
        baseline=manifest['results'][phase+'/baseline']
        for policy in ('baseline','p0','p0_p1'):
            result=read(root/phase/policy/'timing/result.json')
            rows=result['requests'];warm=rows[1:];value=manifest['results'][phase+'/'+policy]
            samples=[s for s in result['host_monitor']['samples'] if any(
                r['start_ns']<=s['monotonic_ns']<=r['end_ns'] for r in rows)]
            windows=[]
            for rank in result['runtime']:
                for event in rank['events']:
                    if event.get('phase')=='dit' and 'window' in event and event['calls']>=8:
                        window=event['window']
                        windows.append(dict(rank=rank['rank'],cumulative_nfe=event['calls'],
                            h2d_gb=window['h2d_bytes']/1e9,gpu_hits=window['hits'],loads=sum(window['loads'].values())))
            state=result['host_budget_after_close']
            case=dict(requests=len(rows),passed=result['passed'],
                native_correct=all(all(c['passed'] for c in r.values()) for r in result['correctness']),
                cold_te_s=value['cold_te_ms']/1000,cold_e2e_s=value['cold_e2e_ms']/1000,
                warm_te_s=value['warm_te_ms']/1000,warm_e2e_s=value['warm_e2e_ms']/1000,
                warm_te_reduction_percent=100*(1-value['warm_te_ms']/baseline['warm_te_ms']),
                warm_e2e_reduction_percent=100*(1-value['warm_e2e_ms']/baseline['warm_e2e_ms']),
                warm_stages_s={s:sum(r['stages'][s]['host_wall_ms'] for r in warm)/len(warm)/1000
                    for s in warm[0]['stages']},
                warm_sda_read_gb=sum(r['host_io']['diskstats']['sda']['read_sectors'] for r in warm)*512/len(warm)/1e9,
                warm_refault_file_pages=sum(r['host_io']['memory_stat']['workingset_refault_file'] for r in warm)/len(warm),
                memory_events_delta={k:sum(r['host_io']['memory_events'].get(k,0) for r in rows)
                    for k in ('oom','oom_kill','max')},
                peak_cgroup_gib=max((s['cgroup']['current']/2**30 for s in samples
                    if s.get('cgroup',{}).get('available')),default=None),
                peak_gpu_allocated_gib=[r['gpu']['allocated_peak']/2**30 for r in result['runtime']],
                peak_host_reserved_gib=state['peak_reserved_bytes']/2**30,
                closed_leases={k:state[k] for k in ('reserved_bytes','preparing','pending_preparation_bytes')},
                same_decode_paths=all(r['decode']['video']['strategy']=='native_aligned_query' and
                    r['decode']['audio']['strategy']=='channel_parallel' and all(
                    not d['fallback'] for d in r['decode'].values()) for r in rows),
                per_rank_cache=[{k:r['host_cache'][k] for k in ('hits','misses','evictions','evictions_before_use',
                    'future_wait_ms','pressure_yields')} for r in result['runtime']],warm_dit_windows=windows)
            output['cases'][phase+'/'+policy]=case;output['timing_requests']+=len(rows)
    for policy in ('baseline','p0','p0_p1'):
        trace=read(root/'diagnostic'/policy/'kineto/te_critical_path.json')
        output['profiles'][policy]=dict(warm_request=trace['requests'][-1],
            prefix=[b for b in trace['blocks'] if b['request']==1 and b['block']<2])
    (root/'residency_summary.json').write_text(json.dumps(output,indent=2))
    print(json.dumps({k:{x:v[x] for x in ('warm_te_s','warm_e2e_s','warm_e2e_reduction_percent',
        'warm_sda_read_gb','warm_refault_file_pages','memory_events_delta','closed_leases')}
        for k,v in output['cases'].items()},indent=2))
    return output


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path)
    analyze(parser.parse_args().root)
