"""Shared-device endpoint accounting and per-rank read unions, never GPU claims."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import statistics
from tools.experiments.analyze_causal_audit import duration


def analyze(folder):
    evidence=json.loads((folder/'storage_controls.json').read_text())
    capabilities=json.loads((folder/'capabilities.json').read_text())
    topology=capabilities['block_topology'];rows=[];blocks=[]
    for case in evidence['cases']:
        data=[b for r in case['ranks'] for b in r['blocks']]
        intervals=[(b['start_ns'],b['end_ns']) for b in data]
        devices={}
        for device,info in topology.items():
            if device.startswith(('loop','dm-')) or info['slaves']:continue
            before=case['before']['diskstats'].get(device);after=case['after']['diskstats'].get(device)
            if not before or not after:continue
            delta={k:after[k]-before[k] for k in ('reads','read_sectors','read_ms','weighted_busy_ms','busy_ms','write_sectors')}
            devices[device]=dict(delta,read_bytes=delta['read_sectors']*512,
                read_MB_s=delta['read_sectors']*512/1e6/case['phase_wall_s'],
                mean_read_completion_ms=delta['read_ms']/delta['reads'] if delta['reads'] else None,
                mean_outstanding_all_io=delta['weighted_busy_ms']/1000/case['phase_wall_s'])
        ranks={}
        for rank in case['ranks']:
            values=rank['blocks'];states=Counter();wchans=Counter()
            for value in values:
                states.update(value['sampled_waits']['states']);wchans.update(value['sampled_waits']['wchans'])
                blocks.append(dict(case=case['case']['name'],mode=case['case']['mode'],rank=rank['rank'],
                    block=value['block'],wall_s=value['wall_ns']/1e9,cpu_s=value['cpu_ns']/1e9,
                    logical_bytes=value['logical_bytes'],kernel_read_bytes=value['thread_io'].get('read_bytes'),
                    source_resident_fraction=value['source_residency_before'].get('resident_fraction'),
                    minor_faults=value['usage']['ru_minflt'],major_faults=value['usage']['ru_majflt'],
                    native_bit_exact=value['native_tensor_bit_exact']))
            ranks[str(rank['rank'])]=dict(read_wall_s=sum(b['wall_ns'] for b in values)/1e9,
                read_cpu_s=sum(b['cpu_ns'] for b in values)/1e9,
                logical_bytes=sum(b['logical_bytes'] for b in values),
                kernel_read_bytes=sum(b['thread_io'].get('read_bytes',0) for b in values),
                block_read_p50_s=statistics.median(b['wall_ns']/1e9 for b in values),
                minor_faults=sum(b['usage']['ru_minflt'] for b in values),
                major_faults=sum(b['usage']['ru_majflt'] for b in values),
                states=dict(states),wchans=dict(wchans))
        pages=sum(b['source_residency_before'].get('pages',0) for b in data)
        resident=sum(b['source_residency_before'].get('resident_pages',0) for b in data)
        cg={k:case['after']['memory_stat'].get(k,0)-case['before']['memory_stat'].get(k,0)
            for k in ('workingset_refault_file','pgscan_kswapd','pgsteal_kswapd','pgscan_direct','pgsteal_direct')}
        events={k:case['after']['memory_events'].get(k,0)-case['before']['memory_events'].get(k,0)
            for k in ('max','oom','oom_kill','oom_group_kill')}
        if not all(b['native_tensor_bit_exact'] for b in data):raise ValueError('Byte correctness failed')
        if any(events[k] for k in ('oom','oom_kill','oom_group_kill')):raise ValueError('OOM event; comparison ineligible')
        rows.append(dict(name=case['case']['name'],mode=case['case']['mode'],serial=case['serial'],
            phase_wall_s=case['phase_wall_s'],read_wall_union_s=duration(intervals)/1000,
            max_rank_read_wall_s=max(r['read_wall_s'] for r in ranks.values()),
            logical_bytes=sum(b['logical_bytes'] for b in data),
            resident_fraction_before=resident/pages if pages else None,devices=devices,ranks=ranks,
            cgroup_deltas=cg,memory_events=events,
            file_cache_gauges={position:{k:case[position]['memory_stat'].get(k) for k in ('anon','file','shmem','kernel')}
                for position in ('before','after')}))
    result=dict(status=evidence['status'],physical=True,cases=rows,
        anonymous_control=evidence.get('anonymous_control',{}).get('bytes'),
        limitations=['CPU/storage control, not GPU inference performance.',
            'Phase wall includes hashes, GC and observation outside timed reads; read unions do not include every phase action.',
            'Whole-device counters include shared activity. Never sum them with partitions, loop or device-mapper layers.',
            'Mean completion time/queue depth are aggregate diskstats estimates, not service time or latency percentiles.',
            'Inode eviction is not provider-cold. Natural cache state and storage supply vary.',
            'Two reversed-order passes per mode cannot establish statistically credible end-to-end improvement.'])
    (folder/'storage_summary.json').write_text(json.dumps(result,indent=2))
    if blocks:
        with (folder/'storage_blocks.csv').open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(blocks[0]));writer.writeheader();writer.writerows(blocks)
    lines=['# Physical TE storage controls','',
        'I/O-only evidence. No TE/E2E speedup claim. Phase wall includes post-read verification.',
        '', '| Case | Phase wall s | Read union s | Logical GB | Source resident | Whole-device read GB |',
        '|---|---:|---:|---:|---:|---|']
    for row in rows:
        device=', '.join(f"{k}: {v['read_bytes']/1e9:.3f}" for k,v in row['devices'].items())
        fraction=row['resident_fraction_before']
        lines.append(f"| {row['name']} | {row['phase_wall_s']:.3f} | {row['read_wall_union_s']:.3f} | {row['logical_bytes']/1e9:.3f} | {fraction:.1%} | {device} |")
    lines+=['',*('- '+item for item in result['limitations'])]
    (folder/'storage_summary.md').write_text('\n'.join(lines)+'\n')
    print('ANALYZED storage cases',len(rows),'status',result['status'],flush=True)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('folder',type=Path);analyze(parser.parse_args().folder)
