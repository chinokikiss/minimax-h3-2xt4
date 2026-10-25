"""Link read intervals to foreground Futures without claiming GPU idle time."""
import argparse
import hashlib
import json
from pathlib import Path

from tools.experiments.analyze_te_io import aggregate, union_ns, read_json, open_text


def dependencies(events,reads,pid,start,end):
    waits=[e for e in events if e['name']=='future_consumer_wait' and e['tid']==pid
        and e.get('ready_on_entry') is False and e['start_ns']<end and e['end_ns']>start]
    rows=[];exposed=[];wait_intervals=[]
    for wait in waits:
        future=wait.get('future');block=wait.get('block') or wait.get('legacy_block')
        ready=[e for e in events if e['name']=='future_ready_bounds' and e.get('future')==future
            and e['lower_ns']<=wait['end_ns'] and e['upper_ns']>=wait['start_ns']]
        prep=[e for e in events if e['name']=='cpu_weight_prepare' and e.get('block')==block
            and e['start_ns']<wait['end_ns'] and e['end_ns']>=wait['start_ns']]
        row=dict(future=future,block=block,start_ns=wait['start_ns'],end_ns=wait['end_ns'],
            request=wait.get('request'),consumer_tid=pid,publication=ready,
            producer_candidates=prep,read_association='unresolved',reads=[])
        wait_intervals.append((max(start,wait['start_ns']),min(end,wait['end_ns'])))
        if len(prep)==1 and len(ready)==1:
            producer=prep[0]
            selected=[r for r in reads if r.get('block')==block and r['tid']==producer['tid']
                and producer['start_ns']<=r['start_ns'] and r['end_ns']<=producer['end_ns']]
            overlap=[(max(start,wait['start_ns'],r['start_ns']),
                      min(end,wait['end_ns'],r['end_ns'])) for r in selected]
            row.update(read_association='One block/thread/time-matched producer plus exact Future publication; not a global GPU dependency DAG',
                read_summary=aggregate(selected),read_wall_overlap_with_wait_s=union_ns(overlap)/1e9,
                reads=[{k:r[k] for k in ('start_ns','end_ns','source','offset','requested_bytes','wall_ns','cpu_ns','counters','samples')} for r in selected])
            exposed+=overlap
        after=[e for e in events if e['name']=='h2d_enqueue' and e['tid']==pid
            and e['start_ns']>=wait['end_ns'] and e['start_ns']<end]
        row['next_same_thread_h2d_enqueue']=min(after,key=lambda e:e['start_ns']) if after else None
        rows.append(row)
    return dict(foreground_wait_union_s=union_ns(wait_intervals)/1e9,
        producer_read_wall_overlap_union_s=union_ns(exposed)/1e9,waits=rows,
        semantics='Rank-local control-flow associations; I/O wall overlap includes CPU work and is not removable end-to-end latency or measured GPU inactivity.')


def analyze(root):
    folder=root/'pipeline';result=json.loads((folder/'result.json').read_text())
    if not result['passed'] or not result['physical']:raise ValueError('Passed physical evidence required')
    output=dict(requests=[],analysis_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    traces={rank:read_json(folder/f'batch_r{rank}_causal.json') for rank in (0,1)}
    reads={}
    for rank in (0,1):
        with open_text(folder/f'io_reads_r{rank}.jsonl') as stream:reads[rank]=[json.loads(line) for line in stream]
    for request in result['requests']:
        a,b=(request['stages']['reference_and_TE'][k] for k in ('start_ns','end_ns'))
        output['requests'].append(dict(request=request['request'],ranks={rank:dependencies(
            traces[rank]['events'],reads[rank],traces[rank]['pid'],a,b) for rank in (0,1)}))
    (root/'io_dependencies.json').write_text(json.dumps(output,separators=(',',':')))
    print('TE I/O dependencies analyzed',len(output['requests']))
    return output


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path);analyze(parser.parse_args().root)
