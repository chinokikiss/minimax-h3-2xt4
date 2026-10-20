"""Reconstruct allocated/pending/reserved bytes from bounded CUDA allocator history."""
import argparse
from collections import Counter
import gzip
import json
import html
from pathlib import Path


def write_timeline(directory):
    roles={}
    for path in sorted(directory.glob('*/*.json.gz')):
        with gzip.open(path,'rt') as stream:payload=json.load(stream)
        role=payload.get('role',path.parent.name)
        roles.setdefault(role,[]).extend(payload['markers'])
    roles={k:sorted(v,key=lambda x:x['time_ns']) for k,v in roles.items() if v}
    if not roles:return
    origin=min(rows[0]['time_ns'] for rows in roles.values())
    duration=max((rows[-1]['time_ns']-origin)/1e9 for rows in roles.values()) or 1
    width=1120;height=95+len(roles)*210
    parts=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="20" y="24" font-size="18" font-family="sans-serif">Offline L1 GPU memory: sampled phase boundaries</text>',
        '<text x="20" y="47" font-size="12" font-family="sans-serif">Gray: reserved; blue: allocated; green: registered weights; orange: remaining allocated (includes inputs/scratch).</text>',
        '<text x="20" y="65" font-size="12" font-family="sans-serif">Allocator histories capture short-lived peaks between these markers. Hover over points for phase metadata.</text>']
    for index,(role,rows) in enumerate(sorted(roles.items())):
        top=90+index*210;left=85;plot_width=1000;plot_height=150
        ymax=max(15.,max(row['reserved']/2**30 for row in rows))
        stride=max(1,len(rows)//1500);sampled=rows[::stride]
        if sampled[-1] is not rows[-1]:sampled.append(rows[-1])
        points={k:[] for k in ('reserved','allocated','weights','remaining')}
        last_weights=0
        for row in sampled:
            if 'known_static_storage_bytes' in row:last_weights=sum(row['known_static_storage_bytes'].values())
            x=left+(row['time_ns']-origin)/1e9/duration*plot_width
            values=dict(reserved=row['reserved'],allocated=row['allocated'],weights=last_weights,
                        remaining=max(0,row['allocated']-last_weights))
            for key,value in values.items():points[key].append(f'{x:.2f},{top+plot_height-value/2**30/ymax*plot_height:.2f}')
            if len(points['allocated'])%8==0:
                y=top+plot_height-row['allocated']/2**30/ymax*plot_height
                title=html.escape(f"{role}, {row['command']}, NFE{row['nfe']}, block{row['block']}, {row['label']}; allocated {row['allocated']/2**30:.3f} GiB")
                parts.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="3" fill="#2563eb"><title>{title}</title></circle>')
        parts.append(f'<text x="20" y="{top+15}" font-size="13" font-family="sans-serif">{html.escape(role)}</text>')
        for tick in (0,4,8,12):
            y=top+plot_height-tick/ymax*plot_height
            parts.append(f'<line x1="{left}" y1="{y:.2f}" x2="{left+plot_width}" y2="{y:.2f}" stroke="#e5e7eb"/>')
            parts.append(f'<text x="45" y="{y+4:.2f}" font-size="11">{tick} GiB</text>')
        for key,color in (('reserved','#64748b'),('allocated','#2563eb'),('weights','#16a34a'),('remaining','#ea580c')):
            parts.append(f'<polyline points="{" ".join(points[key])}" fill="none" stroke="{color}" stroke-width="1.4"/>')
        for tick in range(6):
            x=left+tick/5*plot_width
            parts.append(f'<text x="{x-12:.2f}" y="{top+plot_height+20}" font-size="11">{duration*tick/5:.0f}s</text>')
    parts.append('</svg>')
    (directory/'memory_timeline.svg').write_text('\n'.join(parts),encoding='utf8')


def analyze(payload):
    epochs=[];live={};pending={};reserved=0;serial=0;by_address={}
    def create(addr,size,birth,stack=None):
        nonlocal serial
        item=dict(id=serial,addr=addr,bytes=size,birth_ns=birth,stack=stack,category='temporary_or_unattributed')
        epochs.append(item);by_address.setdefault(addr,[]).append(item);serial+=1;return item
    for segment in payload['initial_segments']:
        reserved+=segment['total_size']
        for block in segment['blocks']:
            if block['state']=='inactive':continue
            item=create(block['address'],block['size'],None)
            item['category']='preexisting_other_storage'
            (live if block['state']=='active_allocated' else pending)[item['addr']]=item
    combined=[]
    for e in payload['events']:
        if e.get('time_us') is not None:combined.append((e['time_us']*1000,0,e))
    for m in payload['markers']:combined.append((m['time_ns'],1,m))
    combined.sort(key=lambda e:(e[0],e[1]))
    peak=dict(allocated=0,active=0,reserved=reserved,live_ids=[])
    active_peak=sum(x['bytes'] for x in [*live.values(),*pending.values()]);reserved_peak=reserved
    unknown_frees=0;last_marker={};phases={}
    for stamp,kind,event in combined:
        if kind==1:
            last_marker=event
            table=payload['static_tables'][event['static_table']] if 'static_table' in event else []
            for descriptor in table+event.get('inputs',[]):
                item=live.get(descriptor['addr']) or pending.get(descriptor['addr'])
                if item is not None:
                    item['category']=descriptor.get('category','request_or_block_input')
                    item['label']=descriptor.get('key','block input')
            continue
        action=event['action'];addr=event.get('addr');size=event.get('size',0)
        if action=='alloc':live[addr]=create(addr,size,stamp,event.get('stack'))
        elif action=='free_requested':
            item=live.pop(addr,None)
            if item is None:unknown_frees+=1
            else:item['free_requested_ns']=stamp;pending[addr]=item
        elif action=='free_completed':
            item=pending.pop(addr,None)
            if item is None:unknown_frees+=1
            else:item['free_completed_ns']=stamp
        elif action=='segment_alloc':reserved+=size
        elif action=='segment_free':reserved-=size
        allocated=sum(x['bytes'] for x in live.values())
        active=allocated+sum(x['bytes'] for x in pending.values())
        active_peak=max(active_peak,active);reserved_peak=max(reserved_peak,reserved)
        if allocated>peak['allocated']:
            peak=dict(time_ns=stamp,allocated=allocated,active=active,reserved=reserved,
                phase=last_marker.get('label'),block=last_marker.get('block'),live_ids=[x['id'] for x in live.values()])
        phase=(last_marker.get('block'),tuple(last_marker.get('scope',[])))
        phases[phase]=max(phases.get(phase,0),allocated)
    for shape in payload['shape_observations']:
        matches=[e for e in by_address.get(shape['addr'],[]) if
            (e['birth_ns'] is None or e['birth_ns']<=shape['time_ns']) and
            shape['time_ns']<=e.get('free_requested_ns',float('inf'))]
        if matches:
            matches[-1].setdefault('observed_shape',shape['shape'])
            matches[-1].setdefault('observed_operator',shape['operator'])
    categories=Counter()
    for index in peak['live_ids']:categories[epochs[index]['category']]+=epochs[index]['bytes']
    peak['category_bytes']=dict(categories)
    peak.pop('live_ids')
    largest=sorted((e for e in epochs if e['category']=='temporary_or_unattributed'),key=lambda e:e['bytes'],reverse=True)[:16]
    for item in largest:
        stack=item.get('stack')
        item['source_stack']=payload['stacks'][stack] if stack is not None else []
    return dict(rank=payload['rank'],role=payload.get('role'),pid=payload.get('pid'),command=payload['command'],nfe=payload['nfe'],peak=peak,
        peak_active_bytes=active_peak,peak_reserved_bytes=reserved_peak,
        actions=dict(Counter(e['action'] for e in payload['events'])),
        history_limit_reached=payload['history_limit_reached'],unknown_free_events=unknown_frees,
        timestamped_events=sum(e.get('time_us') is not None for e in payload['events']),
        events=len(payload['events']),largest_temporary_allocations=largest,
        phase_peaks=[dict(block=k[0],scope=list(k[1]),allocated_bytes=v) for k,v in phases.items()],
        final_known_static=next((m for m in reversed(payload['markers']) if 'static_table' in m),None),
        meaning='Allocation/free event replay; unknown-free or truncated histories limit reconstruction. '
            'Temporary_or_unattributed includes library scratch and inputs not observed at a block boundary. Static classifications are address-lifetime matched, not summed maxima.')


def main():
    p=argparse.ArgumentParser();p.add_argument('--directory',type=Path,required=True);args=p.parse_args()
    summaries=[]
    for path in sorted(args.directory.glob('*/*.json.gz')):
        with gzip.open(path,'rt') as stream:payload=json.load(stream)
        row=analyze(payload);row['file']=path.relative_to(args.directory).as_posix();summaries.append(row)
    args.directory.mkdir(parents=True,exist_ok=True)
    (args.directory/'summary.json').write_text(json.dumps(summaries,indent=2))
    lines=['# GPU memory overhead profile','',
        'Diagnostic run: profiling overhead is included in its latency. Allocator traces cover TE, DiT and representative VAE invocations; ordinary per-command peaks and NVML sampling cover all invocations.',
        '', '| Process role | Command | NFE | Allocated peak GiB | Active peak GiB | Reserved peak GiB | Events | Truncated |',
        '|---|---|---:|---:|---:|---:|---:|---|']
    for r in summaries:
        lines.append(f"| {r['role']} | {r['command']} | {r['nfe']} | {r['peak']['allocated']/2**30:.3f} | {r['peak_active_bytes']/2**30:.3f} | {r['peak_reserved_bytes']/2**30:.3f} | {r['events']} | {r['history_limit_reached']} |")
    lines+=['','Each chunk records initial/final allocator segments, exact alloc/free_requested/free_completed events, source stack IDs, phase markers, deduplicated model/window/retired-weight inventories and tensor shape observations for blocks0/24/49.',
        '', 'CUDA free completion and cached reserved memory are distinct from host release. Static weights, persistent block inputs, and temporary/unattributed bytes are separated at each reconstructed peak. NVML-only/context/NCCL allocations are outside the PyTorch allocator and must be read from the separate board monitor.',
        '', 'See summary.json for peak categories, largest allocation lifetimes, per-phase peaks and history completeness.']
    (args.directory/'MEMORY_PROFILE.md').write_text('\n'.join(lines)+'\n')
    write_timeline(args.directory)
    print('MEMORY_ANALYSIS_COMPLETE',len(summaries),'chunks',flush=True)


if __name__=='__main__':main()
