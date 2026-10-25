"""Evidence-bound DiT memory summary and FC1 allocation diagrams, offline only."""
import argparse
from bisect import bisect_left
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path


def load(path):
    if path.suffix=='.gz':return json.loads(gzip.decompress(path.read_bytes()))
    return json.loads(path.read_text())


def live_at(epochs,stamp):
    return [e for e in epochs if (e['birth_ns'] is None or e['birth_ns']<=stamp)
            and (e['host_release_ns'] is None or stamp<e['host_release_ns'])]


def main():
    p=argparse.ArgumentParser();p.add_argument('--raw',type=Path,required=True)
    p.add_argument('--analysis',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    result=load(args.raw/'dit_memory/result.json')
    digest=hashlib.sha256((args.raw/'download_manifest.json').read_bytes()).hexdigest()
    report=dict(status=result['status'],hardware=result['hardware'],torch=result['torch'],cuda=result['cuda'],
        input_scope=result['input_distribution'],weight_scope=result['weights'],stages=result['stages'],
        download_manifest_sha256=digest,raw_history_complete=True,rank_results=[],
        fc1_component_bytes={},candidate_changes=False)
    summaries=load(args.analysis/'summary.json');max_phases={};representative=None;raw_rep=None
    for rank in (0,1):
        row=next(x for x in summaries if x['rank']==rank and x['command']=='denoise')
        data=load(args.analysis/f'rank{rank}_002_denoise_nfe1_lifetimes.json.gz')
        raw=load(args.raw/f'dit_memory/gpu_memory/rank{rank}/002_denoise_nfe1.json.gz')
        counters=result['ranks'][rank]['benchmark']['commands'][-1]
        record=dict(rank=rank,allocator_peak_bytes=counters['peak_allocated'],
            reserved_peak_bytes=counters['peak_reserved'],requested_replay_peak_bytes=data['peak']['allocated_bytes'],
            allocation_history_events=len(raw['events']),unknown_free_events=data['unknown_free_events'],
            missing_timestamps=data['missing_timestamps'],history_limit_reached=data['history_limit_reached'],
            verbose_marker_cap_reached=data['marker_limit_reached'],verified_checkpoint_markers=row['verified_checkpoint_markers'],
            qkv_outputs=data['column_output_counts'].get('QKV'),fc1_outputs=data['column_output_counts'].get('FC1'),
            snapshots=[])
        report['raw_history_complete'] &= data['complete_prefix_history']
        for block in (0,24,49):
            cat=next(c for c in data['column_output_allocations'] if c['projection']=='FC1' and c['derived_layer']==block)
            qkv=next(c for c in data['column_output_allocations'] if c['projection']=='QKV' and c['derived_layer']==block)
            q_epoch=next(e for e in data['epochs'] if e['address']==qkv['output_address'] and e['birth_ns']==qkv['time_ns'])
            prefix=load(args.raw/f'dit_memory/gpu_memory/rank{rank}/checkpoint_denoise_nfe1_b{block}_fc1_before.json.gz')
            before=raw['events'][len(prefix['events'])]['time_us']*1000
            groups=Counter()
            for e in prefix['static_tables'][-1]:groups[e['key']]+=e['bytes']
            record['snapshots'].append(dict(block=block,cat_requested_live_bytes=cat['allocated_bytes'],
                category_bytes=cat['category_bytes'],qkv_released_before_fc1=q_epoch['host_release_ns']<before,
                qkv_cuda_reusable_before_fc1=(q_epoch['cuda_reusable_ns'] is not None
                    and q_epoch['cuda_reusable_ns']<before),
                refiner_weight_bytes=sum(v for k,v in groups.items() if 'dit_refine' in k),
                registered_weight_groups=dict(groups)))
        report['rank_results'].append(record)
        for phase in data['phase_peaks']:
            label='/'.join(phase['scope'])
            if not label:label='Norm/residual outside projection scopes'
            if label and label!='marker_limit_reached':max_phases[label]=max(max_phases.get(label,0),phase['allocated_bytes'])
        if rank==1:representative=data;raw_rep=raw
    cat=next(c for c in representative['column_output_allocations'] if c['projection']=='FC1' and c['derived_layer']==24)
    at_peak=live_at(representative['epochs'],cat['time_ns'])
    internal=[e for e in at_peak if 'FC1' in e.get('birth_scope',[])]
    def category(item):
        if item['address']==cat['output_address']:return 'Contiguous FP16 output'
        stack=item['source_stack']
        lines={f[1] for f in stack if f[0].endswith('quantization.py') and f[2]=='int8_linear'}
        if 1052 in lines:return 'Padded INT32 accumulator'
        if 1072 in lines:return 'FP16 scaled chunk list'
        if 1042 in lines:return 'Rotated FP16 input'
        if 1048 in lines and item['bytes']>2**20:return 'INT8 activation codes'
        if 1065 in lines:return 'Last FP32 result chunk'
        if 1068 in lines:return 'Last FP32 scale matrix'
        return 'Small FC1 scale buffers'
    parts=Counter()
    for item in internal:parts[category(item)]+=item['bytes']
    report.update(fc1_component_bytes=dict(parts),fc1_internal_live_bytes=sum(parts.values()),
        representative_rank=1,representative_block=24,representative_total_requested_bytes=cat['allocated_bytes'],
        phase_requested_peaks=max_phases,
        attribution_limits=['Verbose line markers capped during block 37; complete allocation/free history continues through all 50 blocks.',
            'Layer numbering of 50 FC1/QKV outputs derives from exact output sizes, eager cat call stacks and execution order; snapshots validate blocks 0/24/49.',
            'Requested-byte replay differs from allocator counters because rounded/reused allocation blocks include additional slack.',
            'Timing includes tracing and snapshot export; no latency speedup or real E2E success is established.'])
    (args.output/'profile_summary.json').write_text(json.dumps(report,indent=2))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    labels=['FC1','Attention output quantization','Norm/residual outside modules','Q/K norm + RoPE','QKV','Sparse attention','FC2']
    keys=['FC1','AttentionO/FC2_global_quantizer/row_global_quantizer',None,'RoPE','QKV','Attention','FC2']
    values=[]
    for key in keys:
        if key:values.append(max_phases[key]/2**30)
        else:values.append(max(v['allocated_bytes'] for s in summaries if s['command']=='denoise' for v in s['phase_peaks'] if not v['scope'])/2**30)
    fig,axes=plt.subplots(1,2,figsize=(14,6),layout='constrained')
    fig.get_layout_engine().set(rect=(0,.08,1,.92))
    axes[0].barh(labels[::-1],values[::-1],color=['#64748b']*6+['#dc2626'])
    for i,v in enumerate(values[::-1]):axes[0].text(v+.06,i,f'{v:.2f}',va='center',fontsize=9)
    axes[0].set(xlim=(0,13.7),xlabel='Requested live GiB, including resident weights',title='DiT phase peaks — actual dual T4')
    names=list(parts);numbers=[parts[n]/2**30 for n in names]
    axes[1].barh(names[::-1],numbers[::-1],color='#2563eb')
    for i,v in enumerate(numbers[::-1]):axes[1].text(v+.03,i,f'{v:.3f}' if v>=.01 else f'{v*1024:.3f} MiB',va='center',fontsize=9)
    axes[1].set(xlim=(0,3.2),xlabel='GiB alive at FP16 output concatenation',title='FC1 internal storage: 6.712 GiB')
    for ax in axes:ax.grid(axis='x',alpha=.2);ax.set_axisbelow(True)
    fig.text(.015,.015,'51,220 tokens; FP16-fix + scalar attention + fused-stored + scaled FP16 transport.\nSeeded shape workload with trained base weights. Allocation evidence; no real-generation or latency acceptance.',fontsize=9)
    for ext in ('png','svg'):fig.savefig(args.output/('fc1_memory_breakdown.'+ext),dpi=160)
    plt.close(fig)
    # Operator-only event-order view excludes the before/after snapshot I/O.
    starts=[m for m in raw_rep['markers'] if m['block']==24 and m['label']=='FC1/begin']
    ends=[m for m in raw_rep['markers'] if m['block']==24 and m['label']=='FC1/end']
    start,end=starts[0]['time_ns'],ends[0]['time_ns']
    points=[p for p in representative['points'] if start<=p['time_ns']<=end]
    times=[p['time_ns'] for p in points]
    fig,axes=plt.subplots(2,1,figsize=(12,8),layout='constrained',gridspec_kw={'height_ratios':[1,2]})
    fig.get_layout_engine().set(rect=(0,.06,1,.94))
    axes[0].step(range(len(points)),[p['allocated_bytes']/2**30 for p in points],where='post',color='#2563eb')
    peak_index=bisect_left(times,cat['time_ns'])
    axes[0].axvline(peak_index,color='#dc2626',linestyle='--',label='Final FP16 output allocated')
    axes[0].legend(loc='upper left');axes[0].set(ylabel='Requested live GiB',title='FC1 allocation / release order — T4 rank 1, block 24')
    grouped={}
    for item in internal:grouped.setdefault(category(item),[]).append(item)
    for y,(name,items) in enumerate(grouped.items()):
        left=min(max(0,bisect_left(times,e['birth_ns'] or start)) for e in items)
        right=max(min(len(points),bisect_left(times,e['host_release_ns'] or end)) for e in items)
        axes[1].barh(y,max(.5,right-left),left=left,color='#2563eb',height=.6)
    axes[1].set_yticks(range(len(grouped)),[f'{n} ({parts[n]/2**30:.3f} GiB)' if parts[n]>=.01*2**30 else f'{n} ({parts[n]/2**20:.3f} MiB)' for n in grouped],fontsize=9)
    axes[1].invert_yaxis();axes[1].axvline(peak_index,color='#dc2626',linestyle='--')
    axes[1].set(xlabel='Allocator event order inside FC1 (not elapsed time)',title='Host-visible storage lifetime; chunk list shows first birth–last release')
    for ax in axes:ax.grid(axis='x',alpha=.2);ax.set_axisbelow(True)
    fig.text(.015,.01,'Aliases counted once. Old INT32 output is released after the final FP16 allocation; rotated input, INT8 codes and scaled parts survive until return.\nThe contiguous output remains live into FC2; its bar is clipped at the end of this FC1 window.',fontsize=8)
    for ext in ('png','svg'):fig.savefig(args.output/('fc1_lifetime_order.'+ext),dpi=160)
    plt.close(fig)
    print('DIT_MEMORY_SUMMARY_READY',report['status'],report['fc1_internal_live_bytes'])


if __name__=='__main__':main()
