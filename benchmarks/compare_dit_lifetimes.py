"""Compare returned T4 lifetime profiles without treating cache growth as workspace."""
import argparse
from collections import Counter
import gzip
import json
from pathlib import Path


def load(path):
    raw=path.read_bytes()
    return json.loads(gzip.decompress(raw) if path.suffix=='.gz' else raw)


def live_at(data,stamp):
    return [e for e in data['epochs'] if (e['birth_ns'] is None or e['birth_ns']<=stamp)
            and (e['host_release_ns'] is None or stamp<e['host_release_ns'])]


def categories_at(data,stamp):
    counts=Counter()
    for e in live_at(data,stamp):
        history=[h for h in e['category_history'] if h['time_ns']<=stamp]
        category=history[-1]['category'] if history else ('preexisting_unclassified' if e['initial'] else 'activation_or_temporary')
        counts[category]+=e['bytes']
    return dict(counts)


def phase_key(scope):
    if not scope:return 'Norm/residual outside projections'
    if scope[0]=='AttentionO' and 'row_global_quantizer' in scope:return 'AttentionO_quantization'
    return scope[0]


def rank_summary(raw_dir,analysis,rank):
    data=load(analysis/f'rank{rank}_002_denoise_nfe1_lifetimes.json.gz')
    raw=load(raw_dir/f'dit_memory/gpu_memory/rank{rank}/002_denoise_nfe1.json.gz')
    phases={}
    for phase in data['phase_peaks']:
        key=phase_key(phase['scope'])
        if key=='marker_limit_reached':continue
        if phase['allocated_bytes']>phases.get(key,{}).get('requested_bytes',-1):
            phases[key]=dict(requested_bytes=phase['allocated_bytes'],block=phase['block'],
                categories=categories_at(data,phase['time_ns']))
    boundaries=[]
    for block in (0,24,49):
        qkv=next(c for c in data['column_output_allocations'] if c['projection']=='QKV' and c['derived_layer']==block)
        epoch=next(e for e in data['epochs'] if e['address']==qkv['output_address'] and e['birth_ns']==qkv['time_ns'])
        marker=next((m for m in raw['markers'] if m.get('block')==block and m['label']=='AttentionO/begin'),None)
        quant=next((p for p in data['phase_peaks'] if p['block']==block
                    and phase_key(p['scope'])=='AttentionO_quantization'),None)
        cat=next(c for c in data['column_output_allocations'] if c['projection']=='FC1' and c['derived_layer']==block)
        internal=[e for e in live_at(data,cat['time_ns']) if 'FC1' in e.get('birth_scope',[])]
        boundaries.append(dict(block=block,
            qkv_reusable_before_O=(epoch['cuda_reusable_ns'] is not None and epoch['cuda_reusable_ns']<marker['time_ns']) if marker else None,
            qkv_reusable_before_O_quant_peak=(epoch['cuda_reusable_ns'] is not None
                and epoch['cuda_reusable_ns']<quant['time_ns']) if quant else None,
            qkv_release_source=[f for f in epoch.get('release_source_stack',[]) if f[0].endswith('tp_dit.py')],
            fc1_internal_at_cat_bytes=sum(e['bytes'] for e in internal),
            padded_int32_alive_at_cat=any(e['bytes']==51232*14336*4 for e in internal),
            rotated_input_alive_at_cat=any(e['bytes']==51220*5376*2 for e in internal),
            codes_alive_at_cat=any(e['bytes']==51220*5376 for e in internal)))
    return dict(rank=rank,requested_peak=data['peak']['allocated_bytes'],
        peak_categories=categories_at(data,data['peak']['time_ns']),phases=phases,boundaries=boundaries,
        complete_history=data['complete_prefix_history'],unknown_frees=data['unknown_free_events'],
        history_limit_reached=data['history_limit_reached'],marker_limit_reached=data['marker_limit_reached'],
        fc1_outputs=data['column_output_counts'].get('FC1'),qkv_outputs=data['column_output_counts'].get('QKV'))


def main():
    p=argparse.ArgumentParser()
    for name in ('baseline_raw','baseline_analysis','candidate_raw','candidate_analysis','output'):
        p.add_argument('--'+name.replace('_','-'),type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    result=dict(scope='Actual T4 DiT-only profiles, seeded shape inputs/base weights; no real-generation acceptance',
                accounting='Phase peaks include resident weights; paired column peaks are incremental allocated bytes',runs={})
    for name,raw,analysis in [('baseline',args.baseline_raw,args.baseline_analysis),
                              ('lifetime',args.candidate_raw,args.candidate_analysis)]:
        run=load(raw/'dit_memory/result.json')
        assert run['status']=='profile_completed'
        assert all(h['capability']==[7,5] and 'T4' in h['name'] for h in run['hardware'])
        counters=[r['benchmark']['commands'][-1] for r in run['ranks']]
        result['runs'][name]=dict(hardware=run['hardware'],torch=run['torch'],cuda=run['cuda'],
            allocator_peak_bytes=[r['peak_allocated'] for r in counters],
            reserved_peak_bytes=[r['peak_reserved'] for r in counters],
            ranks=[rank_summary(raw,analysis,r) for r in (0,1)])
    result['paired_columns']=load(args.candidate_raw/'dit_memory/column_gate.json')
    assert result['paired_columns']['t4_qualified']
    (args.output/'comparison.json').write_text(json.dumps(result,indent=2))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    metrics=[]
    for name in ('FC1','QKV'):
        column=result['paired_columns']['operators'][name]
        metrics.append((name+' incremental allocated',*[max(column[k]['peak_increment_bytes'])/2**30 for k in ('baseline','lifetime')]))
    metrics.append(('Full DiT allocated peak',*[max(result['runs'][k]['allocator_peak_bytes'])/2**30 for k in ('baseline','lifetime')]))
    for phase,label in [('FC1','FC1 requested live, incl. weights'),('AttentionO_quantization','O quant. requested live, incl. weights')]:
        metrics.append((label,*[max(r['phases'][phase]['requested_bytes'] for r in result['runs'][k]['ranks'])/2**30 for k in ('baseline','lifetime')]))
    fig,ax=plt.subplots(figsize=(12,6),layout='constrained')
    names=[m[0] for m in metrics];ys=list(range(len(metrics)))
    for offset,index,label,color in [(-.18,1,'Before','#64748b'),(.18,2,'Shorter lifetimes','#2563eb')]:
        values=[m[index] for m in metrics]
        bars=ax.barh([y+offset for y in ys],values,height=.34,label=label,color=color)
        ax.bar_label(bars,fmt='%.2f',padding=3)
    ax.set_yticks(ys,names);ax.invert_yaxis();ax.set(xlabel='GiB — accounting boundary is stated in each row',
        title='51,220 tokens: lifetime-only changes on actual T4',xlim=(0,max(m[i] for m in metrics for i in (1,2))*1.16))
    ax.legend(loc='lower right');ax.grid(axis='x',alpha=.2);ax.set_axisbelow(True)
    for ext in ('png','svg'):fig.savefig(args.output/('memory_comparison.'+ext),dpi=160)
    print('DIT_LIFETIME_COMPARISON_READY',args.output)


if __name__=='__main__':main()
