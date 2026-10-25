"""Offline storage lifetimes, aliases and phase overlap from allocator evidence."""
import argparse
from collections import Counter, defaultdict
import csv
import gzip
import json
from pathlib import Path
import sys
import re


def add_checkpoint_boundaries(payload,path):
    """Recover representative FC1 boundaries if the verbose marker cap filled.

    Each checkpoint is a verified prefix of the final allocation history. The
    next final-history event is its snapshot event, providing its timestamp.
    Raw files are never rewritten. Unobserved line ranges remain unlabelled.
    """
    payload['_original_marker_end_ns']=payload['markers'][-1]['time_ns'] if len(payload['markers'])>=12000 else None
    windows={};added=[]
    for checkpoint in sorted(path.parent.glob(f'checkpoint_{payload["command"]}_nfe{payload["nfe"]}_*.json.gz')):
        with gzip.open(checkpoint,'rt') as stream:prefix=json.load(stream)
        n=len(prefix['events'])
        if n>=len(payload['events']):continue
        event=payload['events'][n]
        assert event['action']=='snapshot','Checkpoint boundary is not present in final history'
        # PyTorch recalibrates approximate timestamps at each snapshot; old
        # events can shift by a few microseconds. Match event identity/order,
        # then use the final history's single calibrated time base.
        keys=('action','addr','size','stream')
        assert all(all(a.get(k)==b.get(k) for k in keys) for a,b in zip(prefix['events'],payload['events'][:n]))
        match=re.fullmatch(r'b(\d+)_fc1_(before|after)',prefix['checkpoint_label'])
        if not match:continue
        block=int(match[1]);when=match[2];stamp=event['time_us']*1000
        table=len(prefix['static_tables'])-1
        assert prefix['static_tables'][-1]==payload['static_tables'][table]
        added.append(dict(time_ns=stamp,label='verified_checkpoint/'+prefix['checkpoint_label'],block=block,
            scope=['FC1'] if when=='before' else [],static_table=table,derived_from=checkpoint.name))
        windows.setdefault(block,{})[when]=stamp
    payload['markers']=sorted([*payload['markers'],*added],key=lambda m:m['time_ns'])
    payload['_verified_fc1_windows']=[dict(block=b,**v) for b,v in windows.items() if 'before'in v and 'after'in v]
    payload['_verified_checkpoint_markers']=len(added)
    return payload


def lifetimes(payload):
    epochs, live, pending = [], {}, {}
    segments={s['address']:dict(address=s['address'],bytes=s['total_size'],stream=s.get('stream'))
              for s in payload['initial_segments']}
    ooms=[];cats=[];cat_counts=Counter()
    marker_end=payload.get('_original_marker_end_ns',payload['markers'][-1]['time_ns'] if len(payload['markers'])>=12000 else None)
    marker_limit=marker_end is not None
    def add(address, size, birth, stack=None, initial=False):
        item = dict(id=len(epochs), address=address, bytes=size, birth_ns=birth,
                    host_release_ns=None, cuda_reusable_ns=None, initial=initial,
                    stack=stack, aliases=[], category='activation_or_temporary',
                    birth_phase=None, free_phase=None,category_history=[])
        if initial:item['category']='preexisting_unclassified'
        epochs.append(item)
        return item
    for segment in payload['initial_segments']:
        for block in segment['blocks']:
            if block['state'] == 'inactive':
                continue
            item = add(block['address'], block['size'], None, initial=True)
            (live if block['state']=='active_allocated' else pending)[item['address']] = item
    timeline = [(e['time_us']*1000,0,e) for e in payload['events'] if 'time_us' in e]
    timeline += [(m['time_ns'],1,m) for m in payload['markers']]
    timeline += [(m['time_ns'],2,m) for m in payload['shape_observations']]
    timeline.sort(key=lambda row: (row[0],row[1]))
    phase = None
    scope=[];block_index=None;phase_peaks={}
    peak = dict(allocated_bytes=sum(x['bytes'] for x in live.values()), live_ids=[i['id'] for i in live.values()])
    unknown_frees = 0
    points = []
    for stamp,kind,event in timeline:
        if kind == 1:
            phase = event['label']
            scope=list(event.get('scope',[]));block_index=event.get('block')
            if 'static_table' in event:
                for descriptor in payload['static_tables'][event['static_table']]:
                    item = live.get(descriptor['addr']) or pending.get(descriptor['addr'])
                    if item:
                        item.update(category=descriptor['category'], weight_key=descriptor['key'])
                        row=dict(time_ns=stamp,category=descriptor['category'],weight_key=descriptor['key'])
                        if not item['category_history'] or any(row[k]!=item['category_history'][-1][k] for k in ('category','weight_key')):
                            item['category_history'].append(row)
            continue
        if kind == 2:
            item = live.get(event['addr']) or pending.get(event['addr'])
            if item:
                alias = dict(shape=event['shape'],dtype=event['dtype'],operator=event['operator'],
                             scope=event.get('scope',[]),block=event.get('block'))
                if alias not in item['aliases']:
                    item['aliases'].append(alias)
            continue
        action, address = event['action'], event.get('addr')
        if marker_end is not None and stamp>marker_end:
            window=next((v for v in payload.get('_verified_fc1_windows',[]) if v['before']<=stamp<=v['after']),None)
            if window:phase='verified_FC1_checkpoint_window';scope=['FC1'];block_index=window['block']
            else:phase='marker_limit_reached';scope=['marker_limit_reached'];block_index=None
        if action == 'alloc':
            item = add(address,event['size'],stamp,event.get('stack'))
            item['birth_phase'] = phase
            item['birth_scope'],item['block']=list(scope),block_index
            live[address] = item
        elif action == 'free_requested':
            item = live.pop(address,None)
            if item is None: unknown_frees += 1
            else:
                item.update(host_release_ns=stamp,free_phase=phase,release_stack=event.get('stack'),release_scope=list(scope))
                pending[address] = item
        elif action == 'free_completed':
            item = pending.pop(address,None)
            if item is None: unknown_frees += 1
            else: item['cuda_reusable_ns'] = stamp
        elif action == 'segment_alloc':
            segments[address]=dict(address=address,bytes=event['size'],stream=event.get('stream'))
        elif action == 'segment_free':
            segments.pop(address,None)
        elif action == 'oom':
            gaps=[]
            for segment in segments.values():
                start,stop=segment['address'],segment['address']+segment['bytes']
                occupied=sorted((v['address'],v['address']+v['bytes']) for v in [*live.values(),*pending.values()]
                                if start<=v['address']<stop)
                cursor=start
                for left,right in occupied:
                    if left>cursor:gaps.append(left-cursor)
                    cursor=max(cursor,right)
                if cursor<stop:gaps.append(stop-cursor)
            ooms.append(dict(time_ns=stamp,phase=phase,requested_bytes=event['size'],
                scope=list(scope),block=block_index,
                reported_device_free_bytes=event.get('device_free'),
                allocated_bytes=sum(v['bytes'] for v in live.values()),
                pending_bytes=sum(v['bytes'] for v in pending.values()),
                inactive_interval_bytes=sum(gaps),largest_inactive_interval_bytes=max(gaps,default=0),
                live_ids=[v['id'] for v in live.values()],
                fragmentation_scope='Allocator address-interval reconstruction, not a CUDA physical-contiguity test; stream eligibility and allocator rounding can further restrict reuse'))
        allocated = sum(x['bytes'] for x in live.values())
        stack=payload['stacks'][event.get('stack',0)] if payload['stacks'] else []
        column_stack=any((f[0].endswith('quantization.py') and f[1]==1075 and f[2]=='int8_linear')
            or (f[0].endswith('dit_column_linear.py') and f[2]=='_eager_column') for f in stack)
        if action=='alloc' and column_stack:
            projection={51220*14336*2:'FC1',51220*10752*2:'QKV'}.get(event['size'])
            if projection:
                cats.append(dict(projection=projection,derived_layer=cat_counts[projection],
                    time_ns=stamp,allocated_bytes=allocated,active_bytes=allocated+sum(x['bytes'] for x in pending.values()),
                    output_address=address,output_bytes=event['size'],live_ids=[x['id'] for x in live.values()]))
                cat_counts[projection]+=1
        points.append(dict(time_ns=stamp,allocated_bytes=allocated,
                           pending_bytes=sum(x['bytes'] for x in pending.values()),phase=phase,scope=list(scope),block=block_index))
        phase_key=(block_index,tuple(scope))
        if allocated>phase_peaks.get(phase_key,{}).get('allocated_bytes',-1):
            phase_peaks[phase_key]=dict(time_ns=stamp,allocated_bytes=allocated,phase=phase,
                active_bytes=allocated+sum(x['bytes'] for x in pending.values()))
        if allocated > peak['allocated_bytes']:
            peak = dict(time_ns=stamp,allocated_bytes=allocated,phase=phase,
                        scope=list(scope),block=block_index,
                        active_bytes=allocated+sum(x['bytes'] for x in pending.values()),
                        live_ids=[i['id'] for i in live.values()])
    def at_event(item,stamp):
        history=[h for h in item['category_history'] if h['time_ns']<=stamp]
        category=history[-1]['category'] if history else ('registered_later' if item['category_history'] else item['category'])
        return dict(item,category_at_event=category)
    peak_items = [at_event(epochs[i],peak.get('time_ns',0)) for i in peak.pop('live_ids')]
    categories = Counter()
    for item in peak_items: categories[item['category_at_event']] += item['bytes']
    peak['category_bytes'] = dict(categories)
    peak['largest_live'] = sorted(peak_items,key=lambda v:v['bytes'],reverse=True)[:24]
    for item in epochs:
        index = item['stack']
        item['source_stack'] = payload['stacks'][index] if index is not None else []
        index=item.get('release_stack')
        item['release_source_stack']=payload['stacks'][index] if index is not None else []
    # Enrich the event-specific copies after all source/lifetime fields exist.
    peak['largest_live']=[dict(epochs[i['id']],category_at_event=i['category_at_event']) for i in peak['largest_live']]
    for oom in ooms:
        items=[at_event(epochs[i],oom['time_ns']) for i in oom.pop('live_ids')]
        by_category=Counter()
        for item in items:by_category[item['category_at_event']]+=item['bytes']
        oom['category_bytes']=dict(by_category)
        oom['largest_live']=sorted(items,key=lambda v:v['bytes'],reverse=True)[:32]
    for cat in cats:
        items=[at_event(epochs[i],cat['time_ns']) for i in cat.pop('live_ids')]
        categories=Counter()
        for item in items:categories[item['category_at_event']]+=item['bytes']
        cat['category_bytes']=dict(categories)
        cat['largest_live']=sorted(items,key=lambda v:v['bytes'],reverse=True)[:32]
    relevant={'alloc','free_requested','free_completed','segment_alloc','segment_free','oom'}
    missing=sum(e['action'] in relevant and 'time_us' not in e for e in payload['events'])
    return dict(peak=peak,ooms=ooms,column_output_allocations=cats,column_output_counts=dict(cat_counts),
                marker_limit_reached=marker_limit,marker_coverage_end_ns=marker_end,
                epochs=epochs,points=points,unknown_free_events=unknown_frees,
                phase_peaks=[dict(block=k[0],scope=list(k[1]),**v) for k,v in phase_peaks.items()],
                missing_timestamps=missing,complete_prefix_history=not (missing or unknown_frees or payload['history_limit_reached']),
                history_limit_reached=payload['history_limit_reached'])


def plot_profile(result, destination):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    points=result['points']
    if not points:return
    origin=points[0]['time_ns'];end=points[-1]['time_ns']
    x=[(p['time_ns']-origin)/1e9 for p in points]
    fig,axes=plt.subplots(2,1,figsize=(14,10),gridspec_kw={'height_ratios':[1,2]},layout='constrained')
    axes[0].plot(x,[p['allocated_bytes']/2**30 for p in points],label='Allocated',color='#2563eb')
    axes[0].plot(x,[(p['allocated_bytes']+p['pending_bytes'])/2**30 for p in points],
                 label='Allocated + pending free',color='#f59e0b',linestyle='--')
    axes[0].set(ylabel='GiB',title=destination.stem+' — allocator events')
    axes[0].legend(loc='upper left');axes[0].grid(alpha=.2)
    items=sorted(result['epochs'],key=lambda e:e['bytes'],reverse=True)[:20]
    labels=[]
    for i,item in enumerate(items):
        birth=item['birth_ns'] if item['birth_ns'] is not None else origin
        release=item['host_release_ns'] if item['host_release_ns'] is not None else end
        complete=item['cuda_reusable_ns'] if item['cuda_reusable_ns'] is not None else end
        left=max(0.,(birth-origin)/1e9);width=max(.000001,(release-birth)/1e9)
        is_weight='weights' in item['category'] or item['category']=='model_auxiliary'
        axes[1].barh(i,width,left=left,color='#16a34a' if is_weight else '#2563eb',height=.65)
        if complete>release:
            axes[1].barh(i,(complete-release)/1e9,left=(release-origin)/1e9,color='#f59e0b',height=.65)
        alias=item['aliases'][0] if item['aliases'] else None
        shape=str(alias['shape'])+' '+alias['dtype'] if alias else item.get('weight_key',item['category'])
        frame=next((f for f in item['source_stack'] if '/patches/' in f[0].replace('\\','/') or 'comfy_kitchen' in f[0]),None)
        source=(' '+Path(frame[0]).name+':'+str(frame[1])) if frame else ''
        labels.append(f"{item['bytes']/2**30:.3f} GiB  {shape}{source}")
    axes[1].set_yticks(range(len(labels)),labels,fontsize=8)
    axes[1].invert_yaxis();axes[1].set(xlabel='Seconds within history chunk',title='Largest storage lifetimes (aliases counted once)')
    axes[1].grid(axis='x',alpha=.2)
    fig.text(.01,.002,'Blue: live activation/temporary. Green: registered model weights. Orange: host-released, pending allocator reuse. Initial/end-censored lifetimes stop at this chunk boundary.',fontsize=8)
    fig.savefig(destination.with_suffix('.png'),dpi=150)
    fig.savefig(destination.with_suffix('.svg'))
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--final-only',action='store_true',help='Analyze final command histories; leave raw prefix snapshots preserved')
    args = parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    summary = []
    for path in sorted(args.directory.glob('rank*/*.json.gz')):
        if args.final_only and path.name.startswith('checkpoint_'):continue
        with gzip.open(path,'rt') as stream: payload=json.load(stream)
        if payload['command'] not in ('refine','denoise'): continue
        if args.final_only:payload=add_checkpoint_boundaries(payload,path)
        result = lifetimes(payload)
        name = path.parent.name+'_'+path.name.removesuffix('.json.gz')
        with gzip.open(args.output/(name+'_lifetimes.json.gz'),'wt') as stream:
            json.dump(result,stream)
        fields=['id','address','bytes','initial','birth_ns','host_release_ns','cuda_reusable_ns',
                'category','block','birth_phase','birth_scope','free_phase','release_scope','weight_key','aliases','source_stack','release_source_stack']
        with (args.output/(name+'_lifetimes.csv')).open('w',newline='',encoding='utf-8') as stream:
            writer=csv.DictWriter(stream,fieldnames=fields,extrasaction='ignore');writer.writeheader()
            for item in sorted(result['epochs'],key=lambda v:v['bytes'],reverse=True):
                writer.writerow({**item,'aliases':json.dumps(item['aliases']),
                                 'source_stack':json.dumps(item['source_stack']),
                                 'release_source_stack':json.dumps(item['release_source_stack'])})
        summary.append(dict(file=path.relative_to(args.directory).as_posix(),rank=payload['rank'],
            command=payload['command'],nfe=payload['nfe'],peak=result['peak'],ooms=result['ooms'],
            checkpoint=payload.get('checkpoint',False),phase_peaks=result['phase_peaks'],
            marker_limit_reached=result['marker_limit_reached'],marker_coverage_end_ns=result['marker_coverage_end_ns'],
            verified_checkpoint_markers=payload.get('_verified_checkpoint_markers',0),
            column_output_counts=result['column_output_counts'],column_output_allocations=result['column_output_allocations'],
            missing_timestamps=result['missing_timestamps'],complete_prefix_history=result['complete_prefix_history'],
            unknown_free_events=result['unknown_free_events'],
            history_limit_reached=result['history_limit_reached'],
            observed_aliases=sum(len(v['aliases']) for v in result['epochs'])))
        if payload['command']=='denoise':plot_profile(result,args.output/name)
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2))
    print('DIT_LIFETIMES_ANALYZED',len(summary),'history chunks')


if __name__=='__main__': main()
