"""Render retained T4 observations; no model execution or provider requests."""
import argparse
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--analysis',type=Path,required=True)
    p.add_argument('--artifacts',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();r=json.loads(args.analysis.read_text(encoding='utf-8'))
    if not r['physical_two_t4']:raise ValueError('Physical T4 evidence required')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
        'axes.spines.right':False,'figure.facecolor':'white'})
    args.output.mkdir(parents=True,exist_ok=True);old='#8997a6';new='#078b89'
    rows=[next(c for c in r['micro_cases'] if c['label']==f'fp32_rope_{n}') for n in (51220,59134)]
    fig,axes=plt.subplots(1,3,figsize=(12,4.6));positions=[0,1]
    for ax,keys,title,unit in ((axes[0],('baseline_ms','candidate_ms'),'FP32 RMS-RoPE latency','ms'),
        (axes[1],('baseline_peak_allocated','candidate_peak_allocated'),'RMS-RoPE allocation peak','GiB')):
        divisor=1024**3 if unit=='GiB' else 1
        for offset,key,color,label in ((-.19,keys[0],old,'Existing eager'),(.19,keys[1],new,'Fused')):
            values=[c[key]/divisor for c in rows]
            bars=ax.bar([x+offset for x in positions],values,width=.35,color=color,label=label)
            ax.bar_label(bars,fmt='%.2f',fontsize=9,padding=3)
        ax.set_xticks(positions,['51,220','59,134']);ax.set_xlabel('Packed token rows');ax.set_ylabel(unit)
        ax.set_title(title,pad=12);ax.margins(y=.2);ax.legend(frameon=False,fontsize=8)
    ax=axes[2];pairs=r['bake']['pairs']
    for offset,key,color,label in ((-.19,'serial_seconds',old,'Serial'),(.19,'pipeline_seconds',new,'Pipelined')):
        bars=ax.bar([x+offset for x in positions],[v[key] for v in pairs],width=.35,color=color,label=label)
        ax.bar_label(bars,fmt='%.2f',fontsize=9,padding=3)
    ax.set_xticks(positions,['Serial first','Pipeline first']);ax.set_ylabel('seconds');ax.margins(y=.2)
    ax.set_title('Six-layer bake AB/BA',pad=12);ax.legend(frameon=False,fontsize=8)
    fig.suptitle('Physical 2 x Tesla T4 | CUDA 12.8 | Kitchen 0.2.37 | TileLang 0.1.13',fontsize=13,y=.98)
    fig.text(.5,.025,'Operator medians: 2 warmups + 5 samples. Bake includes observers/startup. No full H3 generation.',ha='center',fontsize=9,color='#425064')
    fig.tight_layout(rect=(0,.065,1,.94));fig.savefig(args.output/'operator_bake_results.png',dpi=170);plt.close(fig)

    fig,ax=plt.subplots(figsize=(12,4.7));compute='#078b89';persist='#d69030'
    labels=[];maximum=0
    for group,label in enumerate(('trial0_serial','trial0_pipeline')):
        path=args.artifacts/'operator_batch/bake'/label/'rank0/result.json'
        value=json.loads(path.read_text(encoding='utf-8'));events=value['events']
        total=next(e for e in events if e['phase']=='worker_total');origin=total['start_ns']
        maximum=max(maximum,total['wall_ns']/1e9);base=group*3
        for event in events:
            phase=event['phase']
            if phase in ('merge','independent_verification'):lane=base;color=compute
            elif phase=='persist_layer':lane=base+1;color=persist
            else:continue
            ax.broken_barh([((event['start_ns']-origin)/1e9,event['wall_ns']/1e9)],(lane-.3,.6),facecolors=color)
        mode='Serial' if group==0 else 'Pipelined'
        labels.extend((f'{mode}: merge / verify',
            f'{mode}: persistence '+('(main thread)' if group==0 else '(writer thread)')))
        ax.text(value['seconds']+.3,base+.05,f"{value['seconds']:.2f} s",fontsize=9)
    ax.set_yticks([0,1,3,4],labels);ax.invert_yaxis();ax.set_xlim(0,maximum+4)
    ax.set_xlabel('Seconds since worker entry');ax.set_title('Measured rank-0 host spans: same three layers, serial versus pipelined')
    ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True)
    fig.text(.5,.025,'Host envelopes, not GPU utilization. Gaps include source admission/other work; persistence includes hash and fsync.',ha='center',fontsize=9,color='#425064')
    fig.tight_layout(rect=(0,.07,1,1));fig.savefig(args.output/'bake_host_timeline.png',dpi=170);plt.close(fig)


if __name__=='__main__':main()
