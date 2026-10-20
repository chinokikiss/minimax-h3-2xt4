"""Standalone operator tradeoff plot; no generation-equivalence inference."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1];out=Path('D:/attention_eval/tp2_p2_20261018')
data=json.loads((ROOT/'docs/data/TP2_P2_RESULTS.json').read_text())
fig,axes=plt.subplots(2,2,figsize=(12,8),gridspec_kw={'height_ratios':[2,1]},layout='constrained')
methods=['int32','fp16','fp16_fused','int8_gather','int8_gather_fused'];labels=['INT32','FP16','FP16 fused','INT8','INT8 fused'];colors=['#555555','#2886bd','#2886bd','#d47b13','#d47b13']
for col,site in enumerate(['communication/o_l25_s9','communication/fc2_l0_s0']):
    rows={r['name']:r for r in data['comparisons'] if r['kind']=='transport' and r['family']==site and r['run']=='repeat'}
    ax=axes[0,col];bottom=axes[1,col]
    for i,name in enumerate(methods):
        row=rows[name];rank=max(row['both_ranks'],key=lambda r:r['GPU']['median']);x=row['slower_rank_API_median_ms'];y=row['max_relative_RMSE']*100
        err=np.array([[max(0,x-rank['GPU']['p10'])],[max(0,rank['GPU']['p90']-x)]])
        ax.errorbar(x,y,xerr=err,fmt='none',ecolor=colors[i],alpha=.6,capsize=3)
        ax.scatter(x,y,s=75,facecolors='white' if name in ('fp16','int8_gather') else colors[i],edgecolors=colors[i],marker='s' if name=='int32' else 'o',zorder=3)
        offset=(-40,12) if i==0 else (10,12) if i in (1,3) else (10,-23)
        ax.annotate(labels[i],(x,y),xytext=offset,textcoords='offset points',fontsize=9,arrowprops={'arrowstyle':'-','color':colors[i]},color=colors[i])
    ax.set_yscale('symlog',linthresh=.01);ax.set_ylim(-.002,1.6);ax.set_xlabel('Slower-rank complete API median (ms)');ax.set_ylabel('Output relative RMSE (%)');ax.grid(alpha=.2)
    ax.set_title('O projection: layer25 / call9' if col==0 else 'FC2: layer0 / call0')
    values=[rows[n]['max_peak_allocated_bytes']/2**20 for n in methods]
    bars=bottom.bar(np.arange(5),values,color=colors,alpha=.8)
    for i,b in enumerate(bars):
        if i in (1,3):b.set_hatch('//')
        bottom.text(b.get_x()+b.get_width()/2,b.get_height()+8,f'{values[i]:.1f}',ha='center',fontsize=9)
    bottom.set_xticks(np.arange(5),labels,rotation=20);bottom.set_ylabel('Peak allocated MiB');bottom.set_ylim(0,max(values)*1.2);bottom.grid(axis='y',alpha=.2)
fig.suptitle('Real H3 operators on 2 × Tesla T4 — latency / arithmetic error / allocator memory',fontsize=14)
fig.supxlabel('Reverse-order repeat; horizontal bars: p10–p90 of the slower-median rank. PyTorch allocation excludes NCCL internals.\nApproximate transport generation quality remains unqualified.',fontsize=9)
fig.savefig(out/'transport_pareto.png',dpi=180);fig.savefig(out/'transport_pareto.svg')
print('Saved',out/'transport_pareto.png')
