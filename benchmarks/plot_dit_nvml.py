"""Historical board telemetry plot; never infer tensor liveness from NVML."""
import argparse
import ast
import csv
import gzip
import hashlib
import json
from pathlib import Path
import re


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    state=json.loads((args.source/'result.json').read_text())
    raw=(args.source/'memory_samples.json.gz').read_bytes()
    a,b=(state['stages']['sampler'][n] for n in ('start_ns','end_ns'))
    rows=[r for r in json.loads(gzip.decompress(raw)) if a<=r['time_ns']<=b]
    log=(args.source.parent/'L1_generation.log').read_text(errors='replace')
    commands=[ast.literal_eval(m.group(1)) for m in re.finditer(r'REF2VA_RANK_COMMAND 0 (\{[^\r\n]+\})',log)]
    start=next(c['start_ns'] for c in commands if c['command']=='denoise')
    fig,axes=plt.subplots(2,1,figsize=(11,7),sharex=True,layout='constrained',gridspec_kw={'height_ratios':[2,1]})
    fig.get_layout_engine().set(rect=(0,.06,1,.94))
    for rank,color in enumerate(('#2563eb','#e76f24')):
        x=[(r['time_ns']-a)/1e9 for r in rows]
        axes[0].plot(x,[r['devices'][rank]['used']/2**30 for r in rows],color=color,label=f'T4 GPU {rank} board memory')
        axes[1].plot(x,[r['devices'][rank]['utilization'] for r in rows],color=color)
    for ax in axes:ax.axvline((start-a)/1e9,color='#57534e',linestyle='--',linewidth=1);ax.grid(alpha=.2)
    axes[0].text((start-a)/1e9+.5,6,'DiT starts',fontsize=9)
    axes[0].set(ylabel='Board memory (GiB)',ylim=(0,15.5),title='Prior real dual-T4 run: failed 51,220-token evaluation')
    axes[0].legend(loc='upper left')
    axes[1].set(xlabel='Seconds since sampler entry',ylabel='GPU utilization (%)',ylim=(0,105))
    fig.text(.02,.012,'NVML sampled about every 0.5 s; includes allocator cache, CUDA/NCCL context and other processes.\nThis plot does not identify tensor allocation/free lifetimes. No sampling step completed.',fontsize=8)
    for ext in ('png','svg'):fig.savefig(args.output/('prior_t4_sampler_nvml.'+ext),dpi=150)
    plt.close(fig)
    with (args.output/'prior_t4_sampler_nvml.csv').open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(['seconds','gpu0_used_bytes','gpu1_used_bytes','gpu0_util','gpu1_util'])
        for r in rows:writer.writerow([(r['time_ns']-a)/1e9,*[r['devices'][i]['used'] for i in (0,1)],*[r['devices'][i]['utilization'] for i in (0,1)]])
    (args.output/'prior_t4_plot_provenance.json').write_text(json.dumps(dict(source_sha256=hashlib.sha256(raw).hexdigest(),samples=len(rows),scope='Historical actual T4 board telemetry only'),indent=2))


if __name__=='__main__':main()
