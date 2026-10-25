"""Validate the downloaded physical evidence and summarize disjoint GPU events."""
import argparse
from collections import defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import statistics

ROOT=Path(__file__).resolve().parents[2]
LABELS={'native':'Native (one active GPU)','independent':'Independent slabs (invalid)',
        'global_rope':'Global RoPE only (invalid)','shared_kv_global':'KV + global RoPE (invalid)',
        'full_query_control':'Duplicated full-query control','query_block_control':'Query-block control',
        'aligned':'Native-aligned query ownership'}


def trace_summary(path,rank):
    with gzip.open(path,'rt',encoding='utf-8') as file:trace=json.load(file)
    assert len(trace['deviceProperties'])==2
    assert all(d['name']=='Tesla T4' for d in trace['deviceProperties'])
    buckets=defaultdict(float);counts=defaultdict(int);intervals=[];copy_bytes=0
    kernels=[]
    for event in trace['traceEvents']:
        category=event.get('cat')
        if category not in ('kernel','gpu_memcpy','gpu_memset'):continue
        assert event.get('args',{}).get('device',rank)==rank
        name=event['name'];lower=name.lower();duration=event.get('dur',0)
        if category=='gpu_memcpy':
            bucket='device_copy';copy_bytes+=event.get('args',{}).get('bytes',0)
        elif category=='gpu_memset':bucket='memset'
        elif 'nccl' in lower:bucket='nccl'
        elif 'qk_int_sv' in lower:bucket='int8_attention'
        elif 'quant' in lower or 'convrot' in lower:bucket='quantization_named_kernels'
        elif 'gemm' in lower or 'cutlass' in lower:bucket='gemm'
        elif 'rope_kernel' in lower:bucket='rms_rope'
        else:bucket='other_kernels'
        buckets[bucket]+=duration/1000;counts[bucket]+=1
        intervals.append((event['ts'],event['ts']+duration))
        if category=='kernel':kernels.append(dict(name=name,ms=duration/1000))
    merged=[]
    for start,end in sorted(intervals):
        if merged and start<=merged[-1][1]:merged[-1][1]=max(merged[-1][1],end)
        else:merged.append([start,end])
    span=(merged[-1][1]-merged[0][0])/1000 if merged else 0
    busy=sum(end-start for start,end in merged)/1000
    return dict(buckets_ms=dict(buckets),event_counts=dict(counts),
                gpu_activity_span_ms=span,gpu_activity_union_ms=busy,
                gaps_within_activity_span_ms=span-busy,
                copy_bytes_observed_in_this_process=copy_bytes,
                limits=['Named-kernel classification; fused dequantization is part of GEMM.',
                        'GPU event gaps are profiler observations, not SM occupancy.',
                        'Memcpy bytes exclude traffic performed inside NCCL kernels/proxies.'])


def main():
    parser=argparse.ArgumentParser();parser.add_argument('folder',type=Path,
        nargs='?',default=ROOT/'kaggle_vae_kv_profile_output/vae_kv_profile')
    options=parser.parse_args();folder=options.folder.resolve()
    raw=json.loads((folder/'results.json').read_text())
    manifest=json.loads((folder/'manifest.json').read_text())
    assert raw['physical'] and raw['completed'] and manifest['status']=='complete'
    assert raw['torch']=='2.11.0+cu130' and raw['kitchen']=='0.2.37'
    assert len(raw['cases'])==21
    for name,sha in raw['source_sha256'].items():
        source=(ROOT/name).read_bytes()
        # Git on Windows checks out CRLF; the pinned Linux clone uses LF.
        assert sha in (hashlib.sha256(source).hexdigest(),hashlib.sha256(source.replace(b'\r\n',b'\n')).hexdigest()),name
    rows=[]
    for case in raw['cases']:
        base=next(c for c in raw['cases'] if c['workload']==case['workload'] and c['mode']=='native')
        assert case['input_sha256']==base['input_sha256']
        assert len(case['wall_ms_samples'])==len(case['cuda_ms_samples'])==3
        assert case['median_wall_ms']==statistics.median(case['wall_ms_samples'])
        assert case['accepted']==(case['correctness']['finite'] and case['correctness']['bit_exact'])
        stages=[];traces=[]
        for rank in (0,1):
            stem=f"{case['workload']}_{case['mode']}_rank{rank}"
            stages.append(json.loads((folder/(stem+'.stages.json')).read_text()))
            traces.append(trace_summary(folder/(stem+'.json.gz'),rank))
        row=dict(workload=case['workload'],mode=case['mode'],correctness=case['correctness'],
                 accepted=case['accepted'],median_ms=case['median_wall_ms'],
                 min_ms=min(case['wall_ms_samples']),max_ms=max(case['wall_ms_samples']),
                 accepted_speedup=base['median_wall_ms']/case['median_wall_ms'] if case['accepted'] else None,
                 rank_memory=case['rank_memory'],rank_stages=stages,rank_traces=traces)
        rows.append(row)
    evidence=[p for p in folder.iterdir() if p.is_file() and p.name!='analysis.json']
    output=dict(physical_evidence_validated=True,complete_cases=21,trace_count=42,cases=rows,
                source_sha256=raw['source_sha256'],
                downloaded_evidence_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(evidence)},
                scope=raw['boundary'],warmups=1,timing_samples=3,random_latents=True,
                native_rank1='idle during decoding, but benchmark harness holds a weight replica',
                transport='NCCL SHM/direct/direct across PHB; direct GPU P2P disabled; no NVLink',
                limits=['Single allocation, fixed mode order, three repeated-decode samples.',
                        'No sampler-produced latent or public-VAE/full-pipeline end-to-end claim.',
                        'Profiler stages are nested; do not sum parent/child totals.'])
    (folder/'analysis.json').write_text(json.dumps(output,indent=2))
    for row in rows:
        if row['mode']=='aligned':
            print(row['workload'],'speedup',round(row['accepted_speedup'],3),'seconds',round(row['median_ms']/1000,3),
                  'peak_GiB',[round(m['peak_allocated']/2**30,3) for m in row['rank_memory']])
    realistic=next(r for r in rows if r['mode']=='aligned' and r['workload']=='512x512_124frames')
    for rank,trace in enumerate(realistic['rank_traces']):
        print('realistic rank',rank,'kernel buckets ms',trace['buckets_ms'],
              'activity gaps ms',trace['gaps_within_activity_span_ms'])
    print('VALIDATED 21 cases, 42 physical traces, same inputs and qualified source')


if __name__=='__main__':main()
