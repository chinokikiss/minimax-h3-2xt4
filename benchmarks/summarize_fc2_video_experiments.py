"""Offline, version-bound interpretation of the FP16 FC2 / Video experiment."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
try:
    from .summarize_t4_native_validation import verify_download,kernel_totals
except ImportError:
    from summarize_t4_native_validation import verify_download,kernel_totals


def read(path):return json.loads(path.read_text(encoding='utf-8'))


def timings(a,b):
    if len(a['samples_ms'])!=7 or len(b['samples_ms'])!=7:
        raise ValueError('Seven aligned samples from each rank are required')
    samples=[max(x,y) for x,y in zip(a['samples_ms'],b['samples_ms'])]
    return dict(samples_ms=samples,median_ms=statistics.median(samples),minimum_ms=min(samples),maximum_ms=max(samples),
        per_rank_peak_allocated_bytes=[a['peak_allocated_bytes'],b['peak_allocated_bytes']])


def complete_matrix(ranks,rows,videos):
    expected={(component,layer,mode,transport) for component,layer in (('te',0),('dit',0),('dit',49))
        for mode in ('control','fused_stored','fused_recompute') for transport in ('int32','scaled_fp16')}
    observed={(r['component'],r['layer'],r['preparation'],r['transport']) for r in rows}
    return (len(rows)==18 and observed==expected and len(ranks)==2 and len(videos)==2
        and all(r.get('rank')==i and r.get('status')=='completed_not_generation_acceptance'
            and len(r.get('cases',[]))==18 for i,r in enumerate(ranks))
        and all(r.get('status')=='completed_operator_diagnostic' and r.get('finite')
            and r.get('independent_codec_epilogue_exact') for r in rows)
        and all(v.get('rank')==i and v.get('status')=='completed_localization_not_identity_acceptance'
            and len(v.get('stages',[]))==74 for i,v in enumerate(videos)))


def summarize(root,submission,prior):
    binding=verify_download(root);folder=root/'t4_validation';batch=read(folder/'batch_results.json')
    source=read(folder/'source_hashes.json');submitted=read(submission/'source_binding.json')
    if source!=submitted['files_sha256']:raise RuntimeError('Provider source differs from submitted capsule')
    with prior.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
    if batch['qualification_reference']['receipt_sha256']!=digest:raise RuntimeError('Prior evidence binding mismatch')
    hardware=batch['hardware'];devices=hardware['devices']
    if len(devices)!=2 or len({d['uuid'] for d in devices})!=2 or any(d['capability']!=[7,5] or 'T4' not in d['name'] for d in devices):
        raise RuntimeError('Two distinct physical T4s required')
    if hardware['kitchen']!='0.2.37' or hardware['tilelang']!='0.1.13':raise RuntimeError('Core library version drift')
    ranks=[read(folder/f'fc2_fusion_rank{rank}.json') if (folder/f'fc2_fusion_rank{rank}.json').exists() else {} for rank in (0,1)]
    indexed=[{(c['component'],c['layer'],c['preparation'],c['transport']):c for c in r.get('cases',[])} for r in ranks]
    rows=[]
    for key in sorted(set(indexed[0])|set(indexed[1])):
        pair=[r.get(key) for r in indexed]
        row=dict(component=key[0],layer=key[1],preparation=key[2],transport=key[3])
        if any(r is None or r.get('status')!='completed_operator_diagnostic' for r in pair):
            row.update(status='failed_or_incomplete',records=pair);rows.append(row);continue
        a,b=pair
        row.update(status='completed_operator_diagnostic',tokens=a['tokens'],lora=a['lora'],
            independent_codec_epilogue_exact=all(r['independent_codec_epilogue_exact'] for r in pair),
            finite=all(r['finite'] for r in pair),
            quantized_code_disagreement_fraction=sum(r['quantization']['code_disagreements'] for r in pair)/sum(r['quantization']['elements'] for r in pair),
            row_vs_control=a['row_vs_control'],block_vs_native_fp32=a['block_vs_native_fp32'],block_vs_fp16_control=a['block_vs_fp16_control'],
            preparation=key[2],transport=key[3],timing={name:timings(a[name],b[name]) for name in
                ('row_complete_api','preparation_complete_api','block_complete_api')},
            provenance=a['provenance'])
        rows.append(row)
    for row in rows:
        control=next((c for c in rows if c['component']==row['component'] and c['layer']==row['layer'] and
            c['preparation']=='control' and c['transport']=='int32' and 'timing' in c),None)
        if control and 'timing' in row:row['row_speedup_vs_fp32prep_int32_control']=control['timing']['row_complete_api']['median_ms']/row['timing']['row_complete_api']['median_ms']
    videos=[read(folder/f'video_split_localization_rank{rank}.json') if (folder/f'video_split_localization_rank{rank}.json').exists() else {} for rank in (0,1)]
    complete=complete_matrix(ranks,rows,videos)
    return dict(schema=1,status='passed' if batch['status']=='passed' and complete else 'failed_or_incomplete',
        execution_complete=complete,video_identity_accepted=False,hardware=hardware,batch=batch,download_binding=binding,
        source_binding=submitted,fc2=rows,fc2_rank_status=[r.get('status','missing') for r in ranks],
        expected_fc2_cases_per_rank=18,observed_fc2_cases_per_rank=[len(r.get('cases',[])) for r in ranks],
        video_localization=videos,profiler={p.name:kernel_totals(p) for p in folder.glob('*_trace.json')},
        limits=['No production-default changes or full H3 sampler',
            'TE input is seeded synthetic; DiT S6814 is a selected subset of retained S7895; layer49 reuses first-block input as stress',
            'Fused regular H4 butterfly changes FP32 summation order relative to Native SGEMM; scaled-FP16 SUM is lossy',
            'Operator metrics are not sparse/generation semantic acceptance; profiler sums are not hardware counters'])


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--submission',type=Path,required=True);parser.add_argument('--prior',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    result=summarize(args.data,args.submission,args.prior)
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2))
    print(json.dumps(dict(status=result['status'],fc2_rank_status=result['fc2_rank_status'],
        observed_cases=result['observed_fc2_cases_per_rank'],
        video_first_differences=[v.get('first_differing_stage',v.get('error','missing')) for v in result['video_localization']]),indent=2))


if __name__=='__main__':main()
