"""Offline analysis of source/version-bound physical T4 combined-default data."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
try:
    from .summarize_t4_native_validation import verify_download,kernel_totals
except ImportError:
    from summarize_t4_native_validation import verify_download,kernel_totals


def summarize(root,submission,cpu):
    binding=verify_download(root);folder=root/'t4_validation'
    read=lambda p:json.loads(p.read_text())
    source=read(submission/'source_binding.json')
    cpu_binding=verify_download(cpu);receipt=cpu/'native_profile/preflight_receipt.json'
    assert hashlib.sha256(receipt.read_bytes()).hexdigest()==source['cpu_receipt_sha256'],'CPU receipt mismatch'
    assert read(receipt)['status']=='cpu_ready'
    if not (folder/'batch_results.json').exists():
        return dict(status='gpu_outputs_unavailable',rows=[],vae={},download_binding=binding,
            source_binding=source,cpu_download_binding=cpu_binding,
            reason='No batch receipt returned; physical GPU execution/completion cannot be verified')
    batch=read(folder/'batch_results.json')
    assert read(folder/'source_hashes.json')==source['files_sha256'],'Submitted/runtime source mismatch'
    hardware=batch['hardware'];devices=hardware['devices']
    assert len(devices)==2 and len({d['uuid'] for d in devices})==2
    assert all(d['capability']==[7,5] and 'T4' in d['name'] for d in devices)
    assert hardware['kitchen']=='0.2.37' and hardware['tilelang']=='0.1.13'
    ranks=[read(folder/f'combined_rows_rank{i}.json') if (folder/f'combined_rows_rank{i}.json').exists() else {} for i in (0,1)]
    key=lambda r:(r['kind'],r['layer'],len(r['strengths']),r['projection'],r['mode'])
    indexed=[{key(r):r for r in rank.get('records',[])} for rank in ranks];rows=[]
    for k in sorted(set(indexed[0])|set(indexed[1])):
        pair=[records.get(k) for records in indexed]
        if None in pair:rows.append(dict(key=k,status='missing_rank'));continue
        a,b=pair;samples=[max(x,y) for x,y in zip(a['timing']['samples_ms'],b['timing']['samples_ms'])]
        assert len(samples)==10 and len(a['timing']['samples_ms'])==len(b['timing']['samples_ms'])
        rows.append(dict(key=k,status='measured',kind=a['kind'],layer=a['layer'],tokens=a['tokens'],strengths=a['strengths'],
            projection=a['projection'],mode=a['mode'],samples_ms=samples,median_ms=statistics.median(samples),
            minimum_ms=min(samples),maximum_ms=max(samples),
            per_rank_peak_increment_bytes=[max(p['timing']['peak_increment_bytes']) for p in pair],
            per_rank_resident_before_bytes=[p['timing']['resident_before_bytes'] for p in pair],
            per_rank_accuracy=[{name:p[name] for name in ('vs_gather','vs_native_fp32','independent_lora_exact','finite','output') if name in p} for p in pair],
            per_rank_lora_collectives=[p.get('lora_collectives') for p in pair],provenance=a['provenance']))
    for r in rows:
        if r.get('mode')!='partial':continue
        baseline=next((v for v in rows if tuple(v['key'])==(*r['key'][:-1],'gather')),None)
        if baseline is None or baseline.get('status')!='measured':
            r['comparison_status']='missing_gather_control'
            continue
        r['speedup_ratio_of_medians']=baseline['median_ms']/r['median_ms']
        r['paired_speedups']=[a/b for a,b in zip(baseline['samples_ms'],r['samples_ms'])]
        r['median_paired_speedup']=statistics.median(r['paired_speedups'])
    vae=read(folder/'reference_demand_vae.json') if (folder/'reference_demand_vae.json').exists() else {}
    expected={(kind,layer,stack,projection,mode) for kind,projection in (('row','out_proj'),('row','fc2'),('block','whole_block'))
        for layer in (0,49) for stack in (1,2) for mode in ('gather','partial')}
    complete=all(r.get('status')=='completed_not_generation_acceptance' and r.get('rank')==i
        and r.get('hardware',{}).get('uuid')==devices[i]['uuid'] and len(r.get('records',[]))==24 for i,r in enumerate(ranks))
    complete=complete and {tuple(r['key']) for r in rows}==expected and vae.get('status')=='passed'
    complete=complete and all(r.get('status')=='measured' for r in rows)
    complete=complete and all(
        all(p.get('finite') and p.get('independent_lora_exact') for p in r['per_rank_accuracy']) if r.get('kind')=='row'
        else all(p.get('output',{}).get('nonfinite')==0 for p in r['per_rank_accuracy'])
        for r in rows if r.get('status')=='measured')
    return dict(status='passed_component_scope' if complete else 'incomplete_or_failed',hardware=hardware,batch=batch,
        source_binding=source,download_binding=binding,cpu_download_binding=cpu_binding,rows=rows,vae=vae,
        profiler={p.name:kernel_totals(p) for p in folder.glob('*_trace.json')},
        limits=['No full H3 sampler/generation quality qualification','S6814 is a selected retained-input subset; layer49 reuses first-block input as stress',
            'Two-entry stacks use the same learned adapter with two strengths','Both LoRA implementations are resident for identical benchmark allocation baselines',
            'VAE tests use a lightweight parent and GPU0, not the persistent TP2 worker lifecycle','Nsight whole-process statistics include loading/compilation; use named NVTX regions in the raw capture'])


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--cpu',type=Path,required=True)
    parser.add_argument('--submission',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();result=summarize(args.data,args.submission,args.cpu)
    args.output.write_text(json.dumps(result,indent=2));print(json.dumps(dict(status=result['status'],records=len(result['rows']),vae=result['vae'].get('status'))))


if __name__=='__main__':main()
