"""Version-bound experiment analysis; implementation qualification is not quality approval."""
import argparse
import json
from pathlib import Path


def main():
    ap=argparse.ArgumentParser();ap.add_argument('root',type=Path);args=ap.parse_args();root=args.root
    state=json.loads((root/'status.json').read_text());hardware=state.get('hardware',[])
    assert len(hardware)==2 and all(x['capability']==[7,5] and 'T4' in x['name'] for x in hardware)
    out=dict(job_status=state['status'],hardware=hardware,rounds={},failed_candidates=[],
             quality_status='Not evaluated. Approximate numerical modes require later full-generation distribution/semantic assessment.')
    all_records={}
    for kind in ('comm','attn'):
        for run in ('screen','repeat','profiled'):
            folder=root/(kind+'_'+run)
            if not all((folder/f'rank{i}.json').exists() for i in (0,1)):continue
            ranks=[json.loads((folder/f'rank{i}.json').read_text()) for i in (0,1)]
            if not all(x['status']=='complete' for x in ranks):continue
            records=[x['cases'] for x in ranks];all_records[(kind,run)]=records
            for rank,cases in enumerate(records):
                out['failed_candidates'] += [dict(stage=kind+'_'+run,rank=rank,**x) for x in cases if x['status']!='complete']
            if run=='profiled':continue
            summary={}
            for family in sorted({x['family'] for x in records[0]}):
                maps=[{x['name']:x for x in cases if x['family']==family and x['status']=='complete'} for cases in records]
                common=set(maps[0])&set(maps[1]);baseline='int32' if kind=='comm' else 'current'
                values={}
                for name in sorted(common):
                    gpu=max(x[name]['API']['gpu_median_ms'] for x in maps);wall=max(x[name]['API']['wall_median_ms'] for x in maps)
                    item=dict(gpu_ms=gpu,wall_ms=wall,approximate=maps[0][name]['approximate'])
                    if baseline in common:
                        item['GPU_latency_reduction_pct']=100*(1-gpu/max(x[baseline]['API']['gpu_median_ms'] for x in maps))
                        item['wall_latency_reduction_pct']=100*(1-wall/max(x[baseline]['API']['wall_median_ms'] for x in maps))
                    if kind=='comm':
                        assert maps[0][name]['output_sha256']==maps[1][name]['output_sha256'],'Replicated result differs'
                        item.update(local_payload_bytes=maps[0][name]['local_payload_bytes'],arithmetic_diagnostics=[x[name]['arithmetic_diagnostics'] for x in maps])
                    else:
                        assert all(all(x[name]['operand_identity'].values()) for x in maps)
                        if name=='compact':assert all(x[name]['bitwise_equal_to_same_mode'] for x in maps)
                        item['arithmetic_diagnostics']=[x[name].get('arithmetic_diagnostics') for x in maps]
                    values[name]=item
                summary[family]=values
            out['rounds'][kind+'_'+run]=summary
    for kind in ('comm','attn'):
        if (kind,'screen') in all_records and (kind,'repeat') in all_records:
            for rank in (0,1):
                a={(x['family'],x['name']):x for x in all_records[(kind,'screen')][rank] if x['status']=='complete'}
                b={(x['family'],x['name']):x for x in all_records[(kind,'repeat')][rank] if x['status']=='complete'}
                assert all(a[k]['output_sha256']==b[k]['output_sha256'] for k in set(a)&set(b)),'Output changed across fresh processes'
    units=[root/'attn_screen'/f'unit_rank{i}.json' for i in (0,1)]
    out['independent_attention_unit_cases']=sum(len(json.loads(p.read_text())) for p in units if p.exists())
    (root/'validated_summary.json').write_text(json.dumps(out,indent=2))
    print(json.dumps(out,indent=2))


if __name__=='__main__':main()
