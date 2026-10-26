"""Offline reduction of the TE prefix-invariance / precision operator audit."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import statistics


def aggregate(rows,key):
    return dict(cases=len(rows),byte_equal_cases=sum(r[key]['byte_equal'] for r in rows),
        max_abs=max(r[key]['max_abs'] for r in rows),
        median_relative_l2=statistics.median(r[key]['relative_l2'] for r in rows),
        max_relative_l2=max(r[key]['relative_l2'] for r in rows),
        median_changed_fraction=statistics.median(r[key]['changed_fraction'] for r in rows))


def summarize(batch):
    assert batch['status']=='completed'
    result=dict(status='completed',scope='Synthetic TE operator audit; no trained activation or model-quality acceptance',
        versions=dict(torch=batch['torch'],cuda=batch['cuda'],**batch['packages']),
        ranks=[dict(rank=r['rank'],gpu=r['gpu'],uuid=r['uuid'],seconds=r['seconds'],
                    precision_cases=len(r['precision']),prefix_cases=len(r['prefix']),fixed_packed_cases=len(r['fixed_packed']))
               for r in batch['ranks']],precision={},prefix={})
    precision=[x for r in batch['ranks'] for x in r['precision']]
    prefix=[x for r in batch['ranks'] for x in r['prefix']]
    fixed=[x for r in batch['ranks'] for x in r['fixed_packed']]
    result['fixed_packed']=dict(cases=len(fixed),byte_equal_cases=sum(x['byte_equal'] for x in fixed))
    result['strict_prefix_invariant']=all(x['output']['byte_equal'] for x in prefix)
    result['FP32_input_outputs_on_BF16_lattice']=all(x['matches_storage_lattice'] for x in precision if x['input_dtype']=='torch.float32')
    result['FP32_oracle_max_abs_vs_CPU_FP64']=max(c['max_abs'] for r in batch['ranks'] for c in r['controls'])
    result['FP32_future_control_max_abs']=max(x['FP32_control']['max_abs'] for x in prefix if x['mode']=='fixed_length_future_edit')
    buckets=defaultdict(list)
    for row in precision:buckets[(row['input_dtype'],row['n'],row['amplitude'],row['mask'])].append(row)
    for (dtype,n,amplitude,mask),rows in sorted(buckets.items()):
        result['precision'][f'{dtype}/N{n}/amp{amplitude}/{mask}']=dict(input_dtype=dtype,n=n,amplitude=amplitude,mask=mask,
            versus_FP32=aggregate(rows,'vs_FP32'),row0_versus_FP32=aggregate(rows,'row0_vs_FP32'),
            storage_rounding_only=aggregate(rows,'ideal_storage_rounding_vs_FP32'),
            dequantized_oracle_versus_FP32=aggregate(rows,'dequantized_FP32_vs_FP32'),
            versus_dequantized_oracle=aggregate(rows,'vs_dequantized_FP32'))
    buckets.clear()
    for row in prefix:buckets[(row['input_dtype'],row['prefix'],row['mutation'],row['edit'])].append(row)
    for (dtype,cut,mutation,edit),rows in sorted(buckets.items()):
        result['prefix'][f'{dtype}/prefix{cut}/{mutation}/{edit}']=dict(input_dtype=dtype,prefix=cut,mutation=mutation,edit=edit,
            output=aggregate(rows,'output'),FP32_control=aggregate(rows,'FP32_control'))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('input',type=Path);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();raw=args.input.read_bytes();summary=summarize(json.loads(raw))
    summary['input_sha256']=hashlib.sha256(raw).hexdigest()
    args.output.write_text(json.dumps(summary,indent=2)+'\n',encoding='utf8')
