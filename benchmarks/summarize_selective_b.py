"""Summarize the isolated Selective B physical study; do not modify its evidence."""
import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import statistics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    data = json.loads((args.directory / 'results.json').read_text(encoding='utf8'))
    if not (data.get('passed') and data.get('physical_dual_t4') and data.get('production_unchanged')):
        raise RuntimeError('Physical completion and protected-source gates must pass')
    groups = defaultdict(list)
    for row in data['timings']:
        groups[row['stack'], row['rows'], row['scope'], row['variant']].append(row)
    comparisons = []
    for (stack, rows, scope, variant), records in sorted(groups.items()):
        ranks = [record['timing']['per_rank'] for record in records]
        samples = {}
        for clock in ('wall_ms', 'cuda_event_ms'):
            values = [max(pair[r][clock][i] for r in (0, 1))
                      for pair in ranks for i in range(len(pair[0][clock]))]
            samples[clock] = dict(median=statistics.median(values), minimum=min(values),
                                  maximum=max(values), samples=len(values),
                                  rounds=[statistics.median(max(pair[r][clock][i] for r in (0, 1))
                                          for i in range(len(pair[0][clock]))) for pair in ranks])
        memory = [{key.removesuffix('_bytes') + '_mib':
                   max(pair[rank][key] for pair in ranks) / 2**20
                   for key in ('temporary_peak_bytes', 'peak_allocated_bytes',
                               'peak_reserved_bytes', 'baseline_allocated_bytes')}
                  for rank in (0, 1)]
        comparisons.append(dict(stack=stack, rows=rows, scope=scope, variant=variant,
                                timing=samples, memory_per_rank=memory))
    for row in comparisons:
        base = next(b for b in comparisons if (b['stack'], b['rows'], b['scope'], b['variant']) ==
                    (row['stack'], row['rows'], row['scope'], 'baseline'))
        row['wall_reduction_percent'] = 100 * (1 - row['timing']['wall_ms']['median'] /
                                               base['timing']['wall_ms']['median'])
    numerical = {}
    for scope in ('fc1', 'ffn', 'b_gemm'):
        metrics = [m for case in data['cases'] for m in case[scope + '_per_rank']]
        numerical[scope] = dict(rank_comparisons=len(metrics),
                                bytes_exact=sum(m['bytes_exact'] for m in metrics),
                                max_abs=max(m['max_abs'] for m in metrics),
                                max_rel_l2=max(m['rel_l2'] for m in metrics),
                                min_cosine=min(m['cosine'] for m in metrics))
    quant = [m for case in data['cases'] for m in case['downstream_quant_per_rank']]
    profiles = []
    for path in sorted(args.directory.glob('SelectiveB_*.json.gz')):
        with gzip.open(path, 'rt', encoding='utf8') as stream:
            events = json.load(stream)['traceEvents']
        kernels = defaultdict(list)
        for event in events:
            if event.get('cat') == 'kernel':
                kernels[event['name']].append(event.get('dur', 0))
        mm = [dict(name=e['name'], input_dims=e.get('args', {}).get('Input Dims'),
                   cpu_us=e.get('dur'), external_id=e.get('args', {}).get('External id'))
              for e in events if e.get('name') in ('aten::mm', 'aten::addmm')]
        for gemm in mm:
            gemm['cuda_kernels'] = [dict(name=e['name'], start_us=e.get('ts'),
                                         duration_us=e.get('dur'), grid=e.get('args', {}).get('grid'))
                                    for e in events if e.get('cat') == 'kernel' and
                                    e.get('args', {}).get('External id') == gemm['external_id']]
        copies = Counter(e['name'] for e in events if e.get('name') in
                         ('aten::cat', 'aten::contiguous', 'aten::clone', 'aten::copy_'))
        profiles.append(dict(file=path.name, gemms=mm, copy_ops=dict(copies),
                             kernels=[dict(name=k, count=len(v), activity_us=sum(v))
                                      for k, v in sorted(kernels.items(), key=lambda p: -sum(p[1]))]))
    root = Path(__file__).resolve().parents[1]
    source_matches = {name: hashlib.sha256((root / name).read_text(encoding='utf8').encode()).hexdigest() == sha
                      for name, sha in data['source_sha256'].items()}
    summary = dict(physical_dual_t4=True, cases=len(data['cases']), numerical=numerical,
                   payload_changed=sum(m['changed_payload'] for m in quant),
                   scale_rows_changed=sum(m['changed_scale_rows'] for m in quant),
                   replicas_agree=all(c['baseline_replicas_agree'] and c['selective_replicas_agree']
                                      for c in data['cases']),
                   source_matches=source_matches, comparisons=comparisons,
                   stage_memory=data['stage_memory'], packing=data['packing'], profiles=profiles)
    (args.directory.parent / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf8')
    print('NUMERICAL', json.dumps(numerical), 'PAYLOAD_CHANGED', summary['payload_changed'],
          'SCALE_ROWS_CHANGED', summary['scale_rows_changed'], 'SOURCE_MATCH', all(source_matches.values()))
    for row in comparisons:
        print(row['stack'], row['rows'], row['scope'], row['variant'],
              'wall_ms', round(row['timing']['wall_ms']['median'], 4),
              'cuda_ms', round(row['timing']['cuda_event_ms']['median'], 4),
              'temporary_mib', [m['temporary_peak_mib'] for m in row['memory_per_rank']])


if __name__ == '__main__':
    main()
