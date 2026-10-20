"""CPU-only summary of paired TP2 operator experiments, keeping rounds separate."""
import argparse
import json
from pathlib import Path


def paired(folder):
    return [json.loads((folder / f'rank{i}.json').read_text()) for i in range(2)]


def timing(rows, key):
    records = [r[key] for r in rows]
    return dict(slower_rank_gpu_median_ms=max(r['gpu_median_ms'] for r in records),
                slower_rank_wall_median_ms=max(r['wall_median_ms'] for r in records),
                ranks=[dict(gpu_median_ms=r['gpu_median_ms'],
                            gpu_min_ms=min(r['gpu_ms']), gpu_max_ms=max(r['gpu_ms']),
                            wall_median_ms=r['wall_median_ms'],
                            repeats=r['repeats']) for r in records])


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('root', type=Path); args = ap.parse_args()
    result = {'scope': 'Physical T4, current TP2 operators. No full pipeline or generation-quality conclusions.',
              'communication': [], 'row_linear': [], 'attention': [], 'failures': []}
    root = args.root
    for folder in sorted(root.glob('comm_*')):
        if not all((folder / f'rank{i}.json').exists() for i in range(2)): continue
        ranks = paired(folder)
        for index, first in enumerate(ranks[0]['cases']):
            rows = [r['cases'][index] for r in ranks]
            assert all(r['name'] == first['name'] for r in rows)
            result['communication'].append(dict(setting=folder.name, name=first['name'], **timing(rows, 'timings')))
    row_hashes = {}
    for folder in sorted(root.glob('row[0-9]')):
        if not all((folder / f'rank{i}.json').exists() for i in range(2)): continue
        ranks = paired(folder)
        for index, first in enumerate(ranks[0]['cases']):
            rows = [r['cases'][index] for r in ranks]; module = first['target']['module']
            hashes = [r['output_sha256'] for r in rows]
            previous = row_hashes.setdefault(module, hashes)
            result['row_linear'].append(dict(setting=folder.name, protocol=ranks[0]['protocol'],
                channels=ranks[0]['channels'], module=module, output_identical_across_settings=hashes == previous,
                **timing(rows, 'API')))
    for round_name in ('attention_screen', 'attention_repeat'):
        folder = root / round_name
        if not all((folder / f'rank{i}.json').exists() for i in range(2)): continue
        ranks = paired(folder)
        maps = [{(r['family'], r['name']): r for r in rank['cases']} for rank in ranks]
        for key in sorted(set(maps[0]) & set(maps[1])):
            rows = [m[key] for m in maps]
            if any(r['status'] != 'complete' for r in rows):
                result['failures'].append(dict(round=round_name, family=key[0], name=key[1],
                    errors=[r.get('error') or r.get('rank_errors') for r in rows])); continue
            result['attention'].append(dict(round=round_name, family=key[0], name=key[1],
                bitwise_output_equal=all(r['bitwise_output_equal'] for r in rows),
                operands_identical=all(all(r['operand_identity'].values()) for r in rows),
                **timing(rows, 'API')))
    for row in result['attention']:
        baseline = next(r for r in result['attention'] if r['round'] == row['round'] and r['family'] == row['family'] and r['name'] == 'frozen')
        row['speedup_vs_same_round_frozen'] = baseline['slower_rank_gpu_median_ms'] / row['slower_rank_gpu_median_ms']
    result['confirmed_candidates'] = []
    for family in sorted({r['family'] for r in result['attention']}):
        rows = [r for r in result['attention'] if r['family'] == family and r['round'] == 'attention_repeat' and r['name'] != 'frozen']
        for row in rows:
            screen = next(r for r in result['attention'] if r['family'] == family and r['round'] == 'attention_screen' and r['name'] == row['name'])
            result['confirmed_candidates'].append(dict(family=family, name=row['name'],
                wins_both_rounds=screen['speedup_vs_same_round_frozen'] > 1 and row['speedup_vs_same_round_frozen'] > 1,
                screen_speedup=screen['speedup_vs_same_round_frozen'], repeat_speedup=row['speedup_vs_same_round_frozen']))
    path = root / 'paired_summary.json'; path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__': main()
