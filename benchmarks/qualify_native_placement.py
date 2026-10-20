"""Fresh Native placement check on the local GPU; no performance claim."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]


def main():
    output = ROOT/'semantic_results/loading_native_placement'
    output.mkdir(parents=True, exist_ok=True)
    for name, extra in [('streamed', []), ('placed', ['--native-resident'])]:
        subprocess.run([sys.executable, '-u', str(ROOT/'benchmarks/pipeline_residency_audit.py'),
            '--strategy', 'native', '--layers', '2', '--steps', '4', '--width', '128',
            '--height', '128', '--qualification', '--output', str(output/name), *extra],
            check=True, env=os.environ.copy())
    checks = {}
    for name in ('conditioning', 'video', 'audio', 'pixels', 'waveform'):
        a = torch.load(output/'streamed/native_reference'/f'{name}.pt', weights_only=True)
        b = torch.load(output/'placed/native_reference'/f'{name}.pt', weights_only=True)
        checks[name] = dict(is_bit_exact=torch.equal(a, b), shape=list(a.shape))
    sources = [*list((ROOT/'patches').glob('*.py')),
        ROOT/'benchmarks/pipeline_residency_audit.py', ROOT/'benchmarks/native_resident_reference.py',
        ROOT/'benchmarks/full_models.py', ROOT/'benchmarks/retest.py', Path(__file__)]
    result = dict(passed=all(c['is_bit_exact'] for c in checks.values()), physical=False,
        scope='Two trained layers, four Native evaluations, logical placement; correctness only',
        checks=checks, source_sha256={p.relative_to(ROOT).as_posix():
            hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
    (output/'result.json').write_text(json.dumps(result, indent=2))
    if not result['passed']: raise AssertionError('Native placement changed outputs')
    print('QUALIFIED fresh Native placement', flush=True)


if __name__ == '__main__': main()
