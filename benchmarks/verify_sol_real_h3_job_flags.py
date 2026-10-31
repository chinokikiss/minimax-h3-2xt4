"""Read-only local check of a frozen Real-H3 source-locked Kaggle job."""
from __future__ import annotations
import argparse
import ast
import base64
import hashlib
import json
import re
from pathlib import Path

def inspect(job: Path):
    notebook = job / "experiment.ipynb"
    data = notebook.read_bytes()
    source = ''.join(json.loads(data.decode('utf-8'))['cells'][0]['source'])
    encoded = re.findall(r"worker\.write_bytes\(base64\.b64decode\(([^)]*)\)\)", source)
    if not encoded:
        raise AssertionError("No bounded worker payload found")
    worker = base64.b64decode(ast.literal_eval(encoded[-1])).decode('utf-8')
    binding = json.loads((job / 'source_binding.json').read_text(encoding='utf-8'))
    options = binding['sol_compiler_options']
    for name, enabled in (
        ('H3_SOL_WORKSPACE_POOL', options['workspace_pool']),
    ):
        expected = f"env[{name!r}]='1' if {bool(enabled)} else '0'"
        if expected not in worker:
            raise AssertionError(f"Missing worker flag: {expected!r}")
    if options.get('bitmap_ab'):
        if "env['H3_REAL_SOL_BITMAP_AB']='1' if True else '0'" not in worker:
            raise AssertionError("Real-H3 bitmap A/B flag missing from bounded worker")
        if "env['H3_SOL_LUT_BITMAP']='0'" not in worker:
            raise AssertionError("Real-H3 bitmap A/B must start from baseline LUT")
    if any(name in worker for name in
           ('OPT_WORKSPACE_POOL','OPT_BITMAP_AB')):
        raise AssertionError("Unsubstituted worker flag placeholders")
    receipt=json.loads((job.parent / 'build_receipt.json').read_text(encoding='utf-8'))
    digest=hashlib.sha256(data).hexdigest()
    assert digest == receipt['notebook_sha256'], (digest, receipt['notebook_sha256'])
    return dict(valid=True, options=options, notebook_sha256=digest,
                notebook_bytes=len(data), bounded_worker_occurrences=len(encoded))

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('job', type=Path)
    print(json.dumps(inspect(parser.parse_args().job),indent=2))
