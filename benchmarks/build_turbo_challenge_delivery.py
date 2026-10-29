"""Index-owned current runtime over the qualified T4 capsule; frozen core stays exact."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import re
import subprocess
import sys
import tarfile

from build_kaggle_light_delivery import payload_from_distribution,write_distribution
from turbo_challenge_contract import from_previous

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
FILES=('patches/comfy_tp_runtime.py','custom_nodes/minimax_h3_tp2/__init__.py',
    'benchmarks/run_h3_l1_turbo_e2e.py','benchmarks/run_h3_l1_turbo_batch.py',
    'benchmarks/run_review_cpu.py','benchmarks/test_dit_hash_ablation.py',
    'benchmarks/ref2va_e2e_run.py','benchmarks/e2e_attention_validation.py',
    'benchmarks/test_h3_l1_turbo_e2e.py','benchmarks/test_e2e_attention_validation.py',
    'benchmarks/turbo_challenge_contract.py','benchmarks/test_turbo_challenge.py',
    'benchmarks/qualify_turbo_challenge.py','benchmarks/build_turbo_challenge_job.py',
    'benchmarks/build_turbo_challenge_delivery.py')


def main():
    p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    raw=args.base.read_bytes();digest=hashlib.sha256(raw).hexdigest()
    assert args.base.name==digest+'.tar.gz'
    with tarfile.open(fileobj=io.BytesIO(raw),mode='r:gz') as t:
        distribution={m.name:t.extractfile(m).read() for m in t}
    protected={n:hashlib.sha256(v).hexdigest() for n,v in distribution.items()
               if n.startswith('frozen/') or n=='direct_core_requirements.json'}
    original=payload_from_distribution(distribution);files=dict(original)
    assert {n:hashlib.sha256(v).hexdigest() for n,v in original.items()}==json.loads(
        distribution['delivery_manifest.json'])['source_hashes']
    for n in FILES:files[n]=subprocess.check_output(['git','show',':'+n],cwd=ROOT)
    plan=from_previous(json.loads(original['turbo_plan.json']))
    files['turbo_challenge_plan.json']=json.dumps(plan,indent=2).encode()
    assert all(files[n]==v for n,v in original.items() if n not in FILES)
    assert not any(re.search(rb'(?:hf_[A-Za-z0-9]{25,}|gh[pousr]_[A-Za-z0-9]{30,})',v) for v in files.values())
    write_distribution(args.output,distribution,files,parent=digest)
    assert all(hashlib.sha256(distribution[n]).hexdigest()==v for n,v in protected.items())
    (args.output/'plan.json').write_bytes(files['turbo_challenge_plan.json'])
    (args.output/'preservation.json').write_text(json.dumps(dict(parent_sha256=digest,
        frozen_archive_and_lock_sha256=protected,native_comfyui_preserved=True,core_pins_preserved=True,
        unrelated_dirty_changes_excluded=True,no_local_lora_download=True,ptd=False,
        overlays={n:hashlib.sha256(files[n]).hexdigest() for n in FILES},
        plan_sha256=hashlib.sha256(files['turbo_challenge_plan.json']).hexdigest()),indent=2))


if __name__=='__main__':main()
