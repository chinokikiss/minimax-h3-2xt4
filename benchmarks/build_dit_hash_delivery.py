"""Reviewed index-only overlay on the frozen VRAM-fixture distribution."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import re
import subprocess
import tarfile

from build_kaggle_light_delivery import payload_from_distribution,write_distribution

ROOT=Path(__file__).resolve().parents[1]
FILES=(
    'patches/tilelang_dit_fusion.py','patches/tp_dit.py','patches/h3_tensor_flow.py',
    'patches/tilelang_transport_p2.py','patches/tilelang_swiglu_quant.py',
    'patches/h3_operator_aot.py','patches/h3_operator_dependencies.py','patches/h3_operator_policy.py',
    'patches/comfy_tp_runtime.py','custom_nodes/minimax_h3_tp2/__init__.py',
    'patches/tp2_bundle.py','patches/tp2_disk_bake.py','patches/ptd/bank.py','patches/ptd/runtime.py',
    'benchmarks/test_h3_optimization_batch.py','benchmarks/test_h3_dit_fusion.py',
    'benchmarks/run_review_cpu.py','benchmarks/dit_hash_observer.py',
    'benchmarks/test_dit_hash_ablation.py','benchmarks/run_dit_hash_ablation.py',
    'benchmarks/ptd_worker_probe.py','benchmarks/run_ptd_tp2_t4.py',
    'benchmarks/build_dit_hash_job.py','benchmarks/build_dit_hash_delivery.py',
)


def main():
    p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    raw=args.base.read_bytes();digest=hashlib.sha256(raw).hexdigest()
    assert args.base.name==digest+'.tar.gz'
    with tarfile.open(fileobj=io.BytesIO(raw),mode='r:gz') as archive:
        distribution={m.name:archive.extractfile(m).read() for m in archive}
    protected={n:hashlib.sha256(v).hexdigest() for n,v in distribution.items()
               if n.startswith('frozen/') or n=='direct_core_requirements.json'}
    original=payload_from_distribution(distribution);files=dict(original)
    assert {n:hashlib.sha256(v).hexdigest() for n,v in original.items()}==json.loads(
        distribution['delivery_manifest.json'])['source_hashes']
    for name in FILES:files[name]=subprocess.check_output(['git','show',':'+name],cwd=ROOT)
    assert all(files[n]==v for n,v in original.items() if n not in FILES)
    assert not any(re.search(rb'(?:hf_[A-Za-z0-9]{25,}|gh[pousr]_[A-Za-z0-9]{30,})',v) for v in files.values())
    write_distribution(args.output,distribution,files,parent=digest)
    assert all(hashlib.sha256(distribution[n]).hexdigest()==v for n,v in protected.items())
    (args.output/'preservation.json').write_text(json.dumps(dict(parent_sha256=digest,
        frozen_archive_and_lock_sha256=protected,native_comfyui_preserved=True,core_pins_preserved=True,
        unrelated_dirty_changes_excluded=True,weights_downloaded_locally=False,
        overlays={n:hashlib.sha256(files[n]).hexdigest() for n in FILES}),indent=2),encoding='utf-8')


if __name__=='__main__':main()
