"""Apply reviewed PTD cleanup over a historical capsule; preserve Native/core pins."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import re
import tarfile
from build_kaggle_light_delivery import payload_from_distribution,write_distribution
from ptd_e2e_contract import from_previous
from ptd_delivery_sources import RETIRED,apply as ptd_sources

ROOT=Path(__file__).resolve().parents[1]
FILES=('ptd_e2e_contract.py','run_ptd_e2e.py','run_ptd_e2e_batch.py','test_ptd_e2e.py','qualify_ptd_e2e.py',
       'ref2va_e2e_run.py','e2e_attention_validation.py','test_e2e_attention_validation.py',
       'ptd_storage.py','test_ptd_storage.py')


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--base',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True); args=parser.parse_args()
    raw=args.base.read_bytes(); digest=hashlib.sha256(raw).hexdigest()
    if args.base.name!=digest+'.tar.gz': raise ValueError('Parent archive SHA mismatch')
    with tarfile.open(fileobj=io.BytesIO(raw),mode='r:gz') as archive:
        distribution={m.name:archive.extractfile(m).read() for m in archive}
    frozen={n:hashlib.sha256(v).hexdigest() for n,v in distribution.items()
            if n.startswith('frozen/') or n=='direct_core_requirements.json'}
    original=payload_from_distribution(distribution); files=dict(original)
    prior=json.loads(distribution['delivery_manifest.json'])
    assert {n:hashlib.sha256(v).hexdigest() for n,v in original.items()}==prior['source_hashes']
    for name in FILES: files['benchmarks/'+name]=(ROOT/'benchmarks'/name).read_bytes()
    ptd=ptd_sources(files)
    plan=from_previous(json.loads(original['turbo_plan.json']))
    files['ptd_e2e_plan.json']=json.dumps(plan,indent=2).encode()
    allowed={*ptd,*RETIRED,*('benchmarks/'+n for n in FILES),'ptd_e2e_plan.json'}
    assert all(files[n]==v for n,v in original.items() if n not in allowed), 'Unrelated source must remain byte-exact'
    assert not any(re.search(rb'(?:hf_[A-Za-z0-9]{25,}|gh[pousr]_[A-Za-z0-9]{30,})',v) for v in files.values())
    write_distribution(args.output,distribution,files,parent=digest)
    assert all(hashlib.sha256(distribution[n]).hexdigest()==value for n,value in frozen.items())
    (args.output/'preservation.json').write_text(json.dumps(dict(parent_sha256=digest,
        non_ptd_runtime_files_preserved=True,native_comfyui_preserved=True,core_pins_preserved=True,
        retired_sources_removed=list(RETIRED),ptd_cleanup=True,prefetch_refiner_default=True,
        observer_fix='Classify every RPC, including Temporal decoder',unrelated_dirty_changes_excluded=True,
        no_local_lora_download=True,credentials=False,full_pipeline_authorized=True,
        plan_sha256=hashlib.sha256(files['ptd_e2e_plan.json']).hexdigest(),frozen_sha256=frozen),indent=2))


if __name__=='__main__': main()
