"""Repair benchmark lifetimes only; retain all submitted production bytes."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import tarfile
from build_kaggle_light_delivery import payload_from_distribution,write_distribution

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    if args.output.exists():raise FileExistsError('Preserve prior submitted archives')
    raw=args.base.read_bytes();digest=hashlib.sha256(raw).hexdigest()
    if args.base.name!=digest+'.tar.gz':raise ValueError('Parent archive digest mismatch')
    with tarfile.open(fileobj=io.BytesIO(raw),mode='r:gz') as archive:
        distribution={m.name:archive.extractfile(m).read() for m in archive}
    original=payload_from_distribution(distribution);files=dict(original)
    allowed=['benchmarks/'+n for n in ('run_review_t4.py','test_review_t4.py',
        'build_kaggle_light_job.py','build_review_t4_recovery_delivery.py')]
    for name in allowed:files[name]=(ROOT/name).read_bytes()
    unchanged={n:hashlib.sha256(v).hexdigest() for n,v in original.items() if n not in allowed}
    if any(files[n]!=original[n] for n in unchanged):raise ValueError('Unexpected production/fixture change')
    frozen={n:hashlib.sha256(v).hexdigest() for n,v in distribution.items()
            if n not in ('capsule.tar.gz.bin','delivery_manifest.json')}
    write_distribution(args.output,distribution,files,parent=digest)
    (args.output/'recovery_binding.json').write_text(json.dumps(dict(
        parent_distribution_sha256=digest,changed={n:hashlib.sha256(files[n]).hexdigest() for n in allowed},
        source_unchanged=unchanged,frozen_distribution_unchanged=frozen,
        production_source_changed=False,weights_inputs_parameters_changed=False,
        timing_scope='Only unfinished M51220 QKV LoRA; completed v1 timings retained',
        trace_scope='Complete first long-shape operator replay; v1 did not start profiling'),indent=2))


if __name__=='__main__':main()
