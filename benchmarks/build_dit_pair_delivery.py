"""Benchmark-only extension of the pinned, CPU-qualified source distribution."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import re
import tarfile
from build_kaggle_light_delivery import payload_from_distribution,write_distribution

ROOT=Path(__file__).resolve().parents[1]
FILES=('dit_pair_contract.py','profile_dit_pair.py','analyze_dit_pair.py','run_dit_pair_batch.py','test_dit_pair.py',
       'gpu_memory_overhead.py','build_kaggle_light_job.py')


def main():
    p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    raw=args.base.read_bytes();digest=hashlib.sha256(raw).hexdigest()
    assert args.base.name==digest+'.tar.gz'
    with tarfile.open(fileobj=io.BytesIO(raw),mode='r:gz') as archive:
        distribution={m.name:archive.extractfile(m).read() for m in archive}
    original=payload_from_distribution(distribution);files=dict(original)
    for name in FILES:files['benchmarks/'+name]=(ROOT/'benchmarks'/name).read_bytes()
    allowed={'benchmarks/'+n for n in FILES}
    assert all(files[n]==v for n,v in original.items() if n not in allowed)
    assert not any(re.search(rb'(?:hf_[A-Za-z0-9]{25,}|gh[pousr]_[A-Za-z0-9]{30,})',v) for v in files.values())
    write_distribution(args.output,distribution,files,parent=digest)
    (args.output/'preservation.json').write_text(json.dumps(dict(parent_sha256=digest,
        all_production_patch_bytes_preserved=True,frozen_source_and_core_pins_preserved=True,
        changed={n:hashlib.sha256(v).hexdigest() for n,v in files.items() if original.get(n)!=v},
        no_private_tensors_or_credentials=True),indent=2))


if __name__=='__main__':main()
