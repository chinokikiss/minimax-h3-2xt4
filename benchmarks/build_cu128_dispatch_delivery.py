"""Frozen native-cu128 source plus explicit candidate and operator-only probes."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import re
import tarfile
from build_kaggle_light_delivery import payload_from_distribution,write_distribution

ROOT=Path(__file__).resolve().parents[1]
FILES=('patches/kitchen_cu128_candidate.py','benchmarks/test_cu128_candidate.py',
       'benchmarks/case_cu128_dispatch.py','benchmarks/run_cu128_dispatch.py')


def main():
    p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    raw=args.base.read_bytes();digest=hashlib.sha256(raw).hexdigest();assert args.base.name==digest+'.tar.gz'
    with tarfile.open(fileobj=io.BytesIO(raw),mode='r:gz') as archive:
        distribution={m.name:archive.extractfile(m).read() for m in archive}
    original=payload_from_distribution(distribution);files=dict(original)
    for name in FILES:files[name]=(ROOT/name).read_bytes()
    assert all(files[name]==value for name,value in original.items())
    assert not any(re.search(rb'(?:hf_[A-Za-z0-9]{25,}|gh[pousr]_[A-Za-z0-9]{30,})',v) for v in files.values())
    write_distribution(args.output,distribution,files,parent=digest)
    (args.output/'preservation.json').write_text(json.dumps(dict(parent_sha256=digest,
        existing_runtime_bytes_preserved=True,core_pins_preserved=True,torch='Native Kaggle cu128',
        new_files={n:hashlib.sha256(files[n]).hexdigest() for n in FILES},
        production_default_changed=False,no_models_or_private_inputs=True),indent=2))


if __name__=='__main__':main()
