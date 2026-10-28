"""Extend one qualified archive with benchmark-only files; preserve all math."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import re
import tarfile
from build_kaggle_light_delivery import payload_from_distribution,write_distribution

ROOT=Path(__file__).resolve().parents[1]
ADDITIONS=('vae_decoder_service.py','run_vae_checkpoint_ab.py','profile_tp2_disk_bake.py',
           'probe_bake_cuda.py','tp2_public_dit_assets.py','run_vae_bake_batch.py','test_vae_bake_experiment.py')
ASSETS=('vae_bake_plan.json','bake_expected.json','int8_convrot_header.json','fp16_header.json')


def main():
    p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True)
    p.add_argument('--assets',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    raw=args.base.read_bytes();digest=hashlib.sha256(raw).hexdigest()
    assert args.base.name==digest+'.tar.gz'
    with tarfile.open(fileobj=io.BytesIO(raw),mode='r:gz') as archive:
        distribution={m.name:archive.extractfile(m).read() for m in archive}
    original=payload_from_distribution(distribution);files=dict(original)
    for name in ADDITIONS:files['benchmarks/'+name]=(ROOT/'benchmarks'/name).read_bytes()
    for name in ASSETS:files[name]=(args.assets/name).read_bytes()
    assert all(files[n]==v for n,v in original.items() if n!='benchmarks/vae_decoder_service.py')
    pattern=re.compile(rb'(?:hf_[A-Za-z0-9]{25,}|gh[pousr]_[A-Za-z0-9]{30,}|KGAT_[A-Za-z0-9_\-]{20,})')
    assert not any(pattern.search(v) for v in files.values())
    write_distribution(args.output,distribution,files,parent=digest)
    (args.output/'preservation.json').write_text(json.dumps(dict(
        parent_sha256=digest,all_production_patch_bytes_preserved=True,
        changed={n:hashlib.sha256(v).hexdigest() for n,v in files.items() if original.get(n)!=v},
        no_private_tensors_or_credentials=True),indent=2))


if __name__=='__main__':main()
