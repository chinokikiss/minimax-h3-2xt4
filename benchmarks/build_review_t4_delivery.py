"""Bind the review operator batch to frozen source, pins and public base shards."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import re
import tarfile
from build_kaggle_light_delivery import payload_from_distribution,write_distribution

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True)
    p.add_argument('--prior',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():raise FileExistsError('Preserve earlier source bindings')
    raw=args.base.read_bytes();digest=hashlib.sha256(raw).hexdigest()
    if args.base.name!=digest+'.tar.gz':raise ValueError('Immutable parent hash mismatch')
    with tarfile.open(fileobj=io.BytesIO(raw),mode='r:gz') as archive:
        distribution={m.name:archive.extractfile(m).read() for m in archive}
    original=payload_from_distribution(distribution);files=dict(original)
    paths=(list((ROOT/'patches').glob('*.py'))+list((ROOT/'benchmarks').glob('*.py'))+
           list((ROOT/'tools').rglob('*.py'))+
           [ROOT/'custom_nodes/minimax_h3_tp2/__init__.py',
            ROOT/'workflows/video_minimax_h3_r2v_tp2_int8.json',
            ROOT/'workflows/video_minimax_h3_r2v_tp2_operators.json'])
    for path in paths:files[path.relative_to(ROOT).as_posix()]=path.read_bytes()
    prior=args.prior.read_bytes()
    from run_review_t4 import PRIOR_SHA,fixture_plan
    if hashlib.sha256(prior).hexdigest()!=PRIOR_SHA:raise ValueError('Pre-edit baseline hash mismatch')
    files['benchmarks/review_prior_column.py']=prior
    proof=ROOT/'semantic_results/kaggle_hf_publication_v6'
    pub=json.loads((proof/'publication.json').read_text())
    receipts=json.loads((proof/'upload_receipts.json').read_text())['objects']
    plan=dict(repository=pub['repo'],revision=pub['commit'],token_required=False,
        objects={f'{kind}.block00.rank{rank}.safetensors':
            {key:receipts[f'{kind}.block00.rank{rank}.safetensors'][key] for key in ('bytes','sha256')}
            for kind in ('dit','te') for rank in (0,1)})
    fixture_plan(plan);files['benchmarks/review_t4_assets.json']=json.dumps(plan,indent=2).encode()
    # Do not publish user credentials accidentally embedded in any source.
    credential=re.compile(rb'(?:hf_[A-Za-z0-9]{25,}|gh[pousr]_[A-Za-z0-9]{30,}|KGAT_[A-Za-z0-9_\-]{20,})')
    if any(credential.search(v) for v in files.values()):raise RuntimeError('Credential-like source content; publication blocked')
    frozen={n:hashlib.sha256(v).hexdigest() for n,v in distribution.items()
            if n not in ('capsule.tar.gz.bin','delivery_manifest.json')}
    allowed={p.relative_to(ROOT).as_posix() for p in paths}
    if any(files[n]!=v for n,v in original.items() if n not in allowed):raise ValueError('Unexpected source change')
    write_distribution(args.output,distribution,files,parent=digest)
    (args.output/'review_binding.json').write_text(json.dumps(dict(
        parent_distribution_sha256=digest,baseline_sha256=PRIOR_SHA,
        frozen_distribution_unchanged=frozen,assets=plan,asset_bytes=sum(r['bytes'] for r in plan['objects'].values()),
        changed={n:hashlib.sha256(v).hexdigest() for n,v in files.items() if original.get(n)!=v},
        source_only=True,private_activations_included=False),indent=2))


if __name__=='__main__':main()
