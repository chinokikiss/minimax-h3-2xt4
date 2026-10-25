"""Extend a prior immutable distribution with current runtime and one-NFE tools."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import tarfile

from build_kaggle_light_delivery import payload_from_distribution,write_distribution

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--base',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--bundle-source',type=Path);args=parser.parse_args()
    if args.output.exists():raise FileExistsError('Preserve prior distributions')
    raw=args.base.read_bytes();digest=hashlib.sha256(raw).hexdigest()
    assert args.base.name==digest+'.tar.gz'
    with tarfile.open(fileobj=io.BytesIO(raw),mode='r:gz') as archive:
        distribution={m.name:archive.extractfile(m).read() for m in archive}
    original=payload_from_distribution(distribution);files=dict(original)
    paths=list((ROOT/'patches').glob('*.py'))+[ROOT/'custom_nodes/minimax_h3_tp2/__init__.py']
    paths += [ROOT/'benchmarks'/name for name in ('__init__.py','dit_performance_observer.py','profile_dit_single.py',
        'launch_dit_profile.py','test_dit_single_profile.py','run_h3_l1_current_batch.py','run_h3_l1_current_e2e.py',
        'launch_kaggle_light.py','prepare_kaggle_direct_pip.py','test_video_decode_memory.py',
        'build_kaggle_light_job.py','prepare_tp2_bundle_assets.py','build_tp2_bundle_assets.py','test_tp2_bundle.py',
        'worker_trace_probe.py','worker_trace_coverage.py','qualify_worker_cuda_trace.py',
        'worker_cpu_probe.py','worker_cuda_checkpoint.py','test_worker_trace.py')]
    for path in paths:files[path.relative_to(ROOT).as_posix()]=path.read_bytes()
    if args.bundle_source:
        for path in args.bundle_source.rglob('*.json'):
            files[path.relative_to(args.bundle_source).as_posix()]=path.read_bytes()
    allowed={p.relative_to(ROOT).as_posix() for p in paths}
    assert all(files[name]==data for name,data in original.items() if name not in allowed)
    unchanged={n:hashlib.sha256(data).hexdigest() for n,data in distribution.items()
        if n not in ('capsule.tar.gz.bin','delivery_manifest.json')}
    write_distribution(args.output,distribution,files,parent=digest)
    (args.output/'changes.json').write_text(json.dumps(dict(
        changed={n:hashlib.sha256(v).hexdigest() for n,v in files.items() if original.get(n)!=v},
        frozen_distribution_unchanged=unchanged),indent=2))


if __name__=='__main__':main()
