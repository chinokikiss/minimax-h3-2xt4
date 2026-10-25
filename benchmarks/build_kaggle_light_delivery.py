"""Publishable source-only distribution; libraries are fetched by exact pip pins."""
import argparse
import ast
import base64
import gzip
import hashlib
import io
import json
from pathlib import Path
import tarfile
from packaging.utils import parse_wheel_filename

ROOT=Path(__file__).resolve().parents[1]


def payload_files(notebook):
    code=''.join(json.loads(notebook.read_text())['cells'][0]['source'])
    pending=[code]
    while pending:
        text=pending.pop()
        for node in ast.walk(ast.parse(text)):
            if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr=='b64decode':
                raw=base64.b64decode(ast.literal_eval(node.args[0]))
                if raw[:2]==b'\x1f\x8b':
                    with tarfile.open(fileobj=io.BytesIO(raw),mode='r:gz') as archive:
                        return {m.name:archive.extractfile(m).read() for m in archive}
                pending.append(raw.decode())
    raise ValueError('Missing source capsule')


def tar_bytes(files):
    stream=io.BytesIO()
    with tarfile.open(fileobj=stream,mode='w') as archive:
        for name,data in sorted(files.items()):
            item=tarfile.TarInfo(name);item.size=len(data);archive.addfile(item,io.BytesIO(data))
    return gzip.compress(stream.getvalue(),mtime=0)


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--frozen',type=Path)
    p.add_argument('--cpu-receipt',type=Path);p.add_argument('--operators',type=Path)
    p.add_argument('--e2e',type=Path);p.add_argument('--asset-plans',type=Path)
    p.add_argument('--base-distribution',type=Path,help='Extend a hash-named source archive, preserving its frozen source/core lock/runtime bytes')
    p.add_argument('--dit-lifetimes',action='store_true',help='Replace only the reviewed lifetime-scheduling source files; preserve frozen source/core pins')
    args=p.parse_args()
    if args.output.exists():raise FileExistsError('Preserve published distributions')
    if args.base_distribution:
        raw=args.base_distribution.read_bytes();parent=hashlib.sha256(raw).hexdigest()
        assert args.base_distribution.name==parent+'.tar.gz','Base archive digest mismatch'
        with tarfile.open(fileobj=io.BytesIO(raw),mode='r:gz') as archive:
            distribution={m.name:archive.extractfile(m).read() for m in archive}
        prior=json.loads(distribution['delivery_manifest.json'])
        assert hashlib.sha256(distribution['capsule.tar.gz.bin']).hexdigest()==prior['capsule_sha256']
        with tarfile.open(fileobj=io.BytesIO(distribution['capsule.tar.gz.bin']),mode='r:gz') as archive:
            files={m.name:archive.extractfile(m).read() for m in archive}
        assert {n:hashlib.sha256(v).hexdigest() for n,v in files.items()}==prior['source_hashes']
        for name in ('prepare_kaggle_direct_pip.py','bounded_column_int8.py',
                     'preview_bounded_column.py','test_bounded_column_int8.py'):
            files['benchmarks/'+name]=(ROOT/'benchmarks'/name).read_bytes()
        allowed={'benchmarks/prepare_kaggle_direct_pip.py'}
        if args.dit_lifetimes:
            lifetime_files=('patches/dit_column_linear.py','patches/tp_dit.py',
                'patches/h3_operator_policy.py','patches/tilelang_row_quant.py',
                'patches/tilelang_transport_p2.py','patches/tilelang_swiglu_quant.py',
                'patches/parallel_int8.py','patches/optimized_tp.py',
                'benchmarks/test_dit_temporaries.py','benchmarks/preview_dit_lifetimes.py')
            for name in lifetime_files:files[name]=(ROOT/name).read_bytes()
            allowed.update(lifetime_files)
        # Original publication and frozen source/core lock are never modified.
        assert all(files[n]==v for n,v in payload_from_distribution(distribution).items()
                   if n not in allowed)
        write_distribution(args.output,distribution,files,parent=parent)
        return
    if not all((args.frozen,args.cpu_receipt,args.operators,args.e2e,args.asset_plans)):
        p.error('Provide --base-distribution or all original builder inputs')
    if args.dit_lifetimes:p.error('--dit-lifetimes requires an immutable base distribution')
    files=payload_files(args.operators/'experiment.ipynb')
    files.update(payload_files(args.e2e/'experiment.ipynb'))
    additions=['prepare_kaggle_direct_pip.py','frozen_kaggle_setup.py','test_direct_pip_delivery.py',
        'test_h3_row_partial_combined.py','test_h3_reference_loading.py','test_h3_loading_stages.py',
        'test_h3_l1_current_e2e.py','test_t4_notebook_watchdog.py','t4_notebook_watchdog.py',
        'launch_kaggle_light.py','bounded_column_int8.py','preview_bounded_column.py',
        'test_bounded_column_int8.py']
    for name in additions:files['benchmarks/'+name]=(ROOT/'benchmarks'/name).read_bytes()
    for name,data in files.items():
        if name.endswith('.py'):ast.parse(data,filename=name)
    receipt=json.loads(args.cpu_receipt.read_text());pins=receipt['core_pins'];by_minor={}
    for minor,wheels in receipt['target_wheelsets'].items():
        hashes={}
        for filename,digest in wheels['wheels_sha256'].items():
            name,version,_,_=parse_wheel_filename(filename);assert str(version)==pins[name]
            hashes.setdefault(name,set()).add(digest)
        assert hashes.keys()==pins.keys()
        by_minor[minor]={n:n+'=='+pins[n]+''.join(' --hash=sha256:'+h for h in sorted(hashes[n])) for n in pins}
    lock=dict(core_pins=pins,requirements=by_minor,approved_cpu_receipt_sha256=hashlib.sha256(args.cpu_receipt.read_bytes()).hexdigest())
    capsule=tar_bytes(files)
    manifest=dict(format='h3-light-delivery-v1',capsule_sha256=hashlib.sha256(capsule).hexdigest(),
        source_hashes={n:hashlib.sha256(v).hexdigest() for n,v in files.items()},no_wheels=True,
        source_preserved='Frozen ComfyUI legacy archive; current exact patch/benchmark bytes')
    distribution={'capsule.tar.gz.bin':capsule,'direct_core_requirements.json':json.dumps(lock,indent=2).encode(),
        'delivery_manifest.json':json.dumps(manifest,indent=2).encode(),
        'component_asset_plans.json':args.asset_plans.read_bytes()}
    for name in ('comfy_source.tar.gz.bin','local_conda.json','requirements-local-conda.txt','constraints-local-conda.txt','frozen_kaggle_setup.py'):
        distribution['frozen/'+name]=(args.frozen/name).read_bytes()
    write_distribution(args.output,distribution,files)


def payload_from_distribution(distribution):
    with tarfile.open(fileobj=io.BytesIO(distribution['capsule.tar.gz.bin']),mode='r:gz') as archive:
        return {m.name:archive.extractfile(m).read() for m in archive}


def write_distribution(output,distribution,files,*,parent=None):
    for name,data in files.items():
        if name.endswith('.py'):ast.parse(data,filename=name)
    capsule=tar_bytes(files)
    manifest=json.loads(distribution['delivery_manifest.json'])
    manifest.update(capsule_sha256=hashlib.sha256(capsule).hexdigest(),
                    source_hashes={n:hashlib.sha256(v).hexdigest() for n,v in files.items()})
    if parent:manifest['parent_distribution_sha256']=parent
    distribution['capsule.tar.gz.bin']=capsule
    distribution['delivery_manifest.json']=json.dumps(manifest,indent=2).encode()
    output.mkdir(parents=True)
    archive=tar_bytes(distribution);digest=hashlib.sha256(archive).hexdigest();path=output/(digest+'.tar.gz')
    path.write_bytes(archive)
    (output/'delivery_manifest.json').write_text(json.dumps(manifest,indent=2))
    (output/'artifact.json').write_text(json.dumps(dict(file=path.name,sha256=digest,bytes=len(archive),
        capsule_sha256=manifest['capsule_sha256'],source_files=len(files),core_pins=78,kaggle_mounts=0),indent=2))
    print(json.dumps(dict(file=str(path),bytes=len(archive),source_files=len(files),capsule_sha256=manifest['capsule_sha256'])))


if __name__=='__main__':main()
