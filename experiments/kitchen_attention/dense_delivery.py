"""Immutable extension of the retained capsule for the authorized TE/default change."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile
import subprocess
from .delivery import ROOT,publish,notebook,submit


UPDATES=['patches/h3_operator_policy.py','patches/h3_operator_dependencies.py','patches/kitchen_te_attention.py',
    'custom_nodes/minimax_h3_tp2/__init__.py','benchmarks/test_h3_operator_integration.py',
    'benchmarks/test_h3_configuration_defaults.py','benchmarks/test_kitchen_te_attention.py',
    'benchmarks/test_video_decode_memory.py',
    'benchmarks/test_h3_default_fused_row.py','benchmarks/turbo_e2e_contract.py',
    'benchmarks/historical_te_loading.py','benchmarks/h3_fp16_fc2_experiment.py',
    'benchmarks/run_h3_native_fp32_flow.py','benchmarks/run_h3_physical_fp16_flow.py',
    'benchmarks/run_t4_fp16_lora_diagnosis.py','benchmarks/run_t4_fc2_fusion.py',
    'custom_nodes/minimax_h3_tp2/web/loader_migration.js',
    'benchmarks/ref2va_e2e_run.py','benchmarks/run_h3_l1_current_e2e.py','benchmarks/run_h3_l1_turbo_e2e.py']


def distribution(base,output):
    from benchmarks.build_kaggle_light_delivery import payload_from_distribution,write_distribution
    raw=base.read_bytes();digest=hashlib.sha256(raw).hexdigest()
    assert digest=='3e9947c02cf3452fccc90297c3018d6d37cf8a6e186b85a0d88759e79b985273'
    with tarfile.open(base) as archive:parts={m.name:archive.extractfile(m).read() for m in archive if m.isfile()}
    prior=payload_from_distribution(parts);manifest=json.loads(parts['delivery_manifest.json'])
    assert {n:hashlib.sha256(v).hexdigest() for n,v in prior.items()}==manifest['source_hashes']
    files=dict(prior)
    paths=[ROOT/n for n in UPDATES]+[ROOT/'experiments/__init__.py']
    # Preserve the CURRENT checked-out runtime, rather than transplanting its
    # loader onto an older snapshot. This never fetches or upgrades upstream.
    paths += list((ROOT/'patches').glob('*.py'))
    paths += [p for p in (ROOT/'experiments/kitchen_attention').rglob('*') if p.is_file() and '__pycache__' not in p.parts]
    allowed={p.relative_to(ROOT).as_posix() for p in paths}
    retired={n for n in prior if n.startswith('patches/') and n.endswith('.py') and not (ROOT/n).is_file()}
    for name in retired:files.pop(name)
    for path in paths:files[path.relative_to(ROOT).as_posix()]=path.read_bytes()
    assert all(files[n]==v for n,v in prior.items() if n not in allowed|retired)
    changes={n:dict(before_sha256=hashlib.sha256(prior[n]).hexdigest() if n in prior else None,
                    after_sha256=hashlib.sha256(files[n]).hexdigest()) for n in allowed if prior.get(n)!=files[n]}
    manifest['authorized_adaptation']=dict(scope='Current local runtime, TE Kitchen causal default and isolated native SM75 dense candidates',changes=changes,
        local_runtime_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        frozen_source_and_core_lock='Byte-preserved from the prior immutable distribution')
    manifest['authorized_adaptation']['retired_runtime_paths']=sorted(retired)
    parts['delivery_manifest.json']=json.dumps(manifest,indent=2).encode()
    write_distribution(output,parts,files,parent=digest)


def publish_delta(evidence):
    from benchmarks.build_kaggle_light_delivery import tar_bytes
    meta=json.loads((evidence/'distribution/artifact.json').read_text())
    with tarfile.open(evidence/'distribution'/meta['file']) as archive:
        parts={name:archive.extractfile(name).read() for name in ('capsule.tar.gz.bin','delivery_manifest.json')}
    raw=tar_bytes(parts);digest=hashlib.sha256(raw).hexdigest();filename=digest+'.tar.gz'
    (evidence/'distribution'/filename).write_bytes(raw)
    base_name='3e9947c02cf3452fccc90297c3018d6d37cf8a6e186b85a0d88759e79b985273.tar.gz'
    base=dict(url='https://raw.githubusercontent.com/chinokikiss/minimax-h3-2xt4/f8fffbfd0e3cff2138e1b2ebd4cae02c248d969f/runtime/kaggle_artifacts/'+base_name,
              sha256=base_name[:-7],bytes=17669172)
    delta=meta|dict(file=filename,bytes=len(raw),sha256=digest,base=base,
        reconstructed_full_distribution_sha256=meta['sha256'],delivery='Hash-bound source delta over immutable frozen base')
    (evidence/'distribution/delta_artifact.json').write_text(json.dumps(delta,indent=2)+'\n')
    publish(evidence,artifact=delta)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--action',choices=('distribution','publish','notebook','submit'),required=True)
    p.add_argument('--evidence',type=Path,required=True);p.add_argument('--base',type=Path)
    p.add_argument('--mode',choices=('cpu','gpu'));p.add_argument('--qualification',type=Path)
    args=p.parse_args();args.evidence.mkdir(parents=True,exist_ok=True)
    if args.action=='distribution':distribution(args.base,args.evidence/'distribution')
    elif args.action=='publish':publish_delta(args.evidence)
    elif args.action=='notebook':notebook(args.evidence,args.mode,args.qualification,suite='dense')
    else:submit(args.evidence,args.mode)
