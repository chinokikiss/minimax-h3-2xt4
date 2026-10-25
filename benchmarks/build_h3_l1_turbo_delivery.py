"""Bind the current source, public Turbo weights and prior L1 references."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import re
import tarfile

from build_kaggle_light_delivery import payload_from_distribution, write_distribution

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--turbo-identity', type=Path, required=True)
    parser.add_argument('--turbo-header', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Preserve previously prepared/submitted source')
    raw = args.base.read_bytes()
    parent_sha = hashlib.sha256(raw).hexdigest()
    if args.base.name != parent_sha + '.tar.gz':
        raise ValueError('Immutable parent archive checksum mismatch')
    with tarfile.open(fileobj=io.BytesIO(raw), mode='r:gz') as archive:
        distribution = {m.name: archive.extractfile(m).read() for m in archive}
    original = payload_from_distribution(distribution)
    files = dict(original)
    paths = list((ROOT / 'patches').glob('*.py')) + list((ROOT / 'benchmarks').glob('*.py'))
    paths += list((ROOT / 'custom_nodes/minimax_h3_tp2').rglob('*.py'))
    paths += list((ROOT / 'custom_nodes/minimax_h3_tp2/web').rglob('*.js'))
    paths += list((ROOT / 'tools').rglob('*.py'))
    paths += list((ROOT / 'workflows').glob('*tp2*.json'))
    for path in paths:
        files[path.relative_to(ROOT).as_posix()] = path.read_bytes()
    migration = json.loads((ROOT / 'tools/path_migration.json').read_text())
    for old in migration['paths']:
        files.pop(old, None)
    files.pop('patches/te_loading.py', None)
    turbo = json.loads(args.turbo_identity.read_text())
    header = json.loads(args.turbo_header.read_text())
    assets = json.loads(files['bundle_assets.json'])
    if (assets['publication']['repo'] != 'cnmds/minimax-h3-tp2-int8-561a6211'
            or assets['publication']['commit'] != 'd3e6a4b131056d2c626a01acf6ba48123a885783'):
        raise ValueError('Wrong user-selected base TP2 repository/revision')
    if (turbo['repository'] != 'lightx2v/Minimax-h3-Turbo'
            or turbo['revision'] != '30f485ed874aa44c8183348ad361c5eabad318da'
            or turbo['sha256'] != 'b4da466e43588004097dfbcde245ccc439c34a26bc997cb76a0b53a1d549ea0d'):
        raise ValueError('Wrong requested Turbo file identity')
    from turbo_e2e_contract import validate_header
    targets = validate_header(header, assets['auxiliary']['dit']['native_header'])
    turbo.update(targets=len(targets), strength=1.0,
        header_sha256=hashlib.sha256(args.turbo_header.read_bytes()).hexdigest())
    prior = json.loads((ROOT / 'docs/data/h3_l1_lifetime_e2e_20261023.json').read_text())
    plan = prior['workload'] | dict(name='L1_turbo_960x544_disk_baked', width=960, height=544,
        frames=362, fps=24, steps=4, seed=20261020, sampler='lcm', scheduler='simple', cfg=1.0,
        video_shift=12.0, audio_shift=3.0, offline_lora=True, disk_baked_lora=True,
        turbo_lora=turbo, operators=dict(dit_attention='kitchen'),
        source_note='Current promoted defaults; explicitly requested 0.5MP, shifts12/3 and replacement Turbo. Native offline merge from pre-sliced base, persisted to disk before inference.')
    files['turbo_plan.json'] = json.dumps(plan, indent=2).encode()
    files['turbo_lora_header.json'] = args.turbo_header.read_bytes()
    # Freeze the public references, original base plans and core bootstrap.
    for row in prior['reference_images']:
        data = files['bench_assets/l1_current/' + row['file']]
        if len(data) != row['bytes'] or hashlib.sha256(data).hexdigest() != row['sha256']:
            raise ValueError('Reference image changed')
    protected = {k: hashlib.sha256(v).hexdigest() for k, v in distribution.items()
        if k not in ('capsule.tar.gz.bin', 'delivery_manifest.json')}
    credential = re.compile(rb'(?:hf_[A-Za-z0-9]{25,}|gh[pousr]_[A-Za-z0-9]{30,}|KGAT_[A-Za-z0-9_\-]{20,})')
    if any(credential.search(value) for value in files.values()):
        raise RuntimeError('Credential-like source content; publication blocked')
    write_distribution(args.output, distribution, files, parent=parent_sha)
    (args.output / 'turbo_binding.json').write_text(json.dumps(dict(plan=plan,
        frozen_distribution_sha256=protected, base_repository=assets['publication'],
        changed={n:hashlib.sha256(v).hexdigest() for n,v in files.items() if original.get(n)!=v},
        removed=['patches/te_loading.py'], source_only=True, private_captures=False,
        status_polling=False, full_pipeline_authorized=True), indent=2))


if __name__ == '__main__':
    main()
