"""Lossless local dataset staging, with hashes bound to the native captures."""
import hashlib
import json
import os
from pathlib import Path
import shutil

root = Path(__file__).resolve().parents[1]
source = Path('D:/attention_eval/tokyo_ref2va_20261013')
dest = Path('D:/attention_eval/t4_redo_20261013/dataset')
dest.mkdir(parents=True, exist_ok=True)
manifest = json.loads((source / 'capture_manifest.json').read_text())
for record in manifest['records']:
    original = source / record['file']
    target = dest / original.name
    if target.exists():
        assert os.path.samefile(original, target)
    else:
        os.link(original, target)
    assert target.stat().st_size == record['bytes']
    record['original_file'] = record['file']
    record['file'] = target.name
manifest.update(scope='Native full sequence, all heads; 5 TE layers, 5 DiT layers at calls 0/9/19, 5 Video VAE layers across 30 decode passes.',
                origin_device='RTX 5060 Ti (capture only; benchmark hardware must be T4)',
                requested_workload=json.loads((root / 'bench_assets/tokyo_ref2va.json').read_text()))
(dest / 'capture_manifest.json').write_text(json.dumps(manifest, indent=2))
shutil.copy2(root / 'bench_assets/tokyo_ref2va.json', dest / 'workload.json')
(dest / 'dataset-metadata.json').write_text(json.dumps(dict(
    title='H3 native real attention Tokyo fixed workload',
    id='chinokiki9/h3-native-attention-tokyo-20261013',
    licenses=[{'name': 'other'}]), indent=2))
print(json.dumps({'directory': str(dest), 'records': len(manifest['records']),
                  'bytes': sum(r['bytes'] for r in manifest['records'])}))
