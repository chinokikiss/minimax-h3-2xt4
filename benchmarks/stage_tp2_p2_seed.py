"""Small lossless FC1 seed + pinned public learned-weight ranges.

O operands reuse already-published real captures. Full local raw fixtures are
retained separately. This avoids reuploading gigabytes of existing data.
"""
import hashlib
import json
import os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.experiments.build_h3_reduction_job import plan

OUT=Path('D:/attention_eval/tp2_p2_20261018');dst=OUT/'seed_dataset';dst.mkdir(exist_ok=True)
source=OUT/'real_transport_dataset/fc1_seed.pt';target=dst/source.name
assert source.exists()
if not target.exists():os.link(source,target)
weights=plan('D:/minimaxH3/MiniMax_H3_Ref2VA_pruned_int8_convrot.safetensors',
    'Comfy-Org/MiniMax-H3','e5eb578a89295337b8ff433a035929ce0279e0b6',
    'diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors',
    lambda k:any(k.startswith(p) for p in ('blocks.25.attn.out_proj.','blocks.0.mlp.fc1.','blocks.0.mlp.fc2.')))
(dst/'weight_plan.json').write_text(json.dumps(dict(transport=weights),indent=2))
raw=json.loads((OUT/'real_transport_dataset/real_transport_manifest.json').read_text())
(dst/'seed_manifest.json').write_text(json.dumps(dict(seed_sha256=hashlib.file_digest(source.open('rb'),'sha256').hexdigest(),raw_fixtures=raw,
    scope='Lossless FC1 replay bitwise verified locally; target hardware must recheck FC2 input SHA before use. O operands are existing native dataset captures.'),indent=2))
(dst/'dataset-metadata.json').write_text(json.dumps(dict(title='H3 TP2 P2 Transport Seed',id='chinokiki9/h3-tp2-p2-transport-seed',licenses=[dict(name='other')]),indent=2))
print('SEED_BYTES',source.stat().st_size)
