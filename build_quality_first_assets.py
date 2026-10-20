"""Small private source/reference bundle; no models or activation duplication."""
import hashlib
import json
from pathlib import Path
import shutil
import zipfile
ROOT=Path(__file__).resolve().parent
COMFY=Path('C:/Users/Administrator/Documents/project/ComfyUI-master')
OUT=Path('D:/attention_eval/quality_first_20261014/assets');OUT.mkdir(parents=True,exist_ok=True)
files=[]
for folder in ['comfy','comfy_api','comfy_extras','comfy_execution','comfy_config','app','api_server']:
    files.extend(p for p in (COMFY/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix not in ('.pyc','.safetensors','.pt'))
files.extend(COMFY.glob('*.py'))
files.extend(p for p in [COMFY/'LICENSE',COMFY/'requirements.txt'] if p.exists())
manifest={str(p.relative_to(COMFY)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
with zipfile.ZipFile(OUT/'comfy_master.zip','w',zipfile.ZIP_DEFLATED) as archive:
    for p in files:archive.write(p,str(p.relative_to(COMFY)).replace('\\','/'))
workspace=list((ROOT/'patches').glob('*.py'))+[ROOT/'benchmarks'/name for name in
    ['quality_first_generation.py','capture_real_attention.py','retest.py','full_models.py','fixture_inputs.py']]
workspace+=[ROOT/'bench_assets/tokyo_ref2va.json',ROOT/'bench_assets/tokyo_neon.jpg',ROOT/'semantic_results/pipeline_checkpoint_identity.json']
workspace+=[p for p in [ROOT/'bench_assets/REFERENCE_IMAGE.md'] if p.exists()]
with zipfile.ZipFile(OUT/'experiment_source.zip','w',zipfile.ZIP_DEFLATED) as archive:
    for p in workspace:archive.write(p,str(p.relative_to(ROOT)).replace('\\','/'))
(OUT/'source_identity.json').write_text(json.dumps(dict(comfy_master_commit=None,comfy_master_sha256=manifest,
    workspace_sha256={str(p.relative_to(ROOT)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest() for p in workspace},
    bundle_sha256={name:hashlib.sha256((OUT/name).read_bytes()).hexdigest() for name in ['comfy_master.zip','experiment_source.zip']}),indent=2))
(OUT/'dataset-metadata.json').write_text(json.dumps(dict(id='chinokiki9/h3-quality-first-source-and-reference',
    title='H3 Quality First Source and Reference',licenses=[{'name':'other'}]),indent=2))
print(OUT,'total bytes',sum(p.stat().st_size for p in OUT.iterdir()))
