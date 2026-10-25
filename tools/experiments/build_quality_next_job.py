"""Freeze a paired quality screen after physical-T4 full-head correctness gates."""
import hashlib
import json
from pathlib import Path
import statistics

ROOT=Path(__file__).resolve().parents[2]
OUT=Path('D:/attention_eval/quality_constrained_20261015')
FULL=OUT/'native_full_v1/native_full'
PRIOR=Path('D:/attention_eval/quality_first_20261014/generation_v4/generation_screen')
assert json.loads((FULL/'status.json').read_text())['status']=='complete'
paths=[FULL/f'group_process{i}/results.json' for i in range(3)]
data=[json.loads(p.read_text()) for p in paths]
assert all(d['status']=='complete' and len(d['cases'])==12 for d in data)
rows=[r for d in data for r in d['cases']]
assert all(r['status']=='complete' and r['shape']==[1,56,6814,128] for r in rows)
for r in data[0]['cases']:
    if r['candidate'].startswith('fused_group32'):
        m=r['legacy_operand_match']
        assert m['qi_equal'] and m['ki_equal'] and m['qs_max_diff']==m['ks_max_diff']==0
        assert max(r['implementation_error']['relative_l2'])<.003
choices=['fused_group32','fused_group32_legacy_mma']
medians={name:statistics.median(r['timings']['API']['gpu_median_ms'] for r in rows if r['candidate']==name) for name in choices}
selected=min(medians,key=medians.get)
decision={'candidate':selected,'API_median_ms_across_3layers_3processes':medians,
          'source_results_sha256':{p.parent.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
          'scope':'Choose a performance candidate within the exact Group32 contract; generation quality remains unvalidated'}
schedule={'seeds':[20261013,20261014],'group32_variant':'legacy_mma' if selected.endswith('legacy_mma') else 'cached_qk',
          'selection_evidence':decision,'v4_status_sha256':hashlib.sha256((PRIOR/'status.json').read_bytes()).hexdigest(),
          'v4_override_hashes':json.loads((PRIOR/'benchmark_override_hashes.json').read_text()),
          'challenge':{'width':512,'height':512,'prompt':
              '<Picture 1> shows a Tokyo street at night. Preserve the reference architecture, colorful neon signs, storefronts and night lighting. '
              'In one continuous realistic cinematic shot the camera moves briskly forward along the street, with clear parallax between nearby pedestrians and distant buildings. '
              'A cyclist crosses close to the camera from left to right and briefly occludes the storefronts, while a bus approaches in the opposite lane and pedestrians walk naturally along the sidewalk. '
              'After the cyclist passes, reveal the same buildings and signs in consistent positions. Maintain stable objects and geometry, plausible continuous motion, and coherent lighting through the end. No scene cuts. '
              'Audio: steady city traffic, bicycle tire noise and nearby footsteps, without dialogue or music.'}}
names=['benchmarks/quality_next_driver.py','benchmarks/capture_real_attention.py','benchmarks/quality_first_generation.py',
       'benchmarks/native_resident_reference.py','benchmarks/quality_first_runtime_probe.py','benchmarks/quality_first_video_metrics.py',
       'benchmarks/attention_root_causes.py','patches/tilelang_grouped_fused.py','patches/tilelang_grouped_qk.py',
       'patches/tilelang_sage_qk.py','patches/tilelang_sage_preparation.py']
overlays={name:(ROOT/name).read_text(encoding='utf8') for name in names}
source=(ROOT/'benchmarks/quality_first_t4_generation_bootstrap.py').read_text(encoding='utf8')
folder=ROOT/'kaggle_h3-attention-quality-challenge';folder.mkdir(exist_ok=True)
code="import pathlib,subprocess,sys,json\n"
code+="pathlib.Path('/kaggle/working/benchmark_overrides.json').write_text(json.dumps("+repr(overlays)+"))\n"
code+="pathlib.Path('/kaggle/working/generation_schedule.json').write_text(json.dumps("+repr(schedule)+"))\n"
code+="p=pathlib.Path('/kaggle/working/bootstrap.py');p.write_text("+repr(source)+")\n"
code+="subprocess.run([sys.executable,'-u',str(p)],check=True,timeout=21000)\n"
nb=dict(nbformat=4,nbformat_minor=5,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}},
        cells=[dict(id='paired-quality-screen',cell_type='code',metadata={},execution_count=None,outputs=[],source=code.splitlines(keepends=True))])
(folder/'experiment.ipynb').write_text(json.dumps(nb),encoding='utf8')
(folder/'kernel-metadata.json').write_text(json.dumps(dict(id='chinokiki9/h3-attention-quality-challenge',title='H3 Attention Quality Challenge',
    code_file='experiment.ipynb',language='python',kernel_type='notebook',is_private=True,enable_gpu=True,enable_tpu=False,enable_internet=True,
    dataset_sources=['chinokiki9/h3-quality-first-source-and-reference'],competition_sources=[],
    kernel_sources=['chinokiki9/h3-quality-first-generation'],model_sources=[]),indent=2))
(folder/'generation_schedule.json').write_text(json.dumps(schedule,indent=2))
(folder/'local_source_hashes.json').write_text(json.dumps({name:hashlib.sha256(text.encode()).hexdigest() for name,text in overlays.items()}
    |{'bootstrap':hashlib.sha256(source.encode()).hexdigest(),'notebook':hashlib.sha256((folder/'experiment.ipynb').read_bytes()).hexdigest()},indent=2))
(OUT/'group32_generation_selection.json').write_text(json.dumps(decision,indent=2))
print(folder,selected,medians)
