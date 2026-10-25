import json
import hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];folder=ROOT/'kaggle_h3-quality-first-generation';folder.mkdir(exist_ok=True)
source=(ROOT/'benchmarks/quality_first_t4_generation_bootstrap.py').read_text(encoding='utf8')
names=['benchmarks/capture_real_attention.py','benchmarks/quality_first_generation.py','benchmarks/native_resident_reference.py',
       'benchmarks/quality_first_runtime_probe.py','benchmarks/quality_first_video_metrics.py','benchmarks/attention_root_causes.py',
       'patches/tilelang_sage_qk.py','patches/tilelang_sage_preparation.py']
overlays={name:(ROOT/name).read_text(encoding='utf8') for name in names}
code="import pathlib,subprocess,sys,json\npathlib.Path('/kaggle/working/benchmark_overrides.json').write_text(json.dumps("+repr(overlays)+"))\np=pathlib.Path('/kaggle/working/bootstrap.py');p.write_text("+repr(source)+")\nsubprocess.run([sys.executable,'-u',str(p)],check=True,timeout=21000)\n"
(folder/'local_source_hashes.json').write_text(json.dumps({name:hashlib.sha256(text.encode()).hexdigest() for name,text in overlays.items()}|{'bootstrap':hashlib.sha256(source.encode()).hexdigest()},indent=2))
notebook=dict(nbformat=4,nbformat_minor=5,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}},
    cells=[dict(id='generation-screen',cell_type='code',metadata={},execution_count=None,outputs=[],source=code.splitlines(keepends=True))])
(folder/'experiment.ipynb').write_text(json.dumps(notebook),encoding='utf8')
(folder/'kernel-metadata.json').write_text(json.dumps(dict(id='chinokiki9/h3-quality-first-generation',title='H3 Quality First Generation',
    code_file='experiment.ipynb',language='python',kernel_type='notebook',is_private=True,enable_gpu=True,enable_tpu=False,enable_internet=True,
    dataset_sources=['chinokiki9/h3-quality-first-source-and-reference'],competition_sources=[],kernel_sources=[],model_sources=[]),indent=2))
print(folder)
