"""Build fresh CPU-only studies, preserving native packages and immutable source."""
import argparse
import ast
import base64
import hashlib
import json
from pathlib import Path
import subprocess

from t4_notebook_watchdog import supervise_worker


def main():
    p=argparse.ArgumentParser();p.add_argument('--kind',choices=('read','upload'),required=True)
    p.add_argument('--commit',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    root=Path(__file__).resolve().parents[1]
    name='kaggle_dataset_read_sweep.py' if a.kind=='read' else 'kaggle_dataset_upload_study.py'
    payload=subprocess.check_output(['git','show',a.commit+':benchmarks/'+name],cwd=root)
    assert payload.replace(b'\r\n',b'\n')==(root/'benchmarks'/name).read_bytes().replace(b'\r\n',b'\n')
    ast.parse(payload)
    sizes=(512,768,1024,384,256,128)
    sources=[f'cdeotte/melanoma-{n}x{n}' for n in sizes] if a.kind=='read' else []
    binding=dict(kind='dataset_'+a.kind,source_commit=a.commit,source_sha256=hashlib.sha256(payload).hexdigest(),
        module=name,cpu_only=True,installs=False,h3_runtime_loaded=False,core_libraries_unchanged=True,
        credential_values_embedded=False,dataset_sources=sources,kernel_sources=[],
        scope='Mounted opaque binary reads' if a.kind=='read' else 'Private synthetic uploads using native clients and ambient/attached-secret auth')
    body='''import base64,hashlib,json,os,pathlib,subprocess,sys
OUT=pathlib.Path('/kaggle/working/dataset_KIND');OUT.mkdir(parents=True,exist_ok=True)
BINDING=BINDING_VALUE
(OUT/'source_binding.json').write_text(json.dumps(BINDING,indent=2))
assert not list(pathlib.Path('/dev').glob('nvidia[0-9]*')),'CPU-only instance required'
data=base64.b64decode(PAYLOAD_VALUE);assert hashlib.sha256(data).hexdigest()==BINDING['source_sha256']
source=OUT/BINDING['module'];source.write_bytes(data)
argv=[sys.executable,'-u',str(source),'--output',str(OUT)]
if BINDING['kind']=='dataset_read':
 roots=[]
 for slug in [value.split('/')[-1] for value in BINDING['dataset_sources']]:
  direct=pathlib.Path('/kaggle/input')/slug
  found=[direct] if direct.is_dir() else [p for p in pathlib.Path('/kaggle/input').rglob(slug) if p.is_dir()]
  if len(found)!=1:raise RuntimeError('Required dataset mount not uniquely present: '+slug)
  roots.append(str(found[0]))
 argv+=['--roots',*roots]
env=dict(os.environ,CUDA_VISIBLE_DEVICES='')
subprocess.run(argv,env=env,check=True,timeout=TIMEOUT_VALUE)
'''.replace('KIND',a.kind).replace('BINDING_VALUE',repr(binding)).replace('PAYLOAD_VALUE',repr(base64.b64encode(payload).decode())).replace('TIMEOUT_VALUE','1080' if a.kind=='read' else '650')
    budget=1150 if a.kind=='read' else 720
    script=supervise_worker(body,budget=budget,output=f'/kaggle/working/dataset_{a.kind}_supervision')
    script+=f"\nassert json.loads(pathlib.Path('/kaggle/working/dataset_{a.kind}_supervision/supervision.json').read_text())['exit_code']==0\n"
    ast.parse(script)
    title=f'H3 Dataset {"Read Sweep" if a.kind=="read" else "Upload Study"} CPU 20261030'
    slug='h3-dataset-'+('read-sweep' if a.kind=='read' else 'upload-study')+'-cpu-20261030'
    metadata=dict(id='pamcks/'+slug,title=title,code_file='experiment.py',language='python',kernel_type='script',
        is_private=True,enable_gpu=False,enable_tpu=False,enable_internet=True,docker_image_pinning_type='original',
        dataset_sources=sources,kernel_sources=[],competition_sources=[],model_sources=[])
    a.output.mkdir(parents=True,exist_ok=False)
    (a.output/'experiment.py').write_text(script,encoding='utf-8')
    (a.output/'kernel-metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    (a.output/'source_binding.json').write_text(json.dumps(binding,indent=2),encoding='utf-8')
    print(json.dumps(dict(kernel=metadata['id'],output=str(a.output),watchdog_seconds=budget)))


if __name__=='__main__':main()
