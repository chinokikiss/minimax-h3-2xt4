"""Fresh frozen-source/stdlib CPU storage job, with one public dataset mount."""
import argparse
import ast
import base64
import hashlib
import json
from pathlib import Path

from benchmarks.t4_notebook_watchdog import supervise_worker


def build(publication_path, output):
    root = Path(__file__).resolve().parents[1]
    publication = json.loads(Path(publication_path).read_text())
    assert publication['published']
    assert publication['commit'] == 'f9d8588556ed271e825a6938b27d01aa39436c10'
    assert publication['sha256'] == '645ef141804c67cf6d012387d87cfeada2760db2c4b93809d6a8ee0b7bd1f2be'
    files = {name:(root/'benchmarks'/name).read_bytes() for name in
             ('kaggle_storage_probe.py','test_kaggle_storage_probe.py')}
    encoded = {name:base64.b64encode(value).decode() for name,value in files.items()}
    hashes = {name:hashlib.sha256(value).hexdigest() for name,value in files.items()}
    worker = '''import base64,hashlib,importlib.util,io,json,os,pathlib,subprocess,sys,tarfile,time,urllib.request
OUT=pathlib.Path('/kaggle/working/storage_probe');OUT.mkdir(parents=True,exist_ok=True)
PUB=PUBLICATION
HASHES=SOURCE_HASHES
FILES=SOURCE_FILES
started=time.monotonic()
preflight=dict(status='started',scope='Frozen-source identity and stdlib-only CPU probe; no H3/core operator execution or ABI qualification',source_publication=PUB,probe_hashes=HASHES,package_installation=False)
def save():(OUT/'preflight.json').write_text(json.dumps(preflight,indent=2))
save()
try:
    assert not list(pathlib.Path('/dev').glob('nvidia[0-9]*')),'CPU-only job required'
    print('STORAGE_PREFLIGHT frozen_source_download',flush=True)
    dist=pathlib.Path('/kaggle/temp/h3_storage_frozen_distribution');dist.mkdir(parents=True,exist_ok=True)
    source=pathlib.Path('/kaggle/temp/h3_storage_frozen_source');source.mkdir(parents=True,exist_ok=True)
    with urllib.request.urlopen(PUB['url'],timeout=75) as response:raw=response.read(PUB['bytes']+1)
    assert len(raw)==PUB['bytes'] and hashlib.sha256(raw).hexdigest()==PUB['sha256']
    with tarfile.open(fileobj=io.BytesIO(raw),mode='r:gz') as archive:archive.extractall(dist,filter='data')
    payload=(dist/'capsule.tar.gz.bin').read_bytes()
    assert hashlib.sha256(payload).hexdigest()==PUB['capsule_sha256']
    with tarfile.open(fileobj=io.BytesIO(payload),mode='r:gz') as archive:archive.extractall(source,filter='data')
    frozen_spec=importlib.util.spec_from_file_location('h3_frozen_setup',dist/'frozen/frozen_kaggle_setup.py')
    frozen=importlib.util.module_from_spec(frozen_spec);frozen_spec.loader.exec_module(frozen)
    frozen_manifest=frozen.prepare_source(dist/'frozen',source)
    preflight['frozen_source_archive_sha256']=frozen_manifest['source_archive_sha256']
    spec=importlib.util.spec_from_file_location('h3_frozen_identity',source/'patches/legacy_runtime.py')
    lock=importlib.util.module_from_spec(spec);spec.loader.exec_module(lock)
    lock.validate_source(source/'ComfyUI')
    pins=json.loads((dist/'direct_core_requirements.json').read_text())['core_pins']
    assert pins['comfy-kitchen']=='0.2.37' and pins['tilelang']=='0.1.13'
    preflight.update(frozen_source='passed',core_pin_manifest='preserved_uninstalled_not_used',python=sys.version)
    probe=pathlib.Path('/kaggle/temp/h3_storage_probe_source');(probe/'benchmarks').mkdir(parents=True,exist_ok=True)
    (probe/'benchmarks/__init__.py').write_text('')
    for name,encoded in FILES.items():
        data=base64.b64decode(encoded);assert hashlib.sha256(data).hexdigest()==HASHES[name]
        (probe/'benchmarks'/name).write_bytes(data)
    env=dict(os.environ,PYTHONPATH=str(probe),CUDA_VISIBLE_DEVICES='')
    print('STORAGE_PREFLIGHT tiny_cpu_contract_tests',flush=True)
    with (OUT/'cpu_tests.log').open('w') as stream:
        tested=subprocess.run([sys.executable,'-m','unittest','benchmarks.test_kaggle_storage_probe','-v'],
                              cwd=probe,env=env,stdout=stream,stderr=subprocess.STDOUT,timeout=30)
    assert tested.returncode==0,'Storage probe contract tests failed'
    preflight.update(status='storage_stdlib_ready',cpu_tests='4 passed',seconds=time.monotonic()-started);save()
    print('STORAGE_PREFLIGHT_READY',flush=True)
    executed=subprocess.run([sys.executable,'-u',str(probe/'benchmarks/kaggle_storage_probe.py'),
                             '--output',str(OUT),'--budget','340'],env=env,timeout=355)
    preflight['benchmark_exit_code']=executed.returncode;save()
    assert executed.returncode==0,'Storage benchmark failed; inspect retained partial result'
except BaseException as error:
    preflight.update(status='failed',error=repr(error),seconds=time.monotonic()-started);save();raise
'''.replace('PUBLICATION',repr(publication)).replace('SOURCE_HASHES',repr(hashes)).replace('SOURCE_FILES',repr(encoded))
    body = supervise_worker(worker, budget=450, output='/kaggle/working/storage_supervision')
    body += "\nassert json.loads((OUT/'supervision.json').read_text())['status']=='worker_exited'\n"
    body += "assert json.loads((OUT/'supervision.json').read_text())['exit_code']==0\n"
    ast.parse(body)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    identity='h3-kaggle-storage-read-probe-2026-10-24'
    metadata=dict(id='pamcks/'+identity,title='H3 Kaggle Storage Read Probe 2026-10-24',
                  code_file='experiment.py',language='python',kernel_type='script',is_private=True,
                  enable_gpu=False,enable_tpu=False,enable_internet=True,docker_image_pinning_type='original',
                  dataset_sources=['cdeotte/melanoma-512x512'],kernel_sources=[],competition_sources=[],model_sources=[])
    (output/'experiment.py').write_text(body,encoding='utf-8')
    (output/'kernel-metadata.json').write_text(json.dumps(metadata,indent=2))
    binding=dict(publication=publication,benchmark_sha256=hashes['kaggle_storage_probe.py'],
                 tests_sha256=hashes['test_kaggle_storage_probe.py'],script_sha256=hashlib.sha256(body.encode()).hexdigest(),
                 dataset_sources=metadata['dataset_sources'],dataset_scope='Public large binary files as filesystem I/O proxy; no H3 weights/activations',
                 cpu_only=True,stdlib_only=True,core_libraries_not_loaded=True,internal_budget_seconds=450,
                 benchmark_budget_seconds=340,source='Fresh benchmark; no historical notebook setup',
                 first_pass_is_not_physically_cold=True)
    (output/'source_binding.json').write_text(json.dumps(binding,indent=2))
    print(json.dumps(dict(job=str(output),kernel=metadata['id'],benchmark_sha256=hashes['kaggle_storage_probe.py']),indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--publication',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();build(args.publication,args.output)
