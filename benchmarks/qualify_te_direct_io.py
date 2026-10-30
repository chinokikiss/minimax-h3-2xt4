"""Rehearse the supplemental TE harness against the exact capsule, without GPU/network."""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile


def qualify(qualified, output, repository):
    output.mkdir(parents=True, exist_ok=False)
    publication = json.loads((qualified/'publication.json').read_text(encoding='utf-8'))
    archive = qualified/'delivery'/publication['file']
    raw = archive.read_bytes()
    assert len(raw) == publication['bytes'] and hashlib.sha256(raw).hexdigest() == publication['sha256']
    with tarfile.open(fileobj=io.BytesIO(raw), mode='r:gz') as outer:
        payload = outer.extractfile('capsule.tar.gz.bin').read()
    assert hashlib.sha256(payload).hexdigest() == publication['capsule_sha256']
    source = output/'source'
    source.mkdir()
    with tarfile.open(fileobj=io.BytesIO(payload), mode='r:gz') as capsule:
        capsule.extractall(source, filter='data')
    names = ('te_direct_io.py','test_te_direct_io.py','run_te_direct_io.py')
    hashes = {}
    for name in names:
        data = (repository/'benchmarks'/name).read_bytes()
        (source/'benchmarks'/name).write_bytes(data)
        hashes[name] = hashlib.sha256(data).hexdigest()
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='-1',
        PYTHONPATH=os.pathsep.join(str(p) for p in [source, source/'ComfyUI', source/'benchmarks']))
    # Exercise lazy prepare_assets imports with the exact frozen runners present.
    # Stop only at auxiliary assembly, without invoking network or model reads.
    program = '''import json,os,pathlib,sys,tempfile,unittest
from unittest.mock import patch
source=pathlib.Path.cwd()
from benchmarks import run_te_direct_io as harness,te_direct_io as adapter
from patches import comfy_tp_runtime as runtime
assert pathlib.Path(runtime.__file__).resolve().is_relative_to(source)
class AuxBoundary(Exception):pass
class NoDownloadPool:
 def __init__(self,*a,**k):pass
 def __enter__(self):return self
 def __exit__(self,*a):pass
 def map(self,*a,**k):return []
import checkpoint_ranges
with tempfile.TemporaryDirectory(dir=source.parent) as scratch:
 with patch.object(harness,'ThreadPoolExecutor',NoDownloadPool), \
      patch('huggingface_hub.hf_hub_download',side_effect=AssertionError('No network')), \
      patch('requests.get',side_effect=AssertionError('No network')), \
      patch.object(checkpoint_ranges,'assemble',side_effect=AuxBoundary):
  try:harness.prepare_assets(source,pathlib.Path(scratch),lambda *a,**k:None)
  except AuxBoundary:pass
  else:raise AssertionError('Did not reach auxiliary assembly boundary')
assert harness.download_with_retry is adapter.download_with_retry
result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromName('benchmarks.test_te_direct_io'))
assert result.wasSuccessful()
assert not harness.torch.cuda.is_initialized()
print('FROZEN_TE_HARNESS_REHEARSAL '+json.dumps(dict(passed=True,tests=result.testsRun,lazy_download_imports=True,gpu_initialized=False,network=False)))
'''
    (output/'rehearsal.py').write_text(program,encoding='utf-8')
    process = subprocess.run([sys.executable,'-c',program],cwd=source,env=env,
                             capture_output=True,text=True,timeout=90)
    (output/'rehearsal.log').write_text(process.stdout+process.stderr,encoding='utf-8')
    receipt = dict(status='passed' if process.returncode == 0 else 'failed',returncode=process.returncode,
        capsule_sha256=publication['capsule_sha256'],supplemental_sources=hashes,
        frozen_runtime_unchanged=True,environment='Local conda CPU rehearsal, not Linux/T4 execution',
        gpu=False,network=False,models_downloaded=False)
    (output/'rehearsal.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    if process.returncode:
        raise RuntimeError('Frozen TE harness rehearsal failed; see '+str(output/'rehearsal.log'))
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--qualified',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    print(json.dumps(qualify(args.qualified.resolve(),args.output.resolve(),Path(__file__).resolve().parents[1])))
