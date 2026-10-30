import json,os,pathlib,sys,tempfile,unittest
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
 with patch.object(harness,'ThreadPoolExecutor',NoDownloadPool),       patch('huggingface_hub.hf_hub_download',side_effect=AssertionError('No network')),       patch('requests.get',side_effect=AssertionError('No network')),       patch.object(checkpoint_ranges,'assemble',side_effect=AuxBoundary):
  try:harness.prepare_assets(source,pathlib.Path(scratch),lambda *a,**k:None)
  except AuxBoundary:pass
  else:raise AssertionError('Did not reach auxiliary assembly boundary')
assert harness.download_with_retry is adapter.download_with_retry
result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromName('benchmarks.test_te_direct_io'))
assert result.wasSuccessful()
assert not harness.torch.cuda.is_initialized()
print('FROZEN_TE_HARNESS_REHEARSAL '+json.dumps(dict(passed=True,tests=result.testsRun,lazy_download_imports=True,gpu_initialized=False,network=False)))
