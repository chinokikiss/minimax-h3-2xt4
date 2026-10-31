"""Native TE startup cache ownership/corruption checks; no compiler or GPU."""
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock,patch
from patches import kitchen_te_attention as te


class NativeCache(unittest.TestCase):
    sources={'native/kernel.cu':'retained-source-hash'}

    def library(self,folder):
        folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
        path=folder/('kitchen_sm75.dll' if os.name=='nt' else 'kitchen_sm75.so')
        path.write_bytes(b'cpu-fixture-not-a-loadable-library')
        receipt=dict(exit_code=0,target='sm_75',compiler_version='V12.8.93',sources=self.sources,
                     artifact_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        path.with_name('build_receipt.json').write_text(json.dumps(receipt))
        return path

    def test_source_and_binary_corruption_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder,patch.object(te,'_source_hashes',return_value=self.sources):
            path=self.library(folder);self.assertEqual(te.validate_library(path),path.resolve())
            path.write_bytes(b'changed')
            with self.assertRaisesRegex(RuntimeError,'library hash mismatch'):te.validate_library(path)
            path=self.library(folder)
            with patch.object(te,'_source_hashes',return_value={'native/kernel.cu':'changed-source'}):
                with self.assertRaisesRegex(RuntimeError,'current retained source'):te.validate_library(path)

    def test_cached_cpu_build_is_reused_without_compilation(self):
        with tempfile.TemporaryDirectory() as folder,patch.object(te,'_source_hashes',return_value=self.sources), \
                patch.dict(os.environ,{te.LIBRARY_ENV:'','H3_KITCHEN_ATTENTION_CACHE':folder}), \
                patch('experiments.kitchen_attention.build.build',side_effect=lambda out,**kw:self.library(out)) as build:
            first=te.prepare_native('qualified-nvcc');second=te.prepare_native('qualified-nvcc')
            self.assertEqual(first,second);self.assertEqual(build.call_count,1)
            self.assertEqual(os.environ[te.LIBRARY_ENV],str(first))
            self.assertFalse(first.with_name('build.lock').exists())

    def test_wait_timeout_preserves_another_builders_lock(self):
        with tempfile.TemporaryDirectory() as folder,patch.object(te,'_source_hashes',return_value=self.sources), \
                patch.dict(os.environ,{te.LIBRARY_ENV:'','H3_KITCHEN_ATTENTION_CACHE':folder}):
            fingerprint=hashlib.sha256(json.dumps(te._source_metadata(),sort_keys=True).encode()).hexdigest()[:24]
            cache=Path(folder)/fingerprint;cache.mkdir();lock=cache/'build.lock';lock.write_text('other builder')
            with patch.object(te.time,'monotonic',side_effect=[0,331]):
                with self.assertRaises(TimeoutError):te.prepare_native('qualified-nvcc')
            self.assertEqual(lock.read_text(),'other builder')


if __name__=='__main__':unittest.main()
