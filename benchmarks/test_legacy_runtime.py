"""Deployment checks reject mixed source and dependencies before loading weights."""
import hashlib
from pathlib import Path
import tempfile
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from patches import legacy_runtime as lock


class RuntimeLock(unittest.TestCase):
    def test_local_checkout_matches_qualified_source(self):
        root=Path(__file__).resolve().parents[1]/'ComfyUI'
        self.assertEqual(lock.validate_source(root),str(root.resolve()))

    def test_text_lock_accepts_crlf_and_rejects_changed_or_missing_files(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'source.py';expected=hashlib.sha256(b'first\nsecond\n').hexdigest()
            with patch.object(lock,'SOURCE_HASHES',{'source.py':expected}):
                path.write_bytes(b'first\r\nsecond\r\n');lock.validate_source(folder)
                path.write_bytes(b'changed\n')
                with self.assertRaisesRegex(RuntimeError,'Source mismatch: source.py'):lock.validate_source(folder)
                path.unlink()
                with self.assertRaisesRegex(RuntimeError,'Source mismatch: source.py'):lock.validate_source(folder)

    def test_dependency_lock_rejects_different_torch_including_the_historical_t4_version(self):
        versions=dict(lock.runtime_manifest()['packages'])
        with patch.object(lock,'version',side_effect=versions.__getitem__):
            self.assertEqual(lock.validate_packages(),versions)
            versions['torch']='2.11.0+cu130'
            with self.assertRaisesRegex(RuntimeError,'torch==2.12.1'):lock.validate_packages()

    def test_cached_module_from_another_checkout_is_rejected(self):
        root=Path(__file__).resolve().parents[1]/'ComfyUI'
        with patch.dict(sys.modules,{'comfy':SimpleNamespace(__path__=[str(root/'comfy')]),
                'comfy.ops':SimpleNamespace(__file__=str(root.parent/'ComfyUI-master/comfy/ops.py'))}):
            with self.assertRaisesRegex(RuntimeError,'Mixed ComfyUI modules'):lock.validate_imported_source()


if __name__=='__main__':unittest.main(verbosity=2)
