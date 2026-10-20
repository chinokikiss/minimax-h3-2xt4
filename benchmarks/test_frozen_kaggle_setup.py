"""Prevent source drift and resolver substitutions before Kaggle installation."""
import hashlib
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import zipfile
from benchmarks.frozen_kaggle_setup import check_resolution,prepare_source,require_cpu_setup,offline_install,manifest_digest


class FrozenSetup(unittest.TestCase):
    def candidate(self,name='torch',version='2.12.1+cu132'):
        return dict(metadata=dict(name=name,version=version),download_info=dict(
            archive_info=dict(hashes=dict(sha256='a'*64))))

    def test_version_substitution_and_unknown_platform_dependency_block(self):
        manifest=dict(packages={'torch':'2.12.1+cu132'})
        check_resolution(dict(install=[self.candidate()]),manifest)
        with self.assertRaisesRegex(RuntimeError,'Version substitution'):
            check_resolution(dict(install=[self.candidate(version='2.11.0+cu130')]),manifest)
        with self.assertRaisesRegex(RuntimeError,'Unqualified platform dependency'):
            check_resolution(dict(install=[self.candidate(name='nvidia-new-runtime')]),manifest)

    def test_unhashed_wheel_blocks(self):
        candidate=self.candidate();candidate['download_info']={}
        with self.assertRaisesRegex(RuntimeError,'Missing wheel SHA256'):
            check_resolution(dict(install=[candidate]),dict(packages={'torch':'2.12.1+cu132'}))

    def test_archive_and_requirements_are_verified_before_source_use(self):
        with tempfile.TemporaryDirectory() as folder:
            bundle=Path(folder)/'bundle';root=Path(folder)/'target';bundle.mkdir();root.mkdir()
            content=b'old API\n';archive=bundle/'comfy_source.tar.gz'
            with tarfile.open(archive,'w:gz') as tar:
                item=tarfile.TarInfo('ComfyUI/attention.py');item.size=len(content)
                tar.addfile(item,io.BytesIO(content))
            locks={}
            for name in ('requirements-local-conda.txt','constraints-local-conda.txt'):
                (bundle/name).write_bytes(b'torch==2.12.1+cu132\n')
                locks[name]=hashlib.sha256((bundle/name).read_bytes()).hexdigest()
            manifest=dict(source_archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
                source_files_sha256={'attention.py':hashlib.sha256(content).hexdigest()},lock_files_sha256=locks)
            (bundle/'local_conda.json').write_text(json.dumps(manifest),encoding='utf-8')
            with self.assertRaisesRegex(RuntimeError,'experiment source binding'):prepare_source(bundle,root,'0'*64)
            prepare_source(bundle,root);self.assertEqual((root/'ComfyUI/attention.py').read_bytes(),content)
            archive.rename(bundle/'comfy_source.tar.gz.bin')
            prepare_source(bundle,root)
            (bundle/'requirements-local-conda.txt').write_bytes(b'torch\n')
            with self.assertRaisesRegex(RuntimeError,'requirements checksum mismatch'):prepare_source(bundle,root)

    def test_online_resolution_refuses_an_allocated_gpu(self):
        with patch('benchmarks.frozen_kaggle_setup.subprocess.run',return_value=SimpleNamespace(returncode=0,stdout='Tesla T4\n')):
            with self.assertRaisesRegex(RuntimeError,'CPU-only host'):require_cpu_setup()

    def test_offline_install_checks_wheel_versions_before_calling_pip(self):
        with tempfile.TemporaryDirectory() as folder:
            bundle=Path(folder);wheelhouse=bundle/'linux-wheelhouse';wheelhouse.mkdir()
            manifest=dict(packages={'torch':'2.12.1+cu132'})
            wheel=wheelhouse/'torch-2.12.1+cu132-py3-none-any.whl'
            def make(version):
                with zipfile.ZipFile(wheel,'w') as archive:
                    archive.writestr('torch.dist-info/METADATA','Name: torch\nVersion: '+version+'\n')
                (wheelhouse/'wheelhouse_receipt.json').write_text(json.dumps(dict(
                    manifest_sha256=manifest_digest(manifest),
                    wheels_sha256={wheel.name:hashlib.sha256(wheel.read_bytes()).hexdigest()})),encoding='utf-8')
            make('2.11.0+cu130')
            with patch('benchmarks.frozen_kaggle_setup.subprocess.run') as run:
                with self.assertRaisesRegex(RuntimeError,'differs from frozen packages'):offline_install(bundle,manifest)
                run.assert_not_called()
                make('2.12.1+cu132');offline_install(bundle,manifest)
                self.assertIn('--no-index',run.call_args.args[0])
                self.assertIn('--no-deps',run.call_args.args[0])
                self.assertIn('--require-hashes',run.call_args.args[0])


if __name__=='__main__':unittest.main(verbosity=2)
