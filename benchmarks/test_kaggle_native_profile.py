"""Native Torch preservation, active metadata precedence and offline core hashes."""
import hashlib
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile
from benchmarks.prepare_kaggle_native_torch import core_packages,inventory,install_overlay


class NativeProfile(unittest.TestCase):
    def test_native_abi_family_excluded_but_operator_versions_preserved(self):
        packages={'torch':'local','torchvision':'local','torchaudio':'local','numpy':'local',
                  'pip':'local','setuptools':'local','comfyui-frontend-package':'local',
                  'comfy-kitchen':'0.2.37','tilelang':'0.1.13','safetensors':'0.8.0'}
        self.assertEqual(core_packages(packages),{'comfy-kitchen':'0.2.37','tilelang':'0.1.13','safetensors':'0.8.0'})

    def test_duplicate_metadata_uses_active_distribution(self):
        distributions=[SimpleNamespace(metadata={'Name':'numpy'},version='2.1.3'),
                       SimpleNamespace(metadata={'Name':'numpy'},version='1.26.4')]
        with patch('benchmarks.prepare_kaggle_native_torch.md.distributions',return_value=distributions),\
             patch('benchmarks.prepare_kaggle_native_torch.md.version',return_value='2.1.3'):
            self.assertEqual(inventory(),{'numpy':'2.1.3'})

    def test_untrusted_wheel_rejected_before_pip_and_valid_install_is_offline(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);wheels=root/'wheels';wheels.mkdir()
            wheel=wheels/'comfy_kitchen-0.2.37-py3-none-any.whl'
            with zipfile.ZipFile(wheel,'w') as archive:
                archive.writestr('comfy_kitchen.dist-info/METADATA','Name: comfy-kitchen\nVersion: 0.2.37\n')
            expected={'comfy-kitchen':'0.2.37'}
            with patch('benchmarks.prepare_kaggle_native_torch.run') as run:
                with self.assertRaisesRegex(RuntimeError,'checksum'):
                    install_overlay(wheels,root/'overlay',expected,root/'install.log',{wheel.name:'0'*64})
                run.assert_not_called()
                digest=hashlib.sha256(wheel.read_bytes()).hexdigest()
                install_overlay(wheels,root/'overlay',expected,root/'install.log',{wheel.name:digest})
                command=run.call_args.args[0]
                for flag in ('--no-index','--no-deps','--require-hashes','--target'):self.assertIn(flag,command)


if __name__=='__main__':unittest.main()
