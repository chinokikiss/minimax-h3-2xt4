"""CPU-only regression tests for the PTD 2-NFE nvcc selection/qualification fix."""
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from patches.h3_operator_dependencies import (
    PTD_2NFE_CUDA13_TRIAL, compiler_qualification, configure_compiler)


class CompilerGateTests(unittest.TestCase):
    def test_cuda12_8_stays_validated_without_opt_in(self):
        with patch.dict(os.environ, {PTD_2NFE_CUDA13_TRIAL: ''}):
            self.assertEqual(compiler_qualification(
                'Cuda compilation tools, release 12.8, V12.8.93',
                torch_cuda='13.0'), 'validated_cuda_12_8')

    def test_cuda13_fails_closed_by_default(self):
        with patch.dict(os.environ, {PTD_2NFE_CUDA13_TRIAL: ''}):
            with self.assertRaisesRegex(RuntimeError, 'unqualified'):
                compiler_qualification(
                    'Cuda compilation tools, release 13.0, V13.0.88',
                    torch_cuda='13.0')

    def test_cuda13_requires_matching_torch(self):
        with patch.dict(os.environ, {PTD_2NFE_CUDA13_TRIAL: '1'}):
            with self.assertRaisesRegex(RuntimeError, 'torch_cuda=12.8'):
                compiler_qualification(
                    'Cuda compilation tools, release 13.0, V13.0.88',
                    torch_cuda='12.8')
            self.assertEqual(compiler_qualification(
                'Cuda compilation tools, release 13.0, V13.0.88',
                torch_cuda='13.0'), 'experimental_cuda_13_0_ptd_2nfe_only')

    def test_unrelated_compiler_never_admitted(self):
        with patch.dict(os.environ, {PTD_2NFE_CUDA13_TRIAL: '1'}):
            for version in ('release 12.9', 'release 13.1', 'release 11.8'):
                with self.subTest(version=version), self.assertRaises(RuntimeError):
                    compiler_qualification(version, torch_cuda='13.0')

    def test_prefer_12_8_over_explicit_13_0(self):
        with TemporaryDirectory() as directory:
            native = Path(directory) / 'native_nvcc'
            validated = Path(directory) / 'validated_nvcc'
            native.touch()
            validated.touch()
            def check(argv, **kwargs):
                return ('release 12.8' if Path(argv[0]) == validated
                        else 'release 13.0') + ', Vtest'
            with patch.dict(os.environ, {'CUDACXX': str(native),
                                          PTD_2NFE_CUDA13_TRIAL: '1'}), \
                 patch('shutil.which', return_value=str(validated)), \
                 patch('subprocess.check_output', side_effect=check):
                result = configure_compiler(_TileLangPolicy())
                self.assertEqual(result['qualification'], 'validated_cuda_12_8')
                self.assertEqual(os.environ['CUDACXX'], str(validated.resolve()))
                self.assertEqual(os.environ['CUDA_HOME'], str(validated.parent.parent.resolve()))

    def test_cuda13_fallback_requires_explicit_scope(self):
        import torch
        with TemporaryDirectory() as directory:
            native = Path(directory) / 'nvcc'
            native.touch()
            with patch('shutil.which', return_value=str(native)), \
                 patch('subprocess.check_output',
                       return_value='Cuda compilation tools, release 13.0, V13.0.88'), \
                 patch.object(torch.version, 'cuda', '13.0'):
                with patch.dict(os.environ, {'CUDACXX': str(native),
                                              PTD_2NFE_CUDA13_TRIAL: '1'}):
                    result = configure_compiler(_TileLangPolicy())
                    self.assertEqual(result['qualification'],
                                     'experimental_cuda_13_0_ptd_2nfe_only')
                with patch.dict(os.environ, {'CUDACXX': str(native),
                                              PTD_2NFE_CUDA13_TRIAL: ''}):
                    with self.assertRaises(RuntimeError):
                        configure_compiler(_TileLangPolicy())

    def test_aot_uses_shared_cuda13_gate(self):
        import torch
        from patches import h3_operator_aot
        with patch.dict(os.environ, {'CUDACXX': '/tmp/synthetic/nvcc',
                                      PTD_2NFE_CUDA13_TRIAL: '1'}), \
             patch.object(h3_operator_aot.subprocess, 'check_output',
                          return_value='Cuda compilation tools, release 13.0, V13.0.88'), \
             patch.object(h3_operator_aot, 'version', return_value='0.1.13'), \
             patch.object(torch.version, 'cuda', '13.0'):
            h3_operator_aot.environment.cache_clear()
            try:
                data = h3_operator_aot.environment()
                self.assertEqual(data['compiler_qualification'],
                                 'experimental_cuda_13_0_ptd_2nfe_only')
            finally:
                h3_operator_aot.environment.cache_clear()


class _TileLangPolicy:
    dit_attention = 'kitchen'
    row_linear_backend = 'tilelang_scaled_fp16'
    column_linear_backend = 'preallocated'


if __name__ == '__main__':
    unittest.main()
