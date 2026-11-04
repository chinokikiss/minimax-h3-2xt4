"""CPU-only gates for isolated CUDA13 physical-SM75 operator qualification."""
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from patches.h3_operator_dependencies import (
    ALL_OPS_CUDA13_T4_TRIAL, PTD_E2E_CUDA13_TRIAL,
    TURBO_15S_CUDA13_TRIAL, compiler_qualification,
    sm75_native_compiler_qualification,
)


class Cu130T4CompilerGuards(unittest.TestCase):
    def test_dedicated_cuda130_requires_matching_torch_and_opt_in(self):
        flags={ALL_OPS_CUDA13_T4_TRIAL:'',TURBO_15S_CUDA13_TRIAL:'',
               PTD_E2E_CUDA13_TRIAL:''}
        with patch.dict(os.environ,flags):
            with self.assertRaises(RuntimeError):
                compiler_qualification('Cuda compilation tools, release 13.0, V13.0.88',
                                       torch_cuda='13.0')
            with self.assertRaises(RuntimeError):
                sm75_native_compiler_qualification(
                    'Cuda compilation tools, release 13.0, V13.0.88',torch_cuda='13.0')
        with patch.dict(os.environ,{**flags,ALL_OPS_CUDA13_T4_TRIAL:'1'}):
            self.assertEqual(compiler_qualification('release 13.0, V13.0.88',
                                                    torch_cuda='13.0'),
                'experimental_cuda_13_0_all_ops_t4_only')
            self.assertEqual(sm75_native_compiler_qualification(
                'release 13.0, V13.0.88',torch_cuda='13.0'),
                'experimental_cuda_13_0_all_ops_t4_only')
            for tc in ('12.8',None):
                with self.subTest(torch_cuda=tc),self.assertRaises(RuntimeError):
                    sm75_native_compiler_qualification('release 13.0, V13.0.88',
                                                       torch_cuda=tc)
            for release in ('12.9','13.1'):
                with self.subTest(release=release),self.assertRaises(RuntimeError):
                    compiler_qualification('release '+release,torch_cuda='13.0')

    def test_original_cuda128_native_unchanged(self):
        self.assertEqual(sm75_native_compiler_qualification(
            'Cuda compilation tools, release 12.8, V12.8.93',
            torch_cuda='13.0'),'validated_cuda_12_8_93')
        with self.assertRaises(RuntimeError):
            sm75_native_compiler_qualification('release 12.8, V12.8.99')

    def test_sol_dedicated_trial_admitted_without_turbo(self):
        from patches import sol_attention
        with TemporaryDirectory() as directory:
            nvcc=Path(directory)/'nvcc'
            with patch.dict(os.environ,{'CUDACXX':str(nvcc),
                       ALL_OPS_CUDA13_T4_TRIAL:'1',TURBO_15S_CUDA13_TRIAL:''}),\
                 patch.object(sol_attention.subprocess,'check_output',
                              return_value='release 13.0, V13.0.88'),\
                 patch.object(sol_attention.torch.version,'cuda','13.0'):
                compiler,_=sol_attention._compiler()
                self.assertEqual(compiler,nvcc.resolve())

    def test_native_adapters_bind_shared_version_gate(self):
        import inspect
        from patches import kitchen_attention,kitchen_te_attention
        self.assertIn('sm75_native_compiler_qualification',
                      inspect.getsource(kitchen_attention.build))
        self.assertIn('sm75_native_compiler_qualification',
                      inspect.getsource(kitchen_attention.validate_library))
        self.assertIn('sm75_native_compiler_qualification',
                      inspect.getsource(kitchen_te_attention.validate_library))


if __name__=='__main__':
    unittest.main()
