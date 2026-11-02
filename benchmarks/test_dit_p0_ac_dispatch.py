"""CPU-only A1/C production routing checks; physical T4 proof is separate."""
import unittest
from unittest.mock import patch

import torch
import comfy_kitchen as ck
from comfy_kitchen.backends import cuda as kitchen

from patches import dit_column_linear as column
from patches import dit_kitchen_turing_a1 as a1
from patches import tilelang_row_quant as row


class P0ACDispatch(unittest.TestCase):
    def setUp(self):
        self.x = torch.zeros((3, 5376), dtype=torch.float16)
        self.w = torch.zeros((14336, 5376), dtype=torch.int8)
        self.scale = torch.ones((14336, 1), dtype=torch.float32)

    def test_no_cuda_no_input_mutation_and_none_return(self):
        x = self.x.clone()
        self.assertIsNone(a1.turing_column_a1(x, self.w, self.scale,
            convrot=True, convrot_groupsize=256, out_dtype=torch.float16))
        self.assertEqual(x.numel(), self.x.numel())
        self.assertIsNone(row._local_o_proj_p0c(
            torch.zeros((128, 3584)), object(), 2, 256, None, 'packed', True))

    def test_a1_selected_only_for_pinned_kitchen_cuda_branch(self):
        sentinel = torch.empty((3, 14336), dtype=torch.float16)
        with torch.no_grad(), \
             patch.object(ck.registry, 'get_implementation', return_value=kitchen.int8_linear), \
             patch.object(a1, 'turing_column_a1', return_value=sentinel) as optimized, \
             patch.object(ck, 'int8_linear', side_effect=AssertionError('fallback used')):
            actual = column.column_linear(self.x, self.w, self.scale,
                backend='cuda_int8_tilelang', p0_a1=True)
        self.assertIs(actual, sentinel)
        self.assertEqual(optimized.call_count, 1)

    def test_a1_explicit_off_uses_public_kitchen(self):
        sentinel = torch.empty((3, 14336), dtype=torch.float16)
        with torch.no_grad(), \
             patch.object(ck.registry, 'get_implementation', return_value=kitchen.int8_linear), \
             patch.object(a1, 'turing_column_a1', side_effect=AssertionError('A1 called')), \
             patch.object(ck, 'int8_linear', return_value=sentinel) as public:
            actual = column.column_linear(self.x, self.w, self.scale,
                backend='cuda_int8_tilelang', p0_a1=False)
        self.assertIs(actual, sentinel)
        self.assertEqual(public.call_count, 1)

    def test_a1_pinned_source_contains_exact_turing_gate(self):
        from inspect import getsource
        source = getsource(kitchen.int8_linear)
        for token in ('_prefer_turing_fused_int8(m, n, k)',
                      '_cuda_device_should_use_turing_kernels',
                      '_int8_linear_turing_quantized(', 'quantize_and_rotate_rowwise('):
            self.assertIn(token, source)


if __name__ == '__main__':
    unittest.main()
