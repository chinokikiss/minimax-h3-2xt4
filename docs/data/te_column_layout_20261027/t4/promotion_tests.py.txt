"""Numerical, aliasing and dispatch contracts for column-only layout changes."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
import comfy_kitchen as ck

from benchmarks.te_column_layout import (
    accumulator, balanced_order, column_layout, legacy_copy, torch_view,
)
from patches import dit_column_linear as column


class ColumnLayout(unittest.TestCase):
    @torch.no_grad()
    def test_view_preserves_storage_and_integer_results(self):
        torch.manual_seed(2710)
        for rows in (1, 13, 32, 33):
            codes = torch.randint(-128, 128, (rows, 64), dtype=torch.int8)
            weight = torch.randint(-128, 128, (48, 64), dtype=torch.int8)
            original_codes, original_weight = codes.clone(), weight.clone()
            self.assertEqual(weight.T.data_ptr(), weight.data_ptr())
            self.assertFalse(weight.T.is_contiguous())
            self.assertTrue(torch.equal(legacy_copy(codes, weight), torch_view(codes, weight)))
            self.assertTrue(torch.equal(codes, original_codes))
            self.assertTrue(torch.equal(weight, original_weight))

    @torch.no_grad()
    def test_full_range_int8_bound_and_signed_extremes(self):
        # H3 K=5120 is safely within INT32 even for every operand equal to -128.
        codes = torch.full((2, 5120), -128, dtype=torch.int8)
        weight = torch.full((8, 5120), -128, dtype=torch.int8)
        expected = torch.full((2, 8), 5120 * 128 * 128, dtype=torch.int32)
        self.assertTrue(torch.equal(torch_view(codes, weight), expected))

    @torch.no_grad()
    def test_shared_projection_outputs_exact_and_no_weight_cache(self):
        torch.manual_seed(2711)
        weights = [(torch.randint(-127, 128, (width, 64), dtype=torch.int8),
                    torch.logspace(-5, -2, width).reshape(-1, 1))
                   for width in (32, 8, 8)]
        module_fields = set(vars(column))
        for dtype in (torch.float16, torch.float32):
            x = torch.randn(2, 13, 64, dtype=dtype)
            with ck.registry.use_backend('eager'):
                expected = column.shared_column_linears(x, weights, convrot_groupsize=16)
                with column_layout('torch_view'):
                    actual = column.shared_column_linears(x, weights, convrot_groupsize=16)
            self.assertTrue(all(torch.equal(a, b) for a, b in zip(actual, expected)))
        self.assertEqual(set(vars(column)), module_fields)

    def test_scope_restores_after_error_and_does_not_patch_global_helper(self):
        from comfy_kitchen.backends.eager import quantization as eager
        original, global_helper = column._column_accumulate, eager._int8_matmul_accumulate
        with self.assertRaisesRegex(RuntimeError, 'test'):
            with column_layout('torch_view'):
                self.assertIs(column._column_accumulate, torch_view)
                self.assertIs(eager._int8_matmul_accumulate, global_helper)
                raise RuntimeError('test')
        self.assertIs(column._column_accumulate, original)

    @torch.no_grad()
    def test_default_keeps_contiguous_rhs_and_direct_is_explicit(self):
        codes = torch.ones(3, 64, dtype=torch.int8)
        weight = torch.ones(32, 64, dtype=torch.int8)
        from comfy_kitchen.backends.eager import quantization as eager
        with patch.object(eager, '_int8_matmul_accumulate', wraps=eager._int8_matmul_accumulate) as call:
            column._column_accumulate(codes, weight)
            self.assertEqual(call.call_count, 1)
            self.assertTrue(call.call_args.args[1].is_contiguous())
        with patch('patches.optimized_tp.native_int_mm', return_value='direct') as direct:
            self.assertEqual(accumulator('kitchen_direct')(codes, weight), 'direct')
            direct.assert_called_once_with(codes, weight)
        with self.assertRaises(ValueError):
            accumulator('unknown')

    def test_three_repetitions_balance_variant_positions(self):
        orders = [balanced_order(i) for i in range(3)]
        for position in range(3):
            self.assertEqual(len({order[position] for order in orders}), 3)

    def test_runtime_guard_requires_the_qualified_device_and_geometry(self):
        def tensor(shape, **changes):
            values = dict(shape=shape, ndim=len(shape), is_cuda=True, dtype=torch.int8,
                          device=torch.device('cuda:0'), is_contiguous=lambda: True)
            values.update(changes)
            return SimpleNamespace(**values)
        codes = tensor((353, 5120))
        with patch('torch.cuda.get_device_capability', return_value=(7, 5)):
            for width in (512, 4096, 12800):
                self.assertTrue(column._column_view_eligible(codes, tensor((width, 5120))))
            for value in (tensor((8192, 5120)), tensor((4096, 5376)),
                          tensor((4096, 5120), is_cuda=False),
                          tensor((4096, 5120), dtype=torch.float16),
                          tensor((4096, 5120), device=torch.device('cuda:1')),
                          tensor((4096, 5120), is_contiguous=lambda: False)):
                self.assertFalse(column._column_view_eligible(codes, value))
            self.assertFalse(column._column_view_eligible(tensor((0, 5120)), tensor((4096, 5120))))
        with patch('torch.cuda.get_device_capability', return_value=(12, 0)):
            self.assertFalse(column._column_view_eligible(codes, tensor((4096, 5120))))

    @torch.no_grad()
    def test_qualified_dispatch_uses_original_storage_and_preserves_padding(self):
        from comfy_kitchen.backends.eager import quantization as eager
        codes = torch.randint(-127, 128, (33, 64), dtype=torch.int8)
        weight = torch.randint(-127, 128, (32, 64), dtype=torch.int8)
        expected = legacy_copy(codes, weight)
        with patch.object(column, '_column_view_eligible', return_value=True), \
                patch.object(eager, '_int8_matmul_accumulate', wraps=eager._int8_matmul_accumulate) as call:
            actual = column._column_accumulate(codes, weight)
            rhs = call.call_args.args[1]
            self.assertEqual(rhs.data_ptr(), weight.data_ptr())
            self.assertEqual(rhs.stride(), weight.T.stride())
            self.assertFalse(rhs.is_contiguous())
        self.assertTrue(torch.equal(actual, expected))


if __name__ == '__main__':
    unittest.main()
