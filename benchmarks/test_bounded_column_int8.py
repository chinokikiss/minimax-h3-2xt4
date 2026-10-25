"""Numerical, aliasing and live-temporary guards for the column experiment."""
import unittest
import weakref
import torch
import comfy_kitchen as ck
from benchmarks.bounded_column_int8 import bounded_int8_linear, preallocated_eager_int8_linear


class BoundedColumn(unittest.TestCase):
    @torch.no_grad()
    def test_pinned_eager_exact_with_rotation_scales_and_row_tails(self):
        torch.manual_seed(247)
        for dtype in (torch.float16, torch.float32):
            for rotation in (False, True):
                for per_channel in (False, True):
                    x = torch.randn(2, 13, 64, dtype=dtype)
                    w = torch.randint(-127, 128, (48, 64), dtype=torch.int8)
                    scale = torch.rand(48, 1) * .003 if per_channel else torch.tensor(.002)
                    before = x.clone()
                    with ck.registry.use_backend('eager'):
                        expected = ck.int8_linear(x, w, scale, None, dtype, convrot=rotation, convrot_groupsize=16)
                        actual = bounded_int8_linear(x, w, scale, chunk_rows=7, out_dtype=dtype,
                                                    convrot=rotation, convrot_groupsize=16)
                    self.assertTrue(torch.equal(expected, actual), (dtype, rotation, per_channel))
                    preallocated = preallocated_eager_int8_linear(x, w, scale, out_dtype=dtype,
                                                               convrot=rotation, convrot_groupsize=16)
                    self.assertTrue(torch.equal(expected, preallocated), (dtype, rotation, per_channel))
                    self.assertTrue(torch.equal(before, x))

    @torch.no_grad()
    def test_peak_row_bound_and_temporary_lifetime(self):
        alive = []
        rows = []
        def linear(x, w, scale, bias, dtype, **kwargs):
            self.assertFalse(any(ref() is not None for ref in alive))
            rows.append(x.shape[0])
            output = x @ w.float().T
            alive.append(weakref.ref(output))
            return output
        x = torch.arange(13 * 16, dtype=torch.float32).reshape(13, 16)
        w = torch.ones(8, 16, dtype=torch.int8)
        actual = bounded_int8_linear(x, w, torch.tensor(1.), chunk_rows=5,
                                    out_dtype=torch.float32, linear=linear)
        self.assertEqual(rows, [5, 5, 3])
        self.assertTrue(torch.equal(actual, x @ w.float().T))
        self.assertFalse(any(ref() is not None for ref in alive))

    def test_inference_guard_and_invalid_budget(self):
        x = torch.ones(2, 16)
        w = torch.ones(8, 16, dtype=torch.int8)
        with self.assertRaises(RuntimeError):
            bounded_int8_linear(x, w, torch.tensor(1.))
        with torch.no_grad():
            for budget in (0, -1, True, 1.5):
                with self.assertRaises(ValueError):
                    bounded_int8_linear(x, w, torch.tensor(1.), chunk_rows=budget)


if __name__ == '__main__':
    unittest.main()
