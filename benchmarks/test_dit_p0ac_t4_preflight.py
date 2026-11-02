"""CPU-only fail-closed preflight for the frozen physical dual-T4 P0-A/P0-C job."""
import ast
import inspect
from importlib.metadata import version
from pathlib import Path
import unittest

import comfy_kitchen.backends.cuda as kitchen
from benchmarks.dit_p0ac_t4_candidates import build_p0a1

ROOT = Path(__file__).resolve().parents[1]


class PhysicalT4P0ACPreflight(unittest.TestCase):
    def test_pinned_kitchen_and_experiment_clone(self):
        self.assertEqual(version("comfy-kitchen"), "0.2.37")
        assert callable(build_p0a1({}))
        body = inspect.getsource(kitchen.int8_linear)
        self.assertIn("turing_out = _int8_linear_turing_quantized(", body)
        self.assertIn("return _finish(turing_out)", body)
        self.assertTrue(kitchen._prefer_turing_fused_int8(59134, 10752, 5376))
        self.assertTrue(kitchen._prefer_turing_fused_int8(65166, 14336, 5376))

    def test_current_production_defaults_are_intact(self):
        from patches.h3_operator_policy import H3OperatorPolicy
        policy = H3OperatorPolicy()
        self.assertTrue(policy.dit_p0_2_lifetime)
        self.assertTrue(policy.dit_p0_3_lifetime)
        for name in ("tp_dit.py", "dit_column_linear.py"):
            text = (ROOT / "patches" / name).read_text(encoding="utf-8")
            for retired in ("_vram_p0_1", "_vram_p0_1_inplace", "inplace_quant", "consume_owner"):
                self.assertNotIn(retired, text)

    def test_experiment_only_backend_and_owner_guards(self):
        name = "benchmarks/dit_p0ac_t4_candidates.py"
        source = (ROOT / name).read_text(encoding="utf-8")
        ast.parse(source)
        for required in (
            "torch.cuda.get_device_capability", "Kitchen CUDA column registry was not selected",
            "x._base is not None", "consume_input", "x.set_()",
            "row.maximum_kernel(rows, ROW_WIDTH, ROW_WIDTH, 0)",
            "comm.all_reduce(maximum, op=comm.ReduceOp.MAX)",
            "row.quantize(rotated, maximum, ROW_WIDTH, 0)",
            "Turing CUTLASS branch",
        ):
            self.assertIn(required, source)
        self.assertNotIn("_vram_p0_1", source)

    def test_worker_variants_are_production_scoped(self):
        worker = (ROOT / "benchmarks/run_dit_p0ac_case.py").read_text(encoding="utf-8")
        tree = ast.parse(worker)
        self.assertIn('VARIANTS = ("baseline", "p0_a1", "p0_c", "p0_a1_p0_c")', worker)
        self.assertIn("assert block.out_proj._vram_p0_2 is True", worker)
        self.assertIn("assert block.fc2._vram_p0_3 is True", worker)
        self.assertIn("assert block.fc1._vram_p0_3_producer is True", worker)
        self.assertIn("probe_p0c(probe_rows, device, dist)", worker)
        self.assertIn("bytes_compare(output.detach().cpu(), reference)", worker)
        self.assertIn("baseline_phases(block, observer)", worker)
        self.assertNotIn("inplace_quant", worker)
        self.assertIsInstance(tree, ast.Module)

    def test_batch_is_bounded_and_has_no_status_polling(self):
        source = (ROOT / "benchmarks/run_dit_p0ac_batch.py").read_text(encoding="utf-8")
        ast.parse(source)
        self.assertIn("process.wait(timeout=2100)", source)
        self.assertIn("torch.distributed.run", source)
        self.assertIn("NvidiaTeslaT4", source) if "NvidiaTeslaT4" in source else self.assertIn('"T4"', source)
        self.assertNotIn("kernels status", source)
        self.assertNotIn("kernel_status", source)


if __name__ == "__main__":
    unittest.main(verbosity=2)
