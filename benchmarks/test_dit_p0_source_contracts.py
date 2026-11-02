"""Offline source-contract gate for the promoted P0-3/P0-2 runtime.

This suite uses only Python's standard library so that it can run on the local
Windows authoring host without a CUDA/PyTorch installation. Numerical equality,
allocator peaks, and warmed latency are *not* certified by these tests.
"""
import ast
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]


def source(name):
    return (ROOT / "patches" / name).read_text(encoding="utf-8")


def function(tree, name):
    candidates = [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name]
    if len(candidates) != 1:
        raise AssertionError((name, len(candidates)))
    return candidates[0]


def default_false(fn, name):
    args = fn.args.args + fn.args.kwonlyargs
    all_defaults = [None] * (len(fn.args.args) - len(fn.args.defaults)) + list(fn.args.defaults) + list(fn.args.kw_defaults)
    item = all_defaults[[arg.arg for arg in args].index(name)]
    return isinstance(item, ast.Constant) and item.value is False


def call_lines(fn, attribute):
    found = []
    for node in ast.walk(fn):
        if isinstance(node, ast.Call) and (getattr(node.func, "attr", None) == attribute or
                                           getattr(node.func, "id", None) == attribute):
            found.append(node.lineno)
    return sorted(found)


class P0SourceContracts(unittest.TestCase):
    def test_all_modified_runtime_sources_parse(self):
        for name in ("tp_dit.py", "dit_column_linear.py", "tilelang_dit_fusion.py",
                     "tilelang_transport_p2.py", "tilelang_row_quant.py", "tilelang_swiglu_quant.py",
                     "tilelang_swiglu_inplace.py"):
            with self.subTest(file=name):
                ast.parse(source(name))

    def test_p03_stored_kernel_is_last_reader_and_metadata_is_saved(self):
        tree = ast.parse(source("tilelang_swiglu_quant.py"))
        prepare = function(tree, "prepare")
        row = function(tree, "row_linear")
        self.assertTrue(default_false(prepare, "consume_input"))
        self.assertTrue(default_false(row, "consume_input"))
        self.assertLess(call_lines(prepare, "fused_kernel")[0], call_lines(prepare, "set_")[0])
        self.assertLess(call_lines(prepare, "set_")[0], call_lines(prepare, "quantize")[0])
        text = ast.get_source_segment(source("tilelang_swiglu_quant.py"), row)
        self.assertLess(text.index("shape=x.shape"), text.index("prepare(x,comm,mode,consume_input=consume_input,p0_b=p0_b)"))
        self.assertIn("P0-3 consumption requires fused_stored", source("tilelang_swiglu_quant.py"))

    def test_p02_copy_and_packing_precede_detachment(self):
        fusion = function(ast.parse(source("tilelang_dit_fusion.py")), "row_gated")
        quant = function(ast.parse(source("tilelang_row_quant.py")), "global_row_quantize")
        transport = function(ast.parse(source("tilelang_transport_p2.py")), "row_linear")
        self.assertTrue(default_false(quant, "consume_input"))
        self.assertTrue(default_false(transport, "consume_input"))
        self.assertLess(call_lines(fusion, "float")[0], call_lines(fusion, "set_")[0])
        self.assertLess(call_lines(quant, "packing_kernel")[0], call_lines(quant, "set_")[0])
        self.assertLess(call_lines(quant, "set_")[0], call_lines(quant, "_rotate_activation")[0])
        self.assertLess(call_lines(transport, "gather")[0], call_lines(transport, "set_")[0])
        text = ast.get_source_segment(source("tilelang_transport_p2.py"), transport)
        self.assertLess(text.index("shape=x.shape"), text.index("global_row_quantize("))
        self.assertIn("consume_input=consume", ast.get_source_segment(source("tilelang_dit_fusion.py"), fusion))

    def test_p0b_default_dispatch_preserves_p03_and_kernel_is_owner_scoped(self):
        p=source('h3_operator_policy.py')
        fusion=source('tilelang_dit_fusion.py')
        fused=source('tilelang_swiglu_quant.py')
        inplace=source('tilelang_swiglu_inplace.py')
        self.assertIn('dit_p0_b_inplace: bool = True',p)
        self.assertIn('block.fc2._vram_p0_b=policy.dit_p0_b_inplace',p)
        self.assertIn("p0_b=bool(consume and getattr(module,'_vram_p0_b',False))",fusion)
        self.assertIn('p0_b=p0_b',fusion)
        self.assertIn('if result is not None:return result',fused)
        self.assertIn('x._base is not None',inplace)
        self.assertIn('x.storage_offset() != 0',inplace)
        self.assertIn('x.set_()',inplace)
        self.assertNotIn('from benchmarks',inplace)
        self.assertTrue(default_false(function(ast.parse(fused),'prepare'),'p0_b'))

    def test_p0b_t4_qualified_kernel_bodies_are_promoted_unchanged(self):
        experiment=ast.parse((ROOT/'benchmarks'/'dit_p0b_inplace_probe.py').read_text(encoding='utf-8'))
        production=ast.parse(source('tilelang_swiglu_inplace.py'))
        for name in ('inplace_kernel','remapped_quantizer'):
            with self.subTest(kernel=name):
                self.assertEqual(ast.dump(function(production,name)),
                                 ast.dump(function(experiment,name)))

    def test_p0c_all_positive_rows_without_relaxing_hardware_or_owner_gates(self):
        tree=ast.parse(source('tilelang_row_quant.py'))
        body=ast.get_source_segment(source('tilelang_row_quant.py'),
            function(tree,'_local_o_proj_p0c'))
        self.assertIn('or x.shape[0]<=0',body)
        self.assertNotIn('x.shape[0] not in',body)
        for required in ("variant!='packed'",'world_size!=2','groupsize!=256',
                         'x._base is not None','not x.is_contiguous()',
                         "x.dtype!=torch.float32",'torch.is_grad_enabled()',
                         'torch.compiler.is_compiling()',
                         'torch.cuda.get_device_capability(x.device)!=(7,5)'):
            with self.subTest(guard=required):self.assertIn(required,body)

    def test_removed_p01_has_no_runtime_entrypoints(self):
        for name in ('tp_dit.py','dit_column_linear.py','h3_operator_policy.py'):
            with self.subTest(file=name):
                body=source(name)
                for token in ('_vram_p0_1','inplace_quant','consume_owner'):
                    self.assertNotIn(token,body)

    def test_production_lifetime_flags_are_scoped_and_lora_is_guarded(self):
        tp = source("tp_dit.py")
        fusion = source("tilelang_dit_fusion.py")
        self.assertIn("adapter is None", tp)
        self.assertIn("getattr(module,'lora_adapter',None) is not None", fusion)
        for flag in ("_vram_p0_3", "_vram_p0_2"):
            self.assertIn("getattr(module,'"+flag+"',False)", fusion)
        policy = source("h3_operator_policy.py")
        for flag in ("dit_p0_2_lifetime", "dit_p0_3_lifetime"):
            self.assertIn(flag+": bool = True", policy)
        self.assertIn("block.out_proj._vram_p0_2=policy.dit_p0_2_lifetime", policy)
        self.assertIn("block.fc2._vram_p0_3=policy.dit_p0_3_lifetime", policy)
        self.assertIn("block.fc1._vram_p0_3_producer=policy.dit_p0_3_lifetime", policy)
        self.assertNotIn("block.qkv_proj._vram_p0_", policy)


if __name__ == "__main__":
    unittest.main(verbosity=2)
