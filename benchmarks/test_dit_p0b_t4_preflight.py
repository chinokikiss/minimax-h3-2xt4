"""CPU-only fail-closed qualification for one physical dual-T4 P0-B A/B."""
import ast
from importlib.metadata import version
from pathlib import Path
import unittest

from patches.h3_operator_policy import H3OperatorPolicy


ROOT=Path(__file__).resolve().parents[1]


class P0BPhysicalPreflight(unittest.TestCase):
    def test_environment_and_real_production_defaults(self):
        self.assertEqual(version('comfy-kitchen'),'0.2.37')
        self.assertEqual(version('tilelang'),'0.1.13')
        p=H3OperatorPolicy()
        for key in ('dit_p0_2_lifetime','dit_p0_3_lifetime',
                    'dit_p0_a1_turing','dit_p0_c_local_rotation'):
            self.assertIs(getattr(p,key),True)
        self.assertEqual(p.fc2_quantizer,'fused_stored')

    def test_frozen_experiment_never_imported_by_production(self):
        for path in ('patches/tp_dit.py','patches/tilelang_swiglu_quant.py',
                     'patches/h3_operator_policy.py','patches/dit_column_linear.py'):
            text=(ROOT/path).read_text(encoding='utf-8')
            self.assertNotIn('dit_p0b_inplace_probe',text)
            self.assertNotIn('dit_p0b_t4_candidates',text)

    def test_new_kernel_has_valid_alias_contract(self):
        source=(ROOT/'benchmarks/dit_p0b_inplace_probe.py').read_text(encoding='utf-8')
        ast.parse(source)
        self.assertIn("R: T.Tensor((rows, width), 'float32')",source)
        self.assertIn('transform(X, a, b, row, group)',source)
        self.assertIn('T.sync_threads()',source)
        self.assertIn('view = x.view(torch.float32)',source)
        self.assertIn("out_idx=[2]",source)
        self.assertIn('width // 2 + group * 128 + (i - 128)',source)
        self.assertIn("'__fdividef'",source)
        self.assertIn("'nearbyintf'",source)

    def test_worker_scope_and_real_collectives(self):
        candidate=(ROOT/'benchmarks/dit_p0b_t4_candidates.py').read_text(encoding='utf-8')
        worker=(ROOT/'benchmarks/run_dit_p0b_case.py').read_text(encoding='utf-8')
        batch=(ROOT/'benchmarks/run_dit_p0b_batch.py').read_text(encoding='utf-8')
        for text in (candidate,worker,batch):ast.parse(text)
        self.assertIn("dist.init_process_group('nccl'",worker)
        self.assertIn("probe_codes(rows,device,dist,rank)",worker)
        self.assertIn("active_p0b(name=='p0_b',hits[name])",worker)
        self.assertIn("bytes_compare(trial.detach().cpu(),reference)",worker)
        self.assertIn("baseline_phases(block,observer)",worker)
        self.assertIn("warmups=3,repeats=9",worker)
        self.assertIn("torch.cuda.reset_peak_memory_stats(device)",worker)
        self.assertIn("assert block.out_proj._vram_p0_c is True",worker)
        self.assertIn("assert block.fc1._vram_p0_a1 is True",worker)
        self.assertIn("consume_input and type(x) is torch.Tensor",candidate)
        self.assertIn("x.set_()",candidate)
        self.assertIn("process.wait(timeout=2100)",batch)
        self.assertIn('torch.distributed.run',batch)
        for text in (candidate,worker,batch):
            self.assertNotIn('kernels status',text)
            self.assertNotIn('kernels list',text)
            self.assertNotIn('kaggle kernels output',text)


if __name__=='__main__':
    unittest.main(verbosity=2)
