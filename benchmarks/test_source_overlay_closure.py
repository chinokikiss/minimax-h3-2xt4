"""The Sol E2E overlay must carry every changed internal dependency."""
from pathlib import Path
import json
import unittest

from benchmarks.build_sol_hybrid_e2e import CORE_OVERLAY
from benchmarks.source_overlay_closure import overlay_files,assert_closed

ROOT=Path(__file__).resolve().parents[1]
BASELINE_JOB=Path(r'D:/attention_eval/h3_turbo_085mp_10s_20261029/retry2/kaggle/gpu_job')


class OverlayClosure(unittest.TestCase):
    def baseline_commit(self):
        if not BASELINE_JOB.is_dir():
            self.skipTest('Completed baseline job is unavailable')
        binding=json.loads((BASELINE_JOB/'source_binding.json').read_text(encoding='utf-8'))
        return binding['publication']['commit']

    def test_changed_dependency_closure_contains_known_runtime_edges(self):
        commit=self.baseline_commit()
        files,closure=overlay_files(commit,CORE_OVERLAY)
        self.assertEqual(assert_closed(commit,files,CORE_OVERLAY),closure)
        required={
            'patches/kitchen_attention.py',
            'patches/file_metadata.py',
            'patches/comfy_sampler.py',
            'patches/tp2_bundle.py',
            'patches/windowed_tp.py',
            'patches/tp_dit.py',
            'patches/tp_qwen3vl.py',
            'patches/tilelang_swiglu_quant.py',
            'patches/tilelang_fp32_rms_rope.py',
            'custom_nodes/minimax_h3_tp2/__init__.py',
            'patches/h3_configuration.py',
        }
        self.assertTrue(required.issubset(set(closure)),sorted(required-set(closure)))
        self.assertTrue(any(name.startswith('patches/kitchen_attention_native/') for name in files))
        self.assertTrue(any(name.startswith('patches/sol_attention_native/') for name in files))
        self.assertNotIn('experiments/kitchen_attention/delivery.py',files)
        self.assertNotIn('experiments/kitchen_attention/dense_delivery.py',files)
        for license_path in (
                'patches/sol_attention_native/STAR7_LICENSE',
                'patches/sol_attention_native/third_party/comfy_kitchen_sage/LICENSE',
                'patches/sol_attention_native/third_party/sage_sm75_fp16/LICENSE',
                'patches/kitchen_attention_native/vendor/LICENSE',
                'patches/kitchen_attention_native/vendor/NOTICE'):
            self.assertIn(license_path,files)

    def test_retired_sparse_attention_is_not_reintroduced(self):
        commit=self.baseline_commit()
        files,_=overlay_files(commit,CORE_OVERLAY)
        retired=('tilelang_group32','tilelang_sparse_attention','tilelang_grouped_tp_optimized',
                 'tilelang_attention_p2','tilelang_scalar_pool')
        for name in files:
            self.assertFalse(any(token in name for token in retired),name)


if __name__=='__main__':
    unittest.main(verbosity=2)
