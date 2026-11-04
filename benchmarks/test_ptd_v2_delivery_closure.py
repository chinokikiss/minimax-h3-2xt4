"""PTD v2 Kaggle builders must close current sources over their frozen baseline.

This catches delivery-only API skew such as a current tp2_bundle importing a
symbol added to rank_local_int8 after the frozen capsule commit.
"""
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / 'benchmarks'
if str(BENCH) not in sys.path:
    sys.path.insert(0, str(BENCH))

import build_ptd_v2_bake_job as bake_job
import build_ptd_v2_stage_a_job as stage_a_job
from source_overlay_closure import assert_closed


class PTDV2DeliveryClosure(unittest.TestCase):
    def _check(self, module):
        commit, files, closure = module._closed_overlay(module.DEFAULT_BASE)
        files = set(files)
        closure = set(closure)
        self.assertEqual(
            set(assert_closed(commit, files, module.CORE_OVERLAY)),
            closure,
        )
        required = {
            'patches/tp2_bundle.py',
            'patches/rank_local_int8.py',
            'patches/windowed_tp.py',
            'patches/file_metadata.py',
            'patches/tp2_disk_bake.py',
        }
        self.assertTrue(required.issubset(files), sorted(required - files))
        self.assertGreater(len(files), len(module.CORE_OVERLAY))
        source = (ROOT / 'patches/rank_local_int8.py').read_text(encoding='utf-8')
        self.assertIn('def check_state_schema(', source)
        return files

    def test_stage_a_overlay_is_dependency_closed(self):
        files = self._check(stage_a_job)
        for name in (
            'patches/comfy_tp_runtime.py',
            'patches/ptd/bank_v2.py',
            'patches/ptd/bake_v2.py',
            'patches/ptd/runtime.py',
            'benchmarks/run_ptd_v2_stage_a_t4.py',
        ):
            self.assertIn(name, files)

    def test_full_hop_bake_overlay_is_dependency_closed(self):
        files = self._check(bake_job)
        for name in (
            'patches/ptd/bank_v2.py',
            'patches/ptd/bake_v2.py',
            'patches/ptd/checkpoint.py',
            'benchmarks/run_ptd_v2_bake_t4.py',
        ):
            self.assertIn(name, files)


if __name__ == '__main__':
    unittest.main(verbosity=2)
