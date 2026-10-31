"""CPU-only lease, feature-selection and source-bound Sol workspace contracts."""
import gc
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from patches import sol_sm75_backend as backend
from patches import sol_attention


class SolWorkspaceContracts(unittest.TestCase):
    def setUp(self):
        backend.clear_workspace_pool()

    def tearDown(self):
        backend.clear_workspace_pool()

    def test_one_stream_reuses_released_buffer(self):
        stream = SimpleNamespace(cuda_stream=9001)
        with patch.object(torch.cuda, "current_stream", return_value=stream):
            first = backend._acquire_workspace(torch.device("cpu"))
            tensor = backend._workspace_tensor(
                first, "threshold", (1, 4, 10), torch.float32, "cpu")
            tensor.fill_(73)
            first.release()
            second = backend._acquire_workspace(torch.device("cpu"))
            reused = backend._workspace_tensor(
                second, "threshold", (1, 4, 10), torch.float32, "cpu")
            self.assertEqual(tensor.data_ptr(), reused.data_ptr())
            self.assertEqual(float(reused[0, 0, 0]), 73)
            snapshot = backend.workspace_pool_snapshot()
            self.assertEqual(len(snapshot), 1)
            self.assertEqual(snapshot[0]["reuses"], 1)
            second.release()

    def test_simultaneous_prepared_leases_never_alias(self):
        stream = SimpleNamespace(cuda_stream=9002)
        with patch.object(torch.cuda, "current_stream", return_value=stream):
            first = backend._acquire_workspace(torch.device("cpu"))
            first_tensor = backend._workspace_tensor(
                first, "exact_mask", (1, 4, 10, 10), torch.uint8, "cpu")
            second = backend._acquire_workspace(torch.device("cpu"))
            second_tensor = backend._workspace_tensor(
                second, "exact_mask", (1, 4, 10, 10), torch.uint8, "cpu")
            self.assertNotEqual(first_tensor.data_ptr(), second_tensor.data_ptr())
            self.assertTrue(all(row["busy"] for row in backend.workspace_pool_snapshot()))
            first.release()
            second.release()

    def test_auto_release_on_failure_and_package_deletion(self):
        stream = SimpleNamespace(cuda_stream=9003)
        with patch.object(torch.cuda, "current_stream", return_value=stream):
            def fail_after_acquire():
                lease = backend._acquire_workspace(torch.device("cpu"))
                backend._workspace_tensor(lease, "qc", (4, 8), torch.float16, "cpu")
                raise RuntimeError("native kernel rejected")
            with self.assertRaisesRegex(RuntimeError, "native kernel rejected"):
                fail_after_acquire()
            gc.collect()
            self.assertFalse(backend.workspace_pool_snapshot()[0]["busy"])
            lease = backend._acquire_workspace(torch.device("cpu"))
            pkg = sol_attention._PreparedSol({"_h3_backend": "sol"},
                                              workspace_lease=lease)
            del lease
            self.assertTrue(backend.workspace_pool_snapshot()[0]["busy"])
            del pkg
            gc.collect()
            self.assertFalse(backend.workspace_pool_snapshot()[0]["busy"])

    def test_clear_never_invalidates_busy_owner(self):
        stream = SimpleNamespace(cuda_stream=9004)
        with patch.object(torch.cuda, "current_stream", return_value=stream):
            lease = backend._acquire_workspace(torch.device("cpu"))
            backend._workspace_tensor(lease, "lut", (2, 4), torch.int32, "cpu")
            backend.clear_workspace_pool()
            self.assertEqual(len(backend.workspace_pool_snapshot()), 1)
            lease.release()
            backend.clear_workspace_pool()
            self.assertEqual(backend.workspace_pool_snapshot(), [])

    def test_retired_fusion_is_not_runtime_selectable(self):
        with patch.object(sol_attention, '_source_identity', return_value={'x':'y'}):
            with patch.dict(os.environ, {'H3_SOL_FUSED_KC_THRESHOLD':'0'}):
                normal=sol_attention._cache_key('CUDA12.8')
            with patch.dict(os.environ, {'H3_SOL_FUSED_KC_THRESHOLD':'1'}):
                ignored=sol_attention._cache_key('CUDA12.8')
        self.assertEqual(normal,ignored)
        self.assertFalse(hasattr(sol_attention,'_fused_kc_enabled'))
        native=(Path(__file__).resolve().parents[1]/'patches/sol_attention_native/sol_sm75_preprocess.cuh').read_text()
        self.assertNotIn('compute_kc_stats_threshold_fused',native)


if __name__ == "__main__":
    unittest.main()
