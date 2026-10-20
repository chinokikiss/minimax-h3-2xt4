"""Runtime lifecycle and diagnostic isolation contracts; no H3 generation."""
import atexit
import gc
from pathlib import Path
import subprocess
import sys
import threading
from types import SimpleNamespace
import unittest
import weakref

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


class PipelineReview(unittest.TestCase):
    def test_closed_session_is_not_retained_by_atexit(self):
        from patches.comfy_tp_runtime import PhysicalTPService
        from patches.host_weight_cache import HostBudget
        service=PhysicalTPService.__new__(PhysicalTPService)
        service.closed=False;service.processes=[];service.connections=[];service.store=None
        service.lock=threading.RLock();service.budget=HostBudget(1024,reserve_bytes=0)
        service.staged_vaes=weakref.WeakSet()
        reference=weakref.ref(service)
        atexit.register(service.close)
        service.abort()
        del service
        gc.collect()
        self.assertIsNone(reference())

    def test_runtime_audio_path_does_not_load_historical_manager_or_benchmarks(self):
        code='''import sys, torch
import patches.comfy_tp_runtime
from patches.parallel_vae import channel_decode
class Comm:
    def get_world_size(self):return 2
    def all_gather(self,outputs,x):
        for output in outputs:output.copy_(x)
class Model:
    def decode(self,x):return x.mean(1)
channel_decode(Model(),torch.ones(1,32,2,4),Comm(),0)
assert 'patches.multi_gpu_audio_vae' not in sys.modules
assert not any(n=='benchmarks' or n.startswith('benchmarks.') for n in sys.modules)
'''
        result=subprocess.run([sys.executable,'-c',code],cwd=ROOT,capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)

    def test_normal_launcher_preserves_node_availability_and_isolation_is_explicit(self):
        from launch_h3_legacy import comfy_command
        normal=comfy_command(ROOT/'ComfyUI')
        self.assertNotIn('--disable-partner-nodes',normal)
        self.assertNotIn('--disable-all-custom-nodes',normal)
        self.assertIn('--disable-dynamic-vram',normal)
        isolated=comfy_command(ROOT/'ComfyUI',isolated_ui=True)
        self.assertIn('--disable-all-custom-nodes',isolated)
        self.assertIn('--whitelist-custom-nodes',isolated)

    def test_explicit_diagnostic_snapshot_is_read_only(self):
        from patches.runtime_diagnostics import rank_snapshot
        from patches.h3_operator_policy import H3OperatorPolicy
        runtime=SimpleNamespace(config=SimpleNamespace(operators=H3OperatorPolicy(),adaptive_lookahead=False),
            rank=0,calls=3,kind='dit',sample_active=True,completed_decode=False,vae_kind=None,
            idle_stats={'loads':2},idle_window=None,events=[],
            cache=SimpleNamespace(snapshot=lambda:{'used':4}),startup_cache=SimpleNamespace(snapshot=lambda:{'used':2}))
        before=dict(vars(runtime));snapshot=rank_snapshot(runtime)
        self.assertEqual(vars(runtime),before)
        self.assertEqual(snapshot['calls'],3)
        self.assertEqual(snapshot['operators']['refiner_attention'],'native_sdpa')


if __name__=='__main__':unittest.main(verbosity=2)
