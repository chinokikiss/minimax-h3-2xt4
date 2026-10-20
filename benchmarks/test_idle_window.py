"""Actual CUDA lifetime/reuse tests; local results are never T4 performance."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import threading
import time
import tempfile
import unittest
import torch
from patches.host_weight_cache import HostBudget,HostWeightCache
from patches.weight_residency import ResidencyWindow,WeightSpec,module_bytes


def model(index):
    result=torch.nn.Linear(64,64,bias=False)
    with torch.no_grad():result.weight.copy_(torch.arange(4096).reshape(64,64).float()/10000+index*.01)
    return result.eval().requires_grad_(False)


class IdleWindow(unittest.TestCase):
    def setup_window(self,prepare=None,count=3):
        budget=HostBudget(16*2**20,reserve_bytes=0,pressure_aware=True)
        cache=HostWeightCache(budget,2**20,0,reuse_aware=True)
        size=module_bytes(model(0))
        specs=[WeightSpec(i,(lambda i=i:model(i)) if prepare is None else prepare,size,prepare_peak_bytes=2*size)
            for i in range(count)]
        window=ResidencyWindow(specs,(),'cuda:0',max_weight_bytes=4*size,pinned_bytes=4*size,
            reserve_bytes=0,host_cache=cache,lookahead=2)
        return budget,cache,window,size

    def test_disabled_observer_preserves_idle_cancellation_arguments(self):
        import causal_trace
        from dataclasses import replace
        budget,cache,window,size=self.setup_window(count=1)
        key=('fixture','te',0,0)
        window.specs={key:replace(window.specs[0],key=key)}
        try:
            with tempfile.TemporaryDirectory() as directory:
                recorder=causal_trace.Recorder(directory,0,'cuda:0')
                causal_trace.install(recorder)
                try:
                    window.begin_idle([key],max_bytes=size,reserve_bytes=0)
                    deadline=time.monotonic()+5
                    while not window.entries and time.monotonic()<deadline:window.idle_tick();time.sleep(.002)
                    self.assertEqual(window.stats['idle_errors'],0)
                    self.assertIn(key,window.entries)
                finally:causal_trace.restore()
        finally:window.close();cache.close()
        self.assertEqual(budget.snapshot()['reserved_bytes'],0)

    def test_reuses_the_same_gpu_storage_and_waits_for_copy(self):
        budget,cache,window,size=self.setup_window()
        try:
            window.begin_idle([0,1,2],max_bytes=2*size,reserve_bytes=0)
            with torch.cuda.stream(window.copy_stream):torch.cuda._sleep(2_000_000)
            deadline=time.monotonic()+10
            while len(window.entries)<2 and time.monotonic()<deadline:window.idle_tick();time.sleep(.002)
            self.assertEqual(len(window.entries),2)
            pointers={k:e.module.weight.data_ptr() for k,e in window.entries.items()}
            window.resume();window.extend_uses([0,1])
            value=torch.ones(2,64,device='cuda:0');expected=value.clone()
            for i in (0,1):expected=model(i).cuda()(expected)
            for i in (0,1):
                with window.use(i) as actual:
                    self.assertEqual(actual.weight.data_ptr(),pointers[i]);value=actual(value)
            torch.cuda.synchronize()
            self.assertTrue(torch.equal(value,expected))
            self.assertEqual(window.stats['idle_gpu_hits'],2)
            self.assertEqual(window.stats['idle_avoided_h2d_bytes'],2*size)
        finally:window.close();cache.close()
        state=budget.snapshot();self.assertEqual((state['reserved_bytes'],state['preparing'],state['pending_preparation_bytes']),(0,0,0))

    def test_cancel_does_not_wait_for_blocked_background_reader(self):
        entered=threading.Event();release=threading.Event()
        def prepare():entered.set();release.wait(10);return model(0)
        budget,cache,window,size=self.setup_window(prepare,count=1)
        try:
            window.begin_idle([0],max_bytes=size,reserve_bytes=0)
            window.idle_tick();self.assertTrue(entered.wait(5))
            begin=time.monotonic();window.cancel_idle(keep=False);window.close()
            self.assertLess(time.monotonic()-begin,1.)
            # Only the still-running producer is charged; abandoned pin space is gone.
            state=budget.snapshot();self.assertEqual(state['reserved_bytes'],2*size)
            self.assertEqual(state['pending_preparation_bytes'],2*size)
        finally:release.set();window.close();cache.close()
        self.assertEqual(budget.snapshot()['reserved_bytes'],0)

    def test_insufficient_idle_budget_falls_back_to_normal_staging(self):
        budget,cache,window,size=self.setup_window(count=1)
        try:
            window.begin_idle([0],max_bytes=size-1,reserve_bytes=0)
            self.assertFalse(window.idle_tick());self.assertFalse(window.pending)
            window.resume();window.extend_uses([0])
            value=torch.ones(1,64,device='cuda:0')
            with window.use(0) as actual:output=actual(value)
            self.assertTrue(torch.equal(output,model(0).cuda()(value)))
            self.assertEqual(window.stats['idle_loads'],0)
        finally:window.close();cache.close()
        self.assertEqual(budget.snapshot()['reserved_bytes'],0)

    def test_cancelled_idle_plan_can_restart_but_active_invocation_cannot(self):
        budget,cache,window,size=self.setup_window(count=1)
        try:
            window.begin_idle([0],max_bytes=size,reserve_bytes=0)
            window.cancel_idle(keep=False)
            window.begin_idle([0],max_bytes=size,reserve_bytes=0)
            deadline=time.monotonic()+5
            while not window.entries and time.monotonic()<deadline:window.idle_tick();time.sleep(.002)
            self.assertIn(0,window.entries)
            window.resume();window.extend_uses([0])
            with self.assertRaises(RuntimeError):window.begin_idle([0],max_bytes=size,reserve_bytes=0)
            with window.use(0):
                with self.assertRaises(RuntimeError):window.begin_idle([0],max_bytes=size,reserve_bytes=0)
        finally:window.close();cache.close()
        self.assertEqual(budget.snapshot()['reserved_bytes'],0)


if __name__=='__main__':
    torch.set_grad_enabled(False);torch.backends.cuda.matmul.allow_tf32=False
    unittest.main()
