"""Lifecycle and real cgroup admission/accounting contracts."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import tempfile
import unittest
import torch
from patches.host_weight_cache import HostBudget, HostWeightCache, GiB
from patches.host_memory_pressure import CgroupMemory
from patches.weight_residency import WeightSpec


def spec(index):
    def prepare():
        result=torch.nn.Module();result.register_buffer('w',torch.full((64,),index,dtype=torch.uint8))
        return result
    return WeightSpec(('checkpoint','te',index,0),prepare,64,prepare_peak_bytes=128)


class Residency(unittest.TestCase):
    def test_prefetched_prefix_survives_startup_and_far_speculation(self):
        budget=HostBudget(1024*1024,reserve_bytes=0)
        cache=HostWeightCache(budget,128,0,reuse_aware=True)
        try:
            cache.prefetch([spec(0),spec(1)]);cache.wait_idle(5)
            cache.begin_request([spec(0),spec(1),spec(2)])
            self.assertEqual(list(cache.entries),[spec(0).key,spec(1).key])
            with cache.condition:
                self.assertFalse(cache._evict_one(spec(2).key,False))
            cache.relieve_pressure(64)
            self.assertIn(spec(0).key,cache.entries)
            self.assertNotIn(spec(1).key,cache.entries)
            self.assertEqual(cache.stats['evictions_before_use'],1)
        finally:cache.close()
        self.assertEqual(budget.snapshot()['reserved_bytes'],0)

    def test_cgroup_shmem_kernel_file_floor_and_shared_slots(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            (root/'memory.max').write_text(str(30*GiB))
            (root/'memory.current').write_text(str(29*GiB))
            (root/'memory.stat').write_text(f'anon {20*GiB}\nfile {8*GiB}\nshmem {GiB}\nkernel {GiB}\n')
            (root/'memory.pressure').write_text('full avg10=3.0 avg60=0 total=10\n')
            budget=HostBudget(24*GiB,reserve_bytes=2*GiB,pressure_aware=True)
            budget.memory=CgroupMemory(root)
            self.assertEqual(budget.pressure()['nonfile_bytes'],22*GiB)
            self.assertFalse(budget.acquire(GiB,speculative=True,preparation=True))
            self.assertTrue(budget.acquire(GiB,preparation=True))
            self.assertTrue(budget.acquire(GiB,preparation=True))
            self.assertFalse(budget.acquire(1,preparation=True))
            self.assertLessEqual(budget.speculation_depth(16,GiB),2)
            for _ in range(2):budget.finish_preparation(GiB);budget.release(GiB)
            self.assertEqual(budget.snapshot()['pending_preparation_bytes'],0)

    def test_preparation_failure_releases_shared_lease(self):
        budget=HostBudget(1024*1024,reserve_bytes=0,pressure_aware=True)
        cache=HostWeightCache(budget,128,0,reuse_aware=True)
        def fail():raise ValueError('intentional producer failure')
        item=WeightSpec('failure',fail,64,prepare_peak_bytes=128)
        try:
            with cache.condition:
                future=cache._queue(item,True);cache.condition.notify_all()
            with self.assertRaises(ValueError):future.result(5)
            cache.wait_idle(5)
        finally:cache.close()
        state=budget.snapshot()
        self.assertEqual((state['reserved_bytes'],state['preparing'],state['pending_preparation_bytes']),(0,0,0))


if __name__=='__main__':unittest.main()
