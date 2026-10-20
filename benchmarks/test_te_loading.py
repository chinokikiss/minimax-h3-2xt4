"""Integrity and concurrent-admission contracts for the TE loading candidates."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import tempfile
import threading
import time
import unittest
import torch
from safetensors.torch import save_file
from patches.te_loading import CoalescedTEReader,BoundedTECache
from patches.weight_residency import WeightSpec


class Budget:
    limit_bytes=10000
    reserve_bytes=0
    def __init__(self):self.used=0;self.lock=threading.Lock()
    def acquire(self,size):
        with self.lock:
            if self.used+size>self.limit_bytes:return False
            self.used+=size;return True
    def release(self,size):
        with self.lock:
            assert 0<=size<=self.used;self.used-=size


class Loading(unittest.TestCase):
    def test_byte_preservation_and_changed_artifact(self):
        with tempfile.TemporaryDirectory() as path:
            file=Path(path)/'te.block00.rank0.safetensors'
            state={'qkv':torch.arange(128,dtype=torch.int8).reshape(8,16),
                   'scale':torch.tensor([.125,.75]),'norm':torch.arange(16,dtype=torch.float16)}
            save_file(state,str(file));reader=CoalescedTEReader(file)
            for name in reversed(list(state)):self.assertTrue(torch.equal(reader.read_tensor(name),state[name]))
            self.assertEqual(reader.logical_read_bytes,sum(t.numel()*t.element_size() for t in state.values()))
            with file.open('ab') as stream:stream.write(b'x')
            with self.assertRaises(RuntimeError):reader.read_tensor('norm')

    def test_parallel_preparation_capacity_and_shutdown(self):
        budget=Budget();cache=BoundedTECache(budget,capacity_bytes=300,background_headroom_bytes=0)
        barrier=threading.Barrier(2);calls=[]
        def prepare(index):
            barrier.wait(timeout=5);calls.append(index)
            module=torch.nn.Module();module.register_buffer('w',torch.full((100,),index,dtype=torch.uint8))
            return module
        specs=[WeightSpec(('checkpoint','te',i,0),lambda i=i:prepare(i),100,prepare_peak_bytes=120) for i in range(2)]
        try:
            cache.prefetch(specs);cache.wait_idle(timeout=10)
            self.assertEqual(sorted(calls),[0,1]);self.assertLessEqual(cache._bytes(),cache.capacity_bytes)
            self.assertEqual(budget.used,200)
        finally:cache.close()
        self.assertEqual(budget.used,0)


if __name__=='__main__':unittest.main()
