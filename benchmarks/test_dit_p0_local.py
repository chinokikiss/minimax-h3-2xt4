"""CPU/RTX local owner and allocator gates for P0-3 and P0-2.

Mocked kernel bodies preserve the producer/consumer allocations, but this is
NOT an SM75 kernel numerical/performance qualification. T4 is still mandatory.
"""
from contextlib import nullcontext
from types import SimpleNamespace
import gc
import unittest
from unittest.mock import patch
import torch

from patches import tilelang_swiglu_quant as swiglu
from patches import tilelang_row_quant as row_quant


class Comm:
    ReduceOp = SimpleNamespace(MAX="max")
    def __init__(self, rank=0):
        self.rank = rank
    def get_rank(self):
        return self.rank
    def all_reduce(self, tensor, op=None):
        return None


class LocalP0(unittest.TestCase):
    def test_p03_stored_kernel_releases_before_quantization(self):
        torch.manual_seed(17)
        owner = torch.randn(8, 512, dtype=torch.float16)
        def fake_fused(rows, width, dtype, arch, phase):
            self.assertEqual(phase, "stored")
            def run(flat):
                return flat[:, :width].float().clone(), torch.ones(rows, 1)
            return run
        quant_calls = [0]
        def fake_quant(rotated, maximum):
            quant_calls[0] += 1
            if quant_calls[0] == 2:
                self.assertEqual(owner.numel(), 0)
            return rotated.round().clamp(-127, 127).to(torch.int8), maximum
        with patch.object(swiglu, "fused_kernel", side_effect=fake_fused), \
             patch.object(row_quant, "quantize", side_effect=fake_quant):
            clone = owner.clone()
            baseline_q, baseline_scale = swiglu.prepare(clone, Comm(), "fused_stored")
            clone.set_()
            candidate_q, candidate_scale = swiglu.prepare(owner, Comm(), "fused_stored", consume_input=True)
        self.assertEqual(owner.numel(), 0)
        self.assertTrue(torch.equal(candidate_q, baseline_q))
        self.assertTrue(torch.equal(candidate_scale, baseline_scale))

    def test_p02_packed_input_release_before_rotation(self):
        torch.manual_seed(47)
        x = torch.randn(11, 256, dtype=torch.float32)
        fake_backend = SimpleNamespace(
            _apply_input_act=lambda value, *_: value,
            _build_hadamard=lambda *args, **kwargs: None,
            _rotate_activation=lambda padded, h, groupsize: padded.clone(),
        )
        fake_kitchen = SimpleNamespace(backends=SimpleNamespace(cuda=fake_backend))
        def padded_factory(rows, width, world_size, rank):
            def run(flat):
                return torch.cat([flat, torch.zeros_like(flat)], -1)
            return run
        def max_factory(rows, width, full, start):
            return lambda rotated: rotated[:, start:start+width].abs().amax(-1, keepdim=True)
        with patch.dict("sys.modules", {"comfy_kitchen": fake_kitchen}), \
             patch.object(row_quant, "_device", side_effect=lambda value: None), \
             patch.object(row_quant, "label", side_effect=lambda name: nullcontext()), \
             patch.object(row_quant, "packing_kernel", side_effect=padded_factory), \
             patch.object(row_quant, "maximum_kernel", side_effect=max_factory), \
             patch.object(row_quant, "quantize", side_effect=lambda rotated, maximum, width, start:
                          (rotated[:, start:start+width].round().to(torch.int8), maximum)):
            baseline_q, baseline_scale = row_quant.global_row_quantize(x.clone(), Comm(), 2, 256,
                                                                       variant="packed")
            candidate_q, candidate_scale = row_quant.global_row_quantize(x, Comm(), 2, 256,
                                                        variant="packed", consume_input=True)
        self.assertEqual(x.numel(), 0)
        self.assertTrue(torch.equal(candidate_q, baseline_q))
        self.assertTrue(torch.equal(candidate_scale, baseline_scale))

    def test_p03_column_producer_returns_true_storage_owner(self):
        from patches.tp_dit import ColumnParallelLinearINT8
        model=ColumnParallelLinearINT8(256,128,world_size=1,device=torch.device("cpu"))
        x=torch.randn(5,256,dtype=torch.float16)
        def projected(value,weights,scales,**kwargs):
            raw=torch.arange(5*128,dtype=torch.float16).reshape(5,128).clone()
            return raw.view(5,128)
        with patch("patches.tp_dit.run_int8_linear_forward",side_effect=projected),torch.no_grad():
            model._vram_p0_3_producer=False
            baseline=model(x)
            self.assertIsNotNone(baseline._base)
            model._vram_p0_3_producer=True
            candidate=model(x)
        self.assertIsNone(candidate._base)
        self.assertEqual(tuple(candidate.shape),(5,128))
        self.assertTrue(torch.equal(candidate,baseline))

    @unittest.skipUnless(torch.cuda.is_available(), "GPU allocator gate requires CUDA")
    def test_cuda_alias_and_attention_p02_owner(self):
        torch.cuda.synchronize()
        original = torch.empty((1024, 5376), device="cuda", dtype=torch.float16)
        view = original.view(-1, 5376)
        before = torch.cuda.memory_allocated()
        view.set_()
        self.assertEqual(before, torch.cuda.memory_allocated())
        original.set_()
        gc.collect()
        self.assertLess(torch.cuda.memory_allocated(), before)

        raw = torch.empty((1, 28, 1024, 128), device="cuda", dtype=torch.float16)
        attended = raw.transpose(1, 2).reshape(1, 1024, 3584)
        self.assertIsNone(attended._base)
        del raw
        pinned = attended.reshape(1024, 3584)
        self.assertIsNotNone(pinned._base)
        del attended
        before = torch.cuda.memory_allocated()
        pinned.set_()
        gc.collect()
        self.assertEqual(torch.cuda.memory_allocated(), before)
        del pinned
        gc.collect()

        raw = torch.empty((1, 28, 1024, 128), device="cuda", dtype=torch.float16)
        attended = raw.transpose(1, 2).reshape(1, 1024, 3584)
        del raw
        attended.resize_(1024, 3584)
        before = torch.cuda.memory_allocated()
        attended.set_()
        gc.collect()
        self.assertLess(torch.cuda.memory_allocated(), before)


if __name__ == "__main__":
    unittest.main(verbosity=2)
