"""CPU-only guards for the long-run bridge; no H3 generation or GPU allocation."""
from dataclasses import asdict
from pathlib import Path
import sys
import unittest
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'ComfyUI'),str(ROOT/'benchmarks')]
import torch
from benchmarks.run_h3_l1_current_e2e import runtime_config,summaries,all_finite


class LongRunGuards(unittest.TestCase):
    def test_offline_checkpoint_retains_current_defaults(self):
        config=dict(offline_lora=True,width=896,height=512,paths=dict(te='unused_te',dit='unused_baked'))
        runtime=runtime_config(config)
        self.assertEqual(runtime.lora_stack,())
        self.assertEqual(runtime.tensor_flow.mode,'fp16')
        self.assertEqual(runtime.tensor_flow.divisor,64)
        operators=asdict(runtime.operators)
        self.assertEqual(operators['dit_attention'],'tilelang_scalar')
        self.assertEqual(operators['row_linear_backend'],'tilelang_scaled_fp16')
        self.assertEqual(operators['fc2_quantizer'],'fused_stored')
        with self.assertRaises(ValueError):runtime_config(config|dict(offline_lora=False))

    def test_nested_sampler_summaries_preserve_nonfinite_failure(self):
        from comfy.nested_tensor import NestedTensor
        value=NestedTensor([torch.ones(1,24,2,2,2,dtype=torch.float16),torch.zeros(1,32,2,2)])
        summary=summaries(value)
        self.assertTrue(all_finite(summary))
        self.assertEqual(summary[0]['dtype'],'torch.float16')
        value.tensors[1][0,0,0,0]=float('nan')
        self.assertFalse(all_finite(summaries(value)))


if __name__=='__main__':unittest.main()
