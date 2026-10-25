"""CPU regression for the actual inference-tensor capture failure."""
import unittest
import torch
from benchmarks.run_t4_fp16_lora_diagnosis import physical_captures


class StageCapture(unittest.TestCase):
    def test_scaling_outside_inference_mode_does_not_mutate_inference_storage(self):
        with torch.inference_mode():
            captured={name:torch.tensor([2.,-3.]) for name in ('out_proj.output','fc2.output','fc1.output')}
        actual=physical_captures(captured,64.)
        for name in ('out_proj.output','fc2.output'):
            self.assertTrue(torch.equal(actual[name],torch.tensor([128.,-192.])))
            self.assertFalse(torch.is_inference(actual[name]))
            self.assertTrue(torch.equal(captured[name],torch.tensor([2.,-3.])))
        self.assertIs(actual['fc1.output'],captured['fc1.output'])


if __name__=='__main__':unittest.main()
