"""Analytical gauge/folding boundaries, independent of a checkpoint or GPU."""
from pathlib import Path
import sys, unittest
sys.path[:0]=[str(Path(__file__).resolve().parent)]
import torch
from torch import nn
from fp16_residual_fix import Gauge,rms_from_scaled,folded_rows,boosted_norm_gain,fold_rms_gain_int8


class GaugeContracts(unittest.TestCase):
    def test_epsilon_scaling_is_required(self):
        x=torch.tensor([[1.e-5,2.e-5,3.e-5,4.e-5]])
        gamma=torch.tensor([1.,2.,.5,-1.]);eps=1.e-6;c=16.
        reference=torch.nn.functional.rms_norm(x,(4,),gamma,eps)
        actual=rms_from_scaled(x/c,gamma,eps,c)
        torch.testing.assert_close(actual,reference,rtol=1e-6,atol=1e-8)
        wrong=torch.nn.functional.rms_norm(x/c,(4,),gamma,eps)
        self.assertGreater(float((wrong-reference).abs().max()),.01)

    def test_layernorm_affine_beta_is_unchanged(self):
        x=torch.tensor([[1.e-5,2.e-5,3.e-5,4.e-5]],dtype=torch.float64)
        gamma=torch.tensor([1.,2.,.5,-1.],dtype=x.dtype);beta=torch.tensor([.1,.2,.3,.4],dtype=x.dtype)
        c=64.;eps=1.e-6
        ref=torch.nn.functional.layer_norm(x,(4,),gamma,beta,eps)
        actual=torch.nn.functional.layer_norm(x/c,(4,),gamma,beta,eps/c**2)
        torch.testing.assert_close(actual,ref,rtol=1e-14,atol=1e-14)

    def test_fp32_epsilon_island(self):
        eps=1.e-5/256**2
        self.assertEqual(float(torch.tensor(eps,dtype=torch.float16)),0.)
        y=torch.zeros(2,4,dtype=torch.float16)
        self.assertTrue(torch.isfinite(rms_from_scaled(y,torch.ones(4),1.e-5,256.)).all())

    def test_overflow_and_underflow_are_opposing_constraints(self):
        x=torch.tensor([100000.])
        self.assertFalse(torch.isfinite(x.half()).all())
        self.assertTrue(torch.isfinite((x/4).half()).all())
        tiny=torch.tensor([2.**-23])
        self.assertNotEqual(float(tiny.half()),0.)
        self.assertEqual(float((tiny/16).half()),0.)

    def test_power_of_two_does_not_fix_update_absorption(self):
        for c in (1.,4.,16.,256.):
            self.assertEqual(float((torch.tensor(1000./c)+.1/c).half()*c),1000.)

    def test_configuration_rejects_invalid_gauges(self):
        for exponent in (-1,17,1.5,True):
            with self.assertRaises(ValueError):Gauge(exponent)
        with self.assertRaises(ValueError):Gauge(storage_dtype=torch.bfloat16)

    def test_fold_restores_scales_and_lora_on_exception(self):
        block=nn.Module()
        for name in ('out_proj','fc2'):
            module=nn.Module();module.weight=nn.Parameter(torch.tensor([[1,-2]],dtype=torch.int8),requires_grad=False)
            module.weight_scale=nn.Parameter(torch.tensor([[.25]]),requires_grad=False)
            adapter=nn.Module();term=nn.Module();term.scale=-.2;adapter.terms=nn.ModuleList([term]);module.lora_adapter=adapter
            setattr(block,name,module)
        scales=[getattr(block,n).weight_scale for n in ('out_proj','fc2')]
        with self.assertRaisesRegex(RuntimeError,'probe'):
            with folded_rows(block,16.):
                self.assertEqual(float(block.out_proj.weight_scale),.25/16)
                self.assertEqual(block.fc2.lora_adapter.terms[0].scale,-.2/16)
                self.assertEqual(block.fc2.weight.tolist(),[[1,-2]])
                raise RuntimeError('probe')
        self.assertIs(block.out_proj.weight_scale,scales[0]);self.assertIs(block.fc2.weight_scale,scales[1])
        self.assertEqual(block.fc2.lora_adapter.terms[0].scale,-.2)

    def test_dynamic_adaln_gain_fold_rejected(self):
        with self.assertRaisesRegex(ValueError,'AdaLN'):
            fold_rms_gain_int8(torch.ones(1,256,dtype=torch.int8),torch.ones(1,1),torch.ones(256),modulated=True)

    def test_scalar_gain_fold_includes_dynamic_shifts_and_restores(self):
        torch.manual_seed(881)
        block=nn.Module();block.hidden=4
        block.norm1=nn.RMSNorm(4,eps=1e-5);block.norm2=nn.RMSNorm(4,eps=1e-5)
        block.adaln_proj=nn.Linear(8,3*6*4)
        for name in ('qkv_proj','fc1'):
            module=nn.Module();module.weight_scale=nn.Parameter(torch.ones(4,1))
            setattr(block,name,module)
        x=torch.randn(7,4);t=torch.randn(2,8)
        original=block.adaln_proj(t).view(-1,6,4)
        baseline=block.norm1(x)*(1+original[0,1])+original[0,0]
        parameters={name:value for name,value in block.named_parameters()}
        with boosted_norm_gain(block,6):
            mod=block.adaln_proj(t).view(-1,6,4)
            for index in (1,2,4,5):torch.testing.assert_close(mod[:,index],original[:,index],rtol=0,atol=0)
            actual=(block.norm1(x)*(1+mod[0,1])+mod[0,0])/64
            torch.testing.assert_close(actual,baseline,rtol=0,atol=0)
            self.assertEqual(float(block.fc1.weight_scale[0]),1/64)
        for name,value in block.named_parameters():self.assertIs(value,parameters[name])

    def test_adaln_adapter_gain_fold_is_not_silently_accepted(self):
        block=nn.Module();block.adaln_proj=nn.Sequential(nn.Linear(2,2))
        with self.assertRaisesRegex(ValueError,'AdaLN adapters'):
            with boosted_norm_gain(block,6):pass


if __name__=='__main__':unittest.main()
