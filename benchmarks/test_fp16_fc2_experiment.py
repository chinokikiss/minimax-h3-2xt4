"""Requested storage/GEMM/collective/LoRA dtype boundaries, without CUDA."""
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import torch
from benchmarks import h3_fp16_fc2_experiment as exp


class FC2Contracts(unittest.TestCase):
    def test_original_int8_gemm_int32_sum_then_fp32_lora_and_fp16_store(self):
        q=torch.tensor([[1,-2,3,4]],dtype=torch.int8);w=torch.tensor([[2,3,-1,4]],dtype=torch.int8)
        x=torch.tensor([[1000.,-2.,3.,4.,1000.,2.,3.,4.]],dtype=torch.float16)
        scale=torch.tensor([[.25]],dtype=torch.float32);ws=torch.tensor([[.125]],dtype=torch.float32)
        calls=[]
        class Comm:
            ReduceOp=SimpleNamespace(SUM='sum')
            def all_reduce(self,value,op):
                self.assert_type=(value.dtype,op);calls.append(self.assert_type);value.mul_(2)
        comm=Comm()
        class Adapter:
            def add_to(self,inp,out,**kw):
                self.dtypes=(inp.dtype,out.dtype);return out+1.
        adapter=Adapter();module=SimpleNamespace(weight=w,weight_scale=ws,lora_adapter=adapter)
        def gemm(a,b):
            self.assertEqual((a.dtype,b.dtype),(torch.int8,torch.int8));return a.int()@b.int().t()
        with patch.object(exp,'prepare',return_value=(q,scale)),patch('patches.parallel_int8.int_mm',side_effect=gemm):
            result=exp.row_forward(module,x,comm)
        expected=(((q.int()@w.int().t())*2).float()*scale*ws+1.).half()
        self.assertTrue(torch.equal(result,expected));self.assertEqual(result.dtype,torch.float16)
        self.assertEqual(calls,[(torch.int32,'sum')]);self.assertEqual(adapter.dtypes,(torch.float32,torch.float32))

    def test_te_parts_stay_half_until_preparation_and_are_captured_before_swiglu(self):
        module=SimpleNamespace(world_size=2,convrot_groupsize=256,forward_parts=lambda a,b:None)
        gate=torch.tensor([[1000.]],dtype=torch.float16);up=gate.clone();captured=[]
        with patch.object(exp,'row_forward',side_effect=lambda module,x,*a:x):
            exp.install(module,object(),capture=lambda value:captured.append(value.clone()))
            result=module.forward_parts(gate,up)
        self.assertEqual(result.dtype,torch.float16);self.assertEqual(result.shape,(1,2))
        self.assertTrue(torch.equal(captured[0],torch.cat((gate,up),-1)))


if __name__=='__main__':unittest.main()
