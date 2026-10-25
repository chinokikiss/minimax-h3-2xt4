"""Independent low-rank FP32 SUM arithmetic and real factor-shard selection."""
from dataclasses import replace
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
import torch
from torch.nn import functional as F
from patches.h3_lora import RowPartialLoRA,H3LoRAStore


class PartialReduction(unittest.TestCase):
    def test_both_owners_stacked_adapters_and_swiglu_use_only_fp32_hidden_sum(self):
        torch.manual_seed(29);m,k,n=7,32,24
        packed=[torch.randn(m,2*k) for _ in range(2)]
        terms=[(torch.randn(r,2*k),torch.randn(n,r),s) for r,s in ((3,.7),(5,-.25))]
        activated=[F.silu(x[:,:k])*x[:,k:] for x in packed]
        full=torch.cat(activated,-1);base=torch.randn(m,n)
        reduced=[F.linear(activated[0],a[:,:k])+F.linear(activated[1],a[:,k:]) for a,_,_ in terms]
        expected=base.clone();native=base.clone()
        for hidden,(a,b,s) in zip(reduced,terms):
            expected.add_(F.linear(hidden,b).mul_(s));native.add_(F.linear(F.linear(full,a),b).mul_(s))
        for rank in (0,1):
            events=[]
            class Comm:
                ReduceOp=SimpleNamespace(SUM='sum')
                def get_world_size(self):return 2
                def all_reduce(self,value,op):
                    index=len(events);events.append((value.dtype,tuple(value.shape),op))
                    peer=F.linear(activated[1-rank],terms[index][0][:,(1-rank)*k:(2-rank)*k])
                    value.add_(peer)
            adapter=RowPartialLoRA([(a[:,rank*k:(rank+1)*k],b,s) for a,b,s in terms],(rank*k,(rank+1)*k))
            adapter.fp32_compute=True
            output=adapter.add_to(packed[rank],base.clone(),Comm(),'swiglu')
            self.assertTrue(torch.equal(output,expected))
            torch.testing.assert_close(output,native,rtol=1e-5,atol=3e-5)
            self.assertEqual(events,[(torch.float32,(m,r),'sum') for r in (3,5)])

    def test_store_reads_owned_a_columns_and_keeps_full_b(self):
        a=torch.randn(3,32);b=torch.randn(24,3);reads=[]
        class Reader:
            header={'A':{'shape':[3,32]},'B':{'shape':[24,3]}}
            def read_tensor(self,key,rows=None,columns=None):
                reads.append((key,rows,columns));value=a if key=='A' else b
                return value[:,columns[0]:columns[1]] if columns is not None else value
        store=object.__new__(H3LoRAStore);store.row_partial=True;store.selective_b=False
        store.sources=[(Reader(),1.,{'blocks.0.mlp.fc2':('A','B',.5)})];store.check=lambda:None
        for rank in (0,1):
            adapter=store.adapter('blocks.0.mlp.fc2',rank)
            self.assertIsInstance(adapter,RowPartialLoRA)
            self.assertEqual(adapter.terms[0].down.shape,(3,16));self.assertEqual(adapter.terms[0].up.shape,(24,3))
            self.assertEqual(reads[-2:], [('A',None,(rank*16,(rank+1)*16)),('B',None,None)])
        store.row_partial=False;adapter=store.adapter('blocks.0.mlp.fc2',0)
        self.assertEqual(adapter.terms[0].down.shape,(3,32))

    def test_runtime_default_and_explicit_native_gather_have_distinct_identity(self):
        sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'ComfyUI'))
        from patches.comfy_tp_runtime import RuntimeConfig
        config=RuntimeConfig('te','dit')
        self.assertTrue(config.lora_row_partial_reduce)
        self.assertNotEqual(config,replace(config,lora_row_partial_reduce=False))


if __name__=='__main__':unittest.main()
