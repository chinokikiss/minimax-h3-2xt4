"""Arithmetic and last-consumer ownership gates; no checkpoint or GPU needed."""
from contextlib import nullcontext
import unittest
from unittest.mock import patch
import weakref
import torch
from torch import nn
import comfy_kitchen as ck
from comfy_kitchen.backends.eager import quantization as eager
from patches.dit_column_linear import column_linear


class DiTTemporaries(unittest.TestCase):
    @torch.no_grad()
    def test_column_matches_pinned_eager_and_preserves_input(self):
        torch.manual_seed(1023)
        for dtype in (torch.float16, torch.bfloat16, torch.float32):
            for rotate in (False, True):
                for per_channel in (False, True):
                    x=torch.randn(2,33,64,dtype=dtype)
                    before=x.clone();w=torch.randint(-127,128,(96,64),dtype=torch.int8)
                    scale=torch.rand(96,1)*.003 if per_channel else torch.tensor(.002)
                    with ck.registry.use_backend('eager'):
                        expected=ck.int8_linear(x,w,scale,None,dtype,convrot=rotate,convrot_groupsize=16)
                        actual=column_linear(x,w,scale,out_dtype=dtype,convrot=rotate,convrot_groupsize=16)
                    self.assertTrue(torch.equal(actual,expected),(dtype,rotate,per_channel))
                    self.assertTrue(torch.equal(x,before))

    @torch.no_grad()
    def test_column_releases_rotation_codes_and_accumulator_at_last_consumer(self):
        refs={};rotate=eager._rotate_activation;quant=eager.quantize_int8_rowwise
        mm=eager._int8_matmul_accumulate
        def rotated(*a,**kw):
            value=rotate(*a,**kw);refs['rotation']=weakref.ref(value);return value
        def quantized(*a,**kw):
            q,s=quant(*a,**kw);refs['codes']=weakref.ref(q);return q,s
        def multiply(*a,**kw):
            self.assertIsNone(refs['rotation']())
            value=mm(*a,**kw);refs['accumulator']=weakref.ref(value);return value
        with ck.registry.use_backend('eager'),patch.object(eager,'_rotate_activation',rotated), \
                patch.object(eager,'quantize_int8_rowwise',quantized), \
                patch.object(eager,'_int8_matmul_accumulate',multiply):
            column_linear(torch.randn(33,64),torch.ones(96,64,dtype=torch.int8),
                          torch.tensor(.002),convrot_groupsize=16)
        for name in ('rotation','codes','accumulator'):self.assertIsNone(refs[name]())

    def test_grad_compiled_and_other_backend_calls_keep_public_operator(self):
        x=torch.ones(2,64);w=torch.ones(32,64,dtype=torch.int8);s=torch.tensor(.01)
        with patch.object(ck,'int8_linear',return_value=x) as public:
            self.assertIs(column_linear(x,w,s),x)
            self.assertEqual(public.call_count,1)
        with torch.no_grad(),patch('torch.compiler.is_compiling',return_value=True), \
                patch.object(ck,'int8_linear',return_value=x) as public:
            self.assertIs(column_linear(x,w,s),x)
            self.assertEqual(public.call_count,1)
        with torch.no_grad(),patch.object(ck.registry,'get_implementation',return_value=lambda:None), \
                patch.object(ck,'int8_linear',return_value=x) as public:
            self.assertIs(column_linear(x,w,s),x)
            self.assertEqual(public.call_count,1)

    @torch.no_grad()
    def test_block_releases_attention_before_O_and_FC1_before_residual(self):
        from patches.tp_dit import TPDiTBlockINT8
        block=TPDiTBlockINT8(hidden=32,heads=2,head_dim=16,ffn=32,t_dim=8,
                            world_size=1,device='cpu')
        refs={};gates=[]
        class Projection(nn.Module):
            def __init__(self,name,width):super().__init__();self.name=name;self.width=width
            def forward(self,x,**kwargs):
                if self.name=='O':
                    for name in ('h','qkv','q','k','v'):self_test.assertIsNone(refs[name](),name)
                out=x.new_ones((x.shape[0],self.width))
                refs[self.name]=weakref.ref(out)
                return out
        self_test=self
        block.qkv_proj=Projection('qkv',96);block.out_proj=Projection('O',32)
        block.fc1=Projection('fc1',64);block.fc2=Projection('fc2',32)
        def norm_hook(module,args,out):refs['h']=weakref.ref(out)
        block.norm1.register_forward_hook(norm_hook)
        def attention(block,q,k,v,options):
            refs.update(q=weakref.ref(q),k=weakref.ref(k),v=weakref.ref(v))
            return torch.ones(1,q.shape[-2],32)
        original=block._mod_gate
        def gated(x,gate,branch,segments):
            if gates:self.assertIsNone(refs['fc1']())
            gates.append(True)
            return original(x,gate,branch,segments)
        block._mod_gate=gated
        with patch('patches.h3_operator_policy.dit_attention',attention):
            out=block(torch.ones(7,32),torch.ones(3,8),[(0,7,0)])
        self.assertEqual(out.shape,(7,32));self.assertEqual(len(gates),2)

    @torch.no_grad()
    def test_attention_O_quantizer_releases_packing_input_before_rotation(self):
        from patches import tilelang_row_quant as row
        refs={}
        def pack(rows,width,world,rank):
            def run(flat):
                refs['flat']=weakref.ref(flat)
                padded=torch.zeros(rows,width*world);refs['padded']=weakref.ref(padded)
                return padded
            return run
        def rotate(padded,h,groupsize):
            self.assertIsNone(refs['flat']())
            return padded.clone()
        def maximum(*args):
            def run(rotated):
                self.assertIsNone(refs['padded']())
                return rotated.abs().amax(-1,keepdim=True)
            return run
        class Comm:
            ReduceOp=torch.distributed.ReduceOp
            def get_rank(self):return 0
            def all_reduce(self,*a,**kw):pass
        x=torch.randn(33,128)[:,::2]  # Force a distinct contiguous packing input.
        with patch.object(row,'_device'),patch.object(row,'label',lambda *a:nullcontext()), \
                patch.object(row,'packing_kernel',pack),patch.object(row,'maximum_kernel',maximum), \
                patch.object(ck.backends.cuda,'_build_hadamard',return_value=torch.ones(16,16)), \
                patch.object(ck.backends.cuda,'_rotate_activation',rotate), \
                patch.object(row,'quantize',side_effect=lambda r,m,w,s:(r[:,s:s+w].to(torch.int8),m)):
            row.global_row_quantize(x,Comm(),2,groupsize=16,variant='packed')


if __name__=='__main__':unittest.main(verbosity=2)
