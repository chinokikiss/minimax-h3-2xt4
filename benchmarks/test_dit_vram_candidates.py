"""CPU ownership, row partition and already-quantized chunk contracts."""
import gc
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import weakref
import torch

from benchmarks.dit_vram_candidates import Operand,prepare_column,consume_column,row_labels,VARIANTS,forward


class Contracts(unittest.TestCase):
    def test_block_policy_batch_axis_is_removed_before_row_projection_and_gate(self):
        rows,hidden=2,5376
        x=torch.zeros(rows,hidden,dtype=torch.float16)
        observed=[]
        def projection(value):
            observed.append(tuple(value.shape))
            self.assertEqual(tuple(value.shape),(rows,3584))
            return torch.zeros_like(x)
        block=SimpleNamespace(hidden=hidden,inner_per_rank=3584,
            adaln_proj=lambda _:torch.zeros(1,6*hidden),norm1=None,norm2=None,
            qkv_proj=lambda _:torch.zeros(rows,3*3584,dtype=torch.float16),
            q_norm=SimpleNamespace(weight=torch.ones(128),eps=1e-6),
            k_norm=SimpleNamespace(weight=torch.ones(128)),out_proj=projection,
            fc1=lambda _:torch.zeros(rows,16,dtype=torch.float16),fc2=lambda _,**kw:torch.zeros_like(x))
        def gate(value,gates,branch,labels):
            self.assertEqual(value.shape,branch.shape)
            return value
        with patch('benchmarks.dit_vram_kernels.norm',side_effect=lambda n,v,*args:v), \
                patch('benchmarks.dit_vram_kernels.gate',side_effect=gate), \
                patch('patches.tilelang_fp32_rms_rope.execute',return_value=(torch.zeros(rows,28,128),torch.zeros(rows,28,128))), \
                patch('patches.h3_operator_policy.dit_attention',return_value=torch.zeros(1,rows,3584,dtype=torch.float16)):
            result=forward(block,x,torch.zeros(1,8),[(0,rows,0)],None,torch.zeros(rows,dtype=torch.int32),'norm_gate')
        self.assertEqual(observed,[(rows,3584)])
        self.assertEqual(result.shape,x.shape)

    def test_prepare_does_not_retain_float_input_and_is_single_consumer(self):
        x=torch.randn(5,256);owner=weakref.ref(x)
        with patch('patches.dit_column_linear._prepare_activation',return_value=(torch.ones(5,256,dtype=torch.int8),torch.ones(5,1))):
            prepared=prepare_column(x)
        del x;gc.collect();self.assertIsNone(owner())
        q,s=prepared.take();self.assertEqual(q.dtype,torch.int8)
        self.assertIsNone(prepared.codes);self.assertIsNone(prepared.scale)
        with self.assertRaises(RuntimeError):prepared.take()

    def test_labels_preserve_disjoint_native_rows_and_reject_gaps(self):
        labels=row_labels(7,[(0,2,0),(2,5,2),(5,7,1)],'cpu')
        self.assertEqual(labels.tolist(),[0,0,2,2,2,1,1])
        for segments in ([(0,3,0)],[(0,5,0),(4,7,1)],[(0,7,3)]):
            with self.assertRaises(ValueError):row_labels(7,segments,'cpu')

    def test_chunking_consumes_same_codes_and_writes_final_destination(self):
        torch.manual_seed(29)
        q=torch.randint(-127,128,(19,256),dtype=torch.int8)
        w=torch.randint(-127,128,(128,256),dtype=torch.int8)
        a=torch.rand(19,1)*.001;scale=torch.rand(128,1)*.001
        module=SimpleNamespace(weight=w,weight_scale=scale)
        expected=(q.int()@w.int().T).float()*(a*scale.T)
        calls=[];destinations=[]
        def gemm(codes,weights):
            calls.append(tuple(codes.shape));return codes.int()@weights.int().T
        def kernel(rows,width,scalar):
            def store(codes,rowscale,weightscale,destination):
                destinations.append(destination.untyped_storage().data_ptr())
                destination.copy_(codes.float()*(rowscale*weightscale.reshape(1,-1)))
            return store
        with patch('patches.optimized_tp.native_int_mm',side_effect=gemm), \
                patch('benchmarks.dit_vram_kernels.column_store_kernel',side_effect=kernel):
            output=consume_column(Operand(q,a,(19,256)),module,8)
        self.assertTrue(torch.equal(output,expected.half()))
        self.assertEqual(calls,[(8,256),(8,256),(3,256)])
        self.assertEqual(len(set(destinations)),1)

    def test_production_source_is_not_imported_from_experiment(self):
        # Experimental kernels must remain outside runtime imports/defaults.
        root=Path(__file__).resolve().parents[1]
        for name in ('tp_dit','h3_tensor_flow','h3_operator_policy','dit_column_linear'):
            self.assertNotIn('dit_vram_candidates',(root/'patches'/f'{name}.py').read_text())
        self.assertEqual([VARIANTS[n]['chunk'] for n in ('fc1_4k','fc1_8k','fc1_16k')],[4096,8192,16384])

    def test_deployment_fixture_preserves_native_modality_tags_and_multiple_clocks(self):
        from benchmarks.run_dit_vram_case import make_inputs
        x,segments,labels,freqs=make_inputs(59134,'meta')
        self.assertEqual(segments,[(0,2392,7),(2392,3892,6),(3892,4702,5),(4702,59134,0)])
        self.assertEqual(tuple(x.shape),(59134,5376))
        self.assertEqual(tuple(freqs.shape),(1,59134,1,64,2,2))
        actual=row_labels(7,[(0,2,7),(2,5,6),(5,7,0)],'cpu',modulation_rows=9)
        self.assertEqual(actual.tolist(),[7,7,6,6,6,0,0])


if __name__=='__main__':unittest.main(verbosity=2)
