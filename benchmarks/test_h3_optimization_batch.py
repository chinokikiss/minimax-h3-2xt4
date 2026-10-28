"""CPU contracts for scoped defaults, preparation and bounded disk baking."""
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import torch


class Operators(unittest.TestCase):
    def test_decoder_scope_restores_after_failure_and_does_not_enable_registry(self):
        import comfy_kitchen as ck
        from patches import h3_vae_operators as op
        before=ck.registry.list_backends()
        class Decoder(torch.nn.Module):
            def forward(self,x):
                self.seen=op._active.get()
                raise RuntimeError('expected decoder failure')
        model=SimpleNamespace(decoder=Decoder());op.install(model);op.install(model)
        with self.assertRaisesRegex(RuntimeError,'expected decoder'):model.decoder(None)
        self.assertTrue(model.decoder.seen);self.assertFalse(op._active.get())
        self.assertEqual(before,ck.registry.list_backends())
        self.assertEqual(len(model.decoder._forward_pre_hooks),1)

    def test_te_epilogue_is_default_only_for_qualified_columns(self):
        from patches.dit_column_linear import _dequantize_column
        with patch('torch.cuda.get_device_capability',return_value=(7,5)), \
             patch('patches.tilelang_column_epilogue.dequantize',return_value='tile') as fused, \
             patch('patches.dit_column_linear._dequantize_eager',return_value='eager'):
            x=SimpleNamespace(is_cuda=True,device='cuda:0')
            self.assertEqual(_dequantize_column(x,None,None,torch.float16,SimpleNamespace(shape=(12800,5120))),'tile')
            self.assertEqual(_dequantize_column(x,None,None,torch.float16,SimpleNamespace(shape=(14336,5376))),'eager')
            self.assertEqual(fused.call_count,1)

    def test_vae_linear_is_scoped_and_preserves_other_paths(self):
        from patches import h3_vae_operators as op
        x=SimpleNamespace(is_cuda=True,dtype=torch.float16,device='cuda:0',shape=(1797,2048),numel=lambda:1797*2048)
        w=SimpleNamespace(dtype=torch.int8,ndim=2,shape=(6144,2048),device='cuda:0')
        backend=SimpleNamespace(_EXT_AVAILABLE=True,int8_linear=lambda **kw:('cuda',kw['input_act']))
        original=lambda *a,**kw:'original'
        with torch.no_grad(),patch('torch.cuda.get_device_capability',return_value=(7,5)),patch.object(op,'_backend',return_value=backend):
            self.assertEqual(op.int8_linear(original,x,w,None,out_dtype=torch.float16,convrot=True,input_act='rms_norm'),'original')
            token=op._active.set(True)
            try:
                self.assertEqual(op.int8_linear(original,x,w,None,out_dtype=torch.float16,convrot=True,input_act='rms_norm'),('cuda','rms_norm'))
                self.assertEqual(op.int8_linear(original,x,w,None,out_dtype=torch.float32,convrot=True,input_act='rms_norm'),'original')
            finally:op._active.reset(token)

    def test_aot_plan_has_rank_specific_exact_shapes_and_no_tensor_execution(self):
        from patches.h3_operator_aot import plan
        from patches.h3_operator_policy import H3OperatorPolicy
        self.assertEqual(H3OperatorPolicy().dit_rms_rope,'tilelang_fp32')
        self.assertEqual(H3OperatorPolicy(dit_rms_rope='native').dit_rms_rope,'native')
        calls=plan(59134,1,H3OperatorPolicy(dit_rms_rope='tilelang_fp32'))
        self.assertEqual(len(calls),12)
        self.assertIn(('tilelang_row_quant','quantizer',(59134,3584,7168,3584,'fast')),calls)
        self.assertTrue(all(c[2][0]==59134 for c in calls))
        te=plan(2392,1,H3OperatorPolicy(),'te')
        self.assertEqual(len(te),12)
        self.assertIn(('tilelang_row_quant','quantizer',(2392,4096,8192,4096,'fast')),te)
        self.assertEqual(plan(1797,0,H3OperatorPolicy(),'video'),[])
        with self.assertRaises(ValueError):plan(0,1,H3OperatorPolicy())

    def test_fp32_rope_never_claims_cpu_support(self):
        from patches.tilelang_fp32_rms_rope import supported
        self.assertFalse(supported(torch.empty(1,10752),torch.empty(1,1,1,64,2,2),28,128))

    def test_fp32_rope_accepts_native_half_frequency_storage(self):
        from patches.tilelang_fp32_rms_rope import supported,execute
        packed=SimpleNamespace(is_cuda=True,ndim=2,dtype=torch.float16,device='cuda:0',
            shape=(33,10752),is_contiguous=lambda:True)
        table=torch.full((1,33,1,64,2,2),.125,dtype=torch.float16)
        freqs=SimpleNamespace(device='cuda:0',dtype=table.dtype,ndim=6,shape=table.shape,
            is_contiguous=lambda:True,reshape=table.reshape)
        with patch('torch.cuda.get_device_capability',return_value=(7,5)):
            self.assertTrue(supported(packed,freqs,28,128))
        gain=SimpleNamespace();gain.to=lambda **kw:gain;gain.reshape=lambda *a:gain;gain.contiguous=lambda:gain
        captured=[]
        with patch('patches.tilelang_fp32_rms_rope.supported',return_value=True), \
             patch('patches.tilelang_fp32_rms_rope.kernel',return_value=lambda *a:captured.append(a)):
            execute(packed,freqs,gain,gain)
        self.assertEqual(captured[0][1].dtype,torch.float16)
        self.assertTrue(torch.equal(captured[0][1],table.reshape(33,64,2,2)))

    def test_aot_dispatch_is_capability_gated_and_once_per_shape(self):
        from unittest.mock import Mock
        from patches.comfy_sampler import RemoteDenoiser
        from patches.h3_operator_policy import H3OperatorPolicy
        service=SimpleNamespace(config=SimpleNamespace(operators=H3OperatorPolicy()),call=Mock(),operator_aot=True)
        model=RemoteDenoiser(service);payload={'layout':SimpleNamespace(seq_len=59134)}
        model.prepare_payload(payload);model.prepare_payload(payload)
        service.call.assert_called_once_with('prepare_dit',59134)
        del service.operator_aot;service.call.reset_mock();model._aot_shapes.clear()
        model.prepare_payload(payload);service.call.assert_not_called()

    def test_aot_cache_keys_match_actual_runtime_consumers(self):
        from patches import tilelang_row_quant as row,tilelang_fp32_rms_rope as rope
        from patches.h3_operator_aot import plan
        from patches.h3_operator_policy import H3OperatorPolicy
        p=plan(59134,0,H3OperatorPolicy(dit_rms_rope='tilelang_fp32'))
        x=SimpleNamespace(is_cuda=True,dtype=torch.float32,device='cuda:0',ndim=2,
            shape=(59134,7168),is_contiguous=lambda:True)
        m=SimpleNamespace(shape=(59134,1),dtype=torch.float32,device='cuda:0')
        with patch('torch.cuda.get_device_capability',return_value=(7,5)),patch.object(row,'quantizer',return_value=lambda *a:None) as quantizer:
            row.quantize(x,m)
            self.assertIn(('tilelang_row_quant','quantizer',quantizer.call_args.args),p)
        gain=SimpleNamespace();gain.to=lambda **kw:gain;gain.reshape=lambda *a:gain;gain.contiguous=lambda:gain
        packed=SimpleNamespace(shape=(59134,10752),dtype=torch.float16,device='cuda:0')
        freqs=SimpleNamespace(shape=(1,59134,1,48,2,2),dtype=torch.float16,reshape=lambda *a:None)
        with patch.object(rope,'supported',return_value=True),patch.object(rope,'kernel',return_value=lambda *a:None) as kernel:
            rope.execute(packed,freqs,gain,gain)
            self.assertIn(('tilelang_fp32_rms_rope','kernel',kernel.call_args.args),p)


import benchmarks.test_tp2_disk_bake as bake_fixtures


class BakePipeline(bake_fixtures.DiskBake):
    def test_pipeline_matches_serial_all_shard_bytes_and_proofs(self):
        from patches.tp2_disk_bake import bake
        stack=((str(self.lora),.7),)
        paths=[bake(self.descriptor,stack,self.root/name,devices=('cpu',),pipeline=mode)
               for name,mode in (('serial',False),('pipelined',True))]
        a,b=[json.loads((p.parent/'bake_receipt.json').read_text()) for p in paths]
        self.assertEqual(a,b)

    def test_writer_failure_prevents_publication(self):
        from patches.tp2_disk_bake import bake
        with patch('patches.tp2_disk_bake._persist_layer',side_effect=OSError('writer failed')):
            with self.assertRaisesRegex(OSError,'writer failed'):
                bake(self.descriptor,((str(self.lora),.7),),self.root/'writer_failure',devices=('cpu',))
        self.assertFalse((self.root/'writer_failure').exists())


if __name__=='__main__':unittest.main()
