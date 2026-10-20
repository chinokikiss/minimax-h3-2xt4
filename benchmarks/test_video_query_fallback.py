"""Compatibility gate regressions; fake weights exercise policy, not GPU math."""
from pathlib import Path
import sys
import os
from types import SimpleNamespace
import unittest
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),os.environ.get('COMFY_ROOT',str(ROOT/'ComfyUI'))]
from patches.parallel_vae import query_partitioning_status,query_partitioning_available


class QueryGate(unittest.TestCase):
    def setUp(self):
        import comfy.ops
        import comfy.ldm.minimax.vae as native
        import comfy.ldm.modules.attention as attention
        class Quantized:pass
        self.type=Quantized
        self.patches=[patch.object(comfy.ops,'QuantizedTensor',Quantized),
            patch.object(native,'COMFY_KITCHEN_INT8_ATTENTION_IS_AVAILABLE',True,create=True),
            patch.object(attention,'COMFY_KITCHEN_INT8_ATTENTION_IS_AVAILABLE',True)]
        for p in self.patches:p.start();self.addCleanup(p.stop)
        def legacy(x,rotary_pos_emb,pre_norm,residual,residual_scale):pass
        self.attn=SimpleNamespace(to_qkv=SimpleNamespace(weight=Quantized()),forward=legacy,qk_norm_scale=object())
        self.model=SimpleNamespace(decoder=SimpleNamespace(transformer_blocks=[SimpleNamespace(attn=self.attn)]))

    def test_qualified_legacy_contract_still_passes(self):
        result=query_partitioning_status(self.model)
        self.assertTrue(result['available']);self.assertIsNone(result['reason'])
        self.assertTrue(query_partitioning_available(self.model))

    def test_missing_vae_flag_is_not_kitchen_build_unavailability(self):
        import comfy.ldm.minimax.vae as native
        native.COMFY_KITCHEN_INT8_ATTENTION_IS_AVAILABLE=None
        result=query_partitioning_status(self.model)
        self.assertFalse(result['available']);self.assertTrue(result['kitchen_build_available'])
        self.assertIsNone(result['native_vae_kitchen_flag'])
        self.assertEqual(result['unquantized_blocks'],[])

    def test_simple_attention_signature_fails_even_if_flag_forced_true(self):
        self.attn.forward=lambda x,rotary_pos_emb=None:None
        result=query_partitioning_status(self.model)
        self.assertEqual(result['signature_mismatch_blocks'],[0]);self.assertFalse(result['available'])

    def test_missing_fused_norm_scale_fails_before_parallel_dispatch(self):
        del self.attn.qk_norm_scale
        result=query_partitioning_status(self.model)
        self.assertEqual(result['missing_qk_norm_scale_blocks'],[0]);self.assertFalse(result['available'])

    def test_dense_weight_and_empty_decoder_rejected(self):
        self.attn.to_qkv.weight=object()
        self.assertEqual(query_partitioning_status(self.model)['unquantized_blocks'],[0])
        self.model.decoder.transformer_blocks=[]
        self.assertFalse(query_partitioning_available(self.model))

    def test_uninspectable_forward_falls_back(self):
        self.attn.forward=object()
        result=query_partitioning_status(self.model)
        self.assertFalse(result['available']);self.assertEqual(result['signature_mismatch_blocks'],[0])


if __name__=='__main__':unittest.main(verbosity=2)
