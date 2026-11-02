"""Production control gates for the T4 P0-2/P0-3 lifetime promotion.

These gates verify configuration and ownership selection, not GPU peak or E2E
performance. The historical V3 T4 result is a separately frozen experiment.
"""
from dataclasses import replace
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn

from patches.h3_operator_policy import H3OperatorPolicy, configure_block


def block_shell():
    return SimpleNamespace(
        qkv_proj=nn.Identity(),
        fc1=nn.Identity(),
        out_proj=nn.Identity(),
        fc2=nn.Identity(),
    )


class P0ProductionDefaults(unittest.TestCase):
    def test_policy_defaults_enable_both_lifetimes_without_p01(self):
        p = H3OperatorPolicy()
        self.assertTrue(p.dit_p0_2_lifetime)
        self.assertTrue(p.dit_p0_3_lifetime)
        self.assertTrue(p.dit_p0_b_inplace)
        self.assertTrue(p.dit_p0_a1_turing)
        self.assertTrue(p.dit_p0_c_local_rotation)
        self.assertEqual(p.fc2_quantizer, 'fused_stored')
        self.assertEqual(p.row_linear_backend, 'tilelang_scaled_fp16')
        self.assertTrue(p.dit_fused_row_gate)
        block = configure_block(block_shell(), 'dit', p)
        self.assertTrue(block.out_proj._vram_p0_2)
        self.assertTrue(block.fc2._vram_p0_3)
        self.assertTrue(block.fc2._vram_p0_b)
        self.assertTrue(block.fc1._vram_p0_3_producer)
        self.assertTrue(block.fc1._vram_p0_a1)
        self.assertTrue(block.qkv_proj._vram_p0_a1)
        self.assertTrue(block.out_proj._vram_p0_c)
        self.assertFalse(hasattr(block.qkv_proj, '_vram_p0_1'))
        self.assertFalse(hasattr(block.qkv_proj, '_vram_p0_3_producer'))
        self.assertFalse(hasattr(block.fc1, '_vram_p0_1'))

    def test_explicit_opt_out_and_independent_controls(self):
        p = H3OperatorPolicy()
        for p02, p03 in ((False, False), (False, True), (True, False)):
            with self.subTest(p02=p02, p03=p03):
                b = configure_block(block_shell(), 'dit',
                    replace(p, dit_p0_2_lifetime=p02, dit_p0_3_lifetime=p03))
                self.assertEqual(b.out_proj._vram_p0_2, p02)
                self.assertEqual(b.fc2._vram_p0_3, p03)
                self.assertEqual(b.fc1._vram_p0_3_producer, p03)

    def test_ac_independent_opt_out_does_not_disable_existing_lifetimes(self):
        p = replace(H3OperatorPolicy(),dit_p0_a1_turing=False,dit_p0_c_local_rotation=False)
        b = configure_block(block_shell(),'dit',p)
        self.assertFalse(b.qkv_proj._vram_p0_a1)
        self.assertFalse(b.fc1._vram_p0_a1)
        self.assertFalse(b.out_proj._vram_p0_c)
        self.assertTrue(b.out_proj._vram_p0_2)
        self.assertTrue(b.fc2._vram_p0_3)

    def test_te_is_unchanged(self):
        p = H3OperatorPolicy()
        te = SimpleNamespace(o_proj=nn.Identity(), down_proj=nn.Identity())
        configure_block(te, 'te', p)
        self.assertFalse(hasattr(te.o_proj, '_vram_p0_2'))
        self.assertFalse(hasattr(te.down_proj, '_vram_p0_3'))
        self.assertFalse(hasattr(te.down_proj, '_vram_p0_b'))

    def test_strict_boolean_validation(self):
        for field in ('dit_p0_2_lifetime', 'dit_p0_3_lifetime',
                      'dit_p0_a1_turing','dit_p0_c_local_rotation','dit_p0_b_inplace'):
            with self.subTest(field=field):
                with self.assertRaises(ValueError):
                    H3OperatorPolicy(**{field: 1})

    def test_p02_fused_o_projection_consumes_owned_input(self):
        from patches import tilelang_dit_fusion as fusion
        from patches import tilelang_transport_p2 as transport
        p = H3OperatorPolicy()
        module = SimpleNamespace(operator_policy=p, lora_adapter=None,
            _vram_p0_2=True, _vram_p0_c=True, world_size=2, convrot=True, convrot_groupsize=256,
            weight=torch.zeros((5376, 3584), dtype=torch.int8),
            weight_scale=torch.ones((5376, 1), dtype=torch.float32))
        value = torch.ones((2, 3584), dtype=torch.float16)
        residual = torch.zeros((2, 5376), dtype=torch.float16)
        gate = torch.ones((1, 5376), dtype=torch.float16)
        observed = {}
        def fake_row(x, *args, consume_input, finish_fn, p0_c):
            observed['dtype'] = x.dtype
            observed['consume'] = consume_input
            observed['copy_size'] = x.numel()
            observed['p0_c'] = p0_c
            return residual
        with torch.no_grad(), patch.object(fusion, 'supported', return_value=True), \
             patch.object(fusion, '_table', return_value=True), \
             patch.object(fusion, 'labels_for', return_value=torch.zeros(2, dtype=torch.int32)), \
             patch.object(transport, 'row_linear', side_effect=fake_row):
            result = fusion.row_gated(module, value, residual, gate, None, object())
        self.assertIs(result, residual)
        self.assertEqual(observed, dict(dtype=torch.float32, consume=True, p0_c=True, copy_size=7168))
        self.assertEqual(value.numel(), 0)

    def test_p03_fused_fc2_flag_and_online_lora_guard(self):
        from patches import tilelang_dit_fusion as fusion
        from patches import tilelang_swiglu_quant as swiglu
        p = H3OperatorPolicy()
        module = SimpleNamespace(operator_policy=p, lora_adapter=None,
            _vram_p0_3=True, _vram_p0_b=True, world_size=2, convrot=True, convrot_groupsize=256,
            weight=torch.zeros((5376, 7168), dtype=torch.int8),
            weight_scale=torch.ones((5376, 1), dtype=torch.float32))
        value = torch.ones((2, 14336), dtype=torch.float16)
        residual = torch.zeros((2, 5376), dtype=torch.float16)
        gate = torch.ones((1, 5376), dtype=torch.float16)
        observed = {}
        def fake_row(x, *args, consume_input, mode, scaled_transport, finish_fn, p0_b):
            observed['p0_b'] = p0_b
            observed['consume'] = consume_input
            observed['mode'] = mode
            observed['transport'] = scaled_transport
            return residual
        with torch.no_grad(), patch.object(fusion, 'supported', return_value=True), \
             patch.object(fusion, '_table', return_value=True), \
             patch.object(fusion, 'labels_for', return_value=torch.zeros(2, dtype=torch.int32)), \
             patch.object(swiglu, 'row_linear', side_effect=fake_row):
            self.assertIs(fusion.row_gated(module, value, residual, gate, None, object(), 'swiglu'), residual)
            module.lora_adapter = object()
            self.assertIsNone(fusion.row_gated(module, value, residual, gate, None, object(), 'swiglu'))
        self.assertEqual(observed, dict(consume=True, mode='fused_stored', transport=True, p0_b=True))
        self.assertEqual(value.numel(), 2 * 14336)

    def test_p0b_independent_opt_out_preserves_p03(self):
        p=replace(H3OperatorPolicy(),dit_p0_b_inplace=False)
        b=configure_block(block_shell(),'dit',p)
        self.assertTrue(b.fc2._vram_p0_3)
        self.assertFalse(b.fc2._vram_p0_b)
        from patches.tilelang_swiglu_inplace import prepare
        source=torch.ones((2,14336),dtype=torch.float16)
        self.assertIsNone(prepare(source,object(),mode='fused_stored',arch='sm_75',consume_input=True))
        self.assertEqual(source.numel(),2*14336)

    def test_comfy_loader_defaults_and_manual_opt_out(self):
        from custom_nodes.minimax_h3_tp2 import MiniMaxH3TP2Loader
        import inspect
        loader = MiniMaxH3TP2Loader
        params = inspect.signature(loader.load).parameters
        self.assertTrue(params['dit_p0_2_lifetime'].default)
        self.assertTrue(params['dit_p0_3_lifetime'].default)
        self.assertTrue(params['dit_p0_b_inplace'].default)
        self.assertTrue(params['dit_p0_a1_turing'].default)
        self.assertTrue(params['dit_p0_c_local_rotation'].default)
        node_inputs = loader.INPUT_TYPES()['optional']
        self.assertTrue(node_inputs['dit_p0_2_lifetime'][1]['default'])
        self.assertTrue(node_inputs['dit_p0_3_lifetime'][1]['default'])
        self.assertTrue(node_inputs['dit_p0_b_inplace'][1]['default'])
        self.assertTrue(node_inputs['dit_p0_a1_turing'][1]['default'])
        self.assertTrue(node_inputs['dit_p0_c_local_rotation'][1]['default'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
