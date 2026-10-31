"""Promoted defaults, retired controls and saved-workflow migration; CPU only."""
from dataclasses import fields
import inspect
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from patches.comfy_tp_runtime import RuntimeConfig, GiB
from patches.h3_operator_policy import H3OperatorPolicy
from patches.h3_configuration import check_retired_loader_options
import custom_nodes.minimax_h3_tp2 as nodes

ROOT = Path(__file__).resolve().parents[1]


class ConfigurationDefaults(unittest.TestCase):
    def test_loader_policy_and_runtime_defaults_agree(self):
        schema = nodes.MiniMaxH3TP2Loader.INPUT_TYPES()
        inputs = schema['required'] | schema['optional']
        signature = inspect.signature(nodes.MiniMaxH3TP2Loader.load)
        for name, spec in inputs.items():
            if name not in ('text_encoder', 'diffusion_model', 'lora_stack'):
                self.assertEqual(signature.parameters[name].default, spec[1]['default'], name)
        config = RuntimeConfig('te', 'dit')
        self.assertEqual(config.operators, H3OperatorPolicy())
        for field in fields(config.operators):
            self.assertEqual(getattr(config.operators, field.name), inputs[field.name][1]['default'])
        self.assertTrue(config.adaptive_lookahead and config.prefetch_lifecycle and config.memory_pressure_aware)
        self.assertTrue(config.lora_selective_qkv_b)
        self.assertEqual(config.reserve_bytes, 9 * GiB)
        self.assertEqual(config.timeout_seconds, 1500)
        self.assertEqual(inputs['lora_mode'][1]['default'], 'offline')
        for name in ('lora_concat_a', 'tp_quantization', 'swiglu_reduction', 'checkpoint_read_chunk_bytes'):
            self.assertNotIn(name, {field.name for field in fields(config)})
        for name in ('swiglu_padding', 'selective_b', 'checkpoint_read_chunk_mib', 'tp_quantization'):
            self.assertNotIn(name, inputs)
        self.assertNotIn('tilelang_exact', inputs['column_linear_backend'][0])
        self.assertEqual(inputs['dit_attention'][0], ['kitchen', 'sol', 'native'])
        self.assertEqual(inputs['dit_attention_schedule'][1]['default'], '')
        self.assertEqual(inputs['sol_tau'][1]['default'], 1.0)
        self.assertEqual(inputs['video_attention'][0], ['kitchen', 'native'])
        self.assertEqual(inputs['video_attention'][1]['default'], 'kitchen')
        self.assertTrue(config.sol_topology_v2)
        self.assertTrue(config.sol_sync_free_lut)
        self.assertNotIn('scalar_score_workspace_mib', inputs)
        self.assertEqual(inputs['te_attention'][0], ['kitchen_causal', 'native'])

    def test_offline_bundle_keeps_adapters_and_online_is_explicit(self):
        with tempfile.TemporaryDirectory() as folder:
            adapter = Path(folder) / 'adapter.safetensors'
            adapter.write_bytes(b'fixture')
            stack = ((str(adapter), .7),)
            service = lambda config: SimpleNamespace(config=config, closed=False, close=Mock())
            with patch.object(nodes, '_service', None), patch.object(nodes, 'PhysicalTPService', side_effect=service), \
                    patch.object(nodes, 'create_model'), patch.object(nodes, 'create_clip'), \
                    patch('patches.offline_lora.cached_bake', side_effect=AssertionError('No full checkpoint bake for bundle')):
                loader = nodes.MiniMaxH3TP2Loader()
                offline = loader.load('te.json', 'dit.json', lora_stack=stack)[2]
                self.assertTrue(offline.config.bundle_offline_lora)
                self.assertEqual(offline.config.lora_stack, stack)
                online = loader.load('te.json', 'dit.json', lora_stack=stack, lora_mode='online')[2]
                self.assertFalse(online.config.bundle_offline_lora)
                self.assertTrue(online.config.lora_selective_qkv_b)
                self.assertEqual(online.config.lora_stack, stack)
                offline.close.assert_called_once()

    def test_removed_python_arguments_cannot_reenable_old_paths(self):
        check_retired_loader_options(dict(tp_quantization='scalemax', swiglu_reduction='native_int32', lora_concat_a=False))
        for name, value in [('tp_quantization', 'fixed'), ('swiglu_reduction', 'fp32'), ('lora_concat_a', True)]:
            with self.assertRaises(ValueError):
                check_retired_loader_options({name: value})
        with self.assertWarns(FutureWarning):
            check_retired_loader_options(dict(swiglu_padding=False, selective_b=False, checkpoint_read_chunk_mib=8))
        check_retired_loader_options(dict(scalar_score_workspace_mib=96))
        with self.assertRaises(TypeError):
            check_retired_loader_options(dict(misspelled_option=True))
        with self.assertRaises(TypeError):
            H3OperatorPolicy(swiglu_padding=False)
        with self.assertRaises(ValueError):
            H3OperatorPolicy(column_linear_backend='tilelang_exact')

    def test_int32_base_sum_and_fixed_scalemax_have_no_alternate_runtime_mode(self):
        import torch
        from patches import optimized_tp as opt, parallel_int8 as pi
        comm = SimpleNamespace(ReduceOp=SimpleNamespace(SUM='sum'))
        partial = torch.ones(3, 4, dtype=torch.int32)
        def reduce(value, op):
            self.assertEqual(value.dtype, torch.int32)
            self.assertEqual(op, 'sum')
            value.mul_(2)
        comm.all_reduce = reduce
        with patch.object(pi, 'int_mm', return_value=partial), patch.object(pi, 'dequantize', side_effect=lambda value, *a: value):
            output = opt.projected_reduce(None, None, None, None, comm, torch.float16, None)
        self.assertTrue(torch.equal(output, torch.full_like(partial, 2)))
        with self.assertRaises(ValueError):
            opt.enable('fixed')

    def test_rejected_te_ablation_is_outside_runtime(self):
        self.assertFalse((ROOT / 'patches/te_loading.py').exists())
        self.assertTrue((ROOT / 'benchmarks/historical_te_loading.py').is_file())
        for path in (ROOT / 'patches').glob('*.py'):
            self.assertNotIn('historical_te_loading', path.read_text(encoding='utf-8'))

    def test_browser_restores_legacy_widgets_by_name_and_leaves_v2_intact(self):
        node = shutil.which('node')
        if node is None:
            self.skipTest('Node.js required to execute browser migration unit checks')
        schema = nodes.MiniMaxH3TP2Loader.INPUT_TYPES()
        payload = dict(script=str(ROOT / 'custom_nodes/minimax_h3_tp2/web/loader_migration.js'),
                       schema=schema['required'] | schema['optional'])
        source = r'''
import fs from 'node:fs';
import assert from 'node:assert/strict';
const payload = JSON.parse(fs.readFileSync(0, 'utf8'));
const source = fs.readFileSync(payload.script, 'utf8').replace(
  /import \{ app \} from .*?;/,
  'const app = {registerExtension(extension) { globalThis.migration = extension; }};');
const { legacyWidgets, v2Widgets, v3Widgets, restoreWidgets } = await import('data:text/javascript;base64,' + Buffer.from(source).toString('base64'));
const names = Object.keys(payload.schema).filter(name => name !== 'lora_stack');
const makeNode = () => ({properties:{}, widgets:names.map(name => ({name,value:'positional-junk'}))});
const oldValues = legacyWidgets.map(name => payload.schema[name]?.[1]?.default ?? 'retired');
oldValues[legacyWidgets.indexOf('text_encoder')] = '/user/te.json';
oldValues[legacyWidgets.indexOf('lora_mode')] = 'online';
oldValues[legacyWidgets.indexOf('column_linear_backend')] = 'tilelang_exact';
oldValues[legacyWidgets.indexOf('dit_attention')] = 'tilelang_scalar';
oldValues[legacyWidgets.indexOf('te_attention')] = 'tilelang_group32';
const value = (node,name) => node.widgets.find(widget => widget.name === name).value;
for (const length of [17,21,30]) {
  const node = makeNode();
  restoreWidgets(node, {widgets_values:oldValues.slice(0,length)}, payload.schema);
  assert.equal(value(node,'text_encoder'), '/user/te.json');
  assert.equal(value(node,'lora_mode'), 'online');
  assert.equal(value(node,'idle_weight_gib'), 4);
  assert.equal(value(node,'tensor_flow'), 'fp16');
  assert.equal(value(node,'adaptive_lookahead'), true);
  assert.equal(value(node,'column_linear_backend'), 'cuda_int8_tilelang');
  assert.equal(value(node,'dit_attention'), 'kitchen');
  assert.equal(value(node,'te_attention'), 'kitchen_causal');
  assert.equal(value(node,'dit_attention_schedule'), '');
  assert.equal(value(node,'sol_tau'), 1.0);
  assert.equal(value(node,'video_attention'), 'kitchen');
  assert.equal(node.properties.h3_configuration_version, 4);
}
const v2 = makeNode();
const v2Values = v2Widgets.map(name => payload.schema[name]?.[1]?.default ?? 'legacy-v2');
v2Values[v2Widgets.indexOf('dit_attention')] = 'tilelang_drop';
restoreWidgets(v2, {properties:{h3_configuration_version:2},widgets_values:v2Values}, payload.schema);
assert.equal(value(v2,'dit_attention'), 'kitchen');
assert.equal(value(v2,'dit_attention_schedule'), '');
assert.equal(value(v2,'sol_tau'), 1.0);
assert.equal(value(v2,'video_attention'), 'kitchen');
const v3 = makeNode();
const v3Values = v3Widgets.map(name => payload.schema[name]?.[1]?.default ?? 'legacy-v3');
restoreWidgets(v3, {properties:{h3_configuration_version:3},widgets_values:v3Values}, payload.schema);
assert.equal(value(v3,'sol_tau'), 1.0);
assert.equal(v3.properties.h3_configuration_version, 4);
const current = makeNode();
restoreWidgets(current, {properties:{h3_configuration_version:4}}, payload.schema);
assert.equal(value(current,'text_encoder'), 'positional-junk');
function Node() { Object.assign(this,makeNode()); }
Node.prototype.onConfigure = function() { this.originalCalled = true; };
await globalThis.migration.beforeRegisterNodeDef(Node, {name:'MiniMaxH3TP2Loader', input:{required:payload.schema,optional:{}}});
const restored = new Node();
restored.onConfigure({widgets_values:oldValues});
assert.equal(restored.originalCalled, true);
assert.equal(value(restored,'text_encoder'), '/user/te.json');
assert.equal(value(restored,'dit_attention'), 'kitchen');
assert.equal(value(restored,'column_linear_backend'), 'cuda_int8_tilelang');
console.log('Legacy widget migration passed (prefix/full/v2 and callback chain).');
'''
        result = subprocess.run([node, '--input-type=module', '-e', source],
            input=json.dumps(payload), capture_output=True, text=True, timeout=20)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == '__main__':
    unittest.main(verbosity=2)
