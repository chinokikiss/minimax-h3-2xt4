"""PTD contracts; run in the frozen T4 job, never with local LoRA weights."""
from contextlib import contextmanager
import importlib.util
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from safetensors.torch import save_file

from patches.ptd.checkpoint import PTDCheckpoint, HopStore
from patches.ptd.mapping import expected_adapter_keys, map_adapter_key
from patches.ptd.schedule import hop_plans, hop_sigmas
from patches.ptd.sampler import sample_ptd, initial_state


def fixture(directory):
    values, base = {}, {}
    for index, key in enumerate(sorted(expected_adapter_keys(1, 2))):
        target, factor = map_adapter_key(key)
        output = 8 if target.swap_halves else 4
        shape = (8, 2, 4) if factor == 'down' else (8, output, 2)
        values[key] = (torch.arange(torch.tensor(shape).prod()).reshape(shape).float()+index+1) / 100
        base[target.module+'.weight'] = {'shape': [output*(3 if target.qkv_index is not None else 1), 4]}
    for prefix, native, output in (('proj_out', 'video_out', 3), ('audio_proj_out', 'audio_out', 2)):
        values[prefix+'.weight'] = torch.arange(32*output*4).reshape(32, output, 4).float()/100
        values[prefix+'.bias'] = torch.arange(32*output).reshape(32, output).float()/10
        base['final_layer.'+native+'.weight'] = {'shape': [output, 4]}
    path = Path(directory)/'tiny.safetensors'
    save_file(values, str(path), metadata={'ptd_num_hops':'8', 'pdd_num_steps':'32', 'lora_rank':'2', 'lora_alpha':'2'})
    return path, values, SimpleNamespace(header=base)


class CheckpointContracts(unittest.TestCase):
    def test_independent_qkv_and_gate_order_every_hop(self):
        with tempfile.TemporaryDirectory() as directory:
            path, values, base = fixture(directory)
            checkpoint = PTDCheckpoint(path, base)
            for hop in range(8):
                store = HopStore(checkpoint, hop)
                terms = store.terms('blocks.0.attn.qkv_proj')
                actual = sum(b@a*scale for a,b,scale in terms)
                expected = torch.cat([values[f'transformer_blocks.0.attn.to_{name}.lora_up'][hop]
                    @ values[f'transformer_blocks.0.attn.to_{name}.lora_down'][hop] for name in ('q','k','v')])
                torch.testing.assert_close(actual, expected)
                a,b,scale = store.terms('blocks.0.mlp.fc1')[0]
                raw = values['transformer_blocks.0.ff.net.0.proj.lora_up'][hop]
                torch.testing.assert_close(b, torch.cat((raw[4:], raw[:4])))
                self.assertEqual(scale, 1.)
                self.assertEqual(len(store.targets), 12)

    def test_unknown_keys_incomplete_pair_bad_shapes_and_mutation_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            path, values, base = fixture(directory)
            for change in ('extra', 'missing', 'shape', 'head'):
                altered = dict(values)
                if change == 'extra': altered['unknown.lora_down'] = torch.ones(1)
                elif change == 'missing': altered.pop(next(iter(values)))
                elif change == 'shape': altered[next(iter(values))] = torch.ones(7, 2, 4)
                else: altered['proj_out.bias'] = torch.ones(31, 3)
                bad = Path(directory)/(change+'.safetensors')
                save_file(altered, str(bad), metadata={'ptd_num_hops':'8', 'lora_rank':'2'})
                with self.assertRaises((ValueError, KeyError)): PTDCheckpoint(bad, base)
            checkpoint = PTDCheckpoint(path, base)
            with path.open('ab') as stream: stream.write(b'x')
            with self.assertRaises(RuntimeError): checkpoint.check()

    def test_full_head_bias_fusion_and_independent_audio_weights(self):
        for video, audio in hop_plans(8):
            self.assertAlmostEqual(float(video.sum()), 1., places=6)
            self.assertAlmostEqual(float(audio.sum()), 1., places=6)
            self.assertEqual(int(torch.count_nonzero(video)), 4)
            self.assertFalse(torch.equal(video, audio))
            heads = torch.arange(32*3*4).reshape(32,3,4).float()/100
            bias = torch.arange(32*3).reshape(32,3).float()/10
            x = torch.ones(2,4)
            w = torch.einsum('pn,noi->poi',video,heads).flatten(0,1)
            b = torch.einsum('pn,no->po',video,bias).flatten()
            expected = sum(float(video[0,i])*(x@heads[i].T+bias[i]) for i in range(32))
            torch.testing.assert_close(torch.nn.functional.linear(x,w,b), expected)


class SamplerContracts(unittest.TestCase):
    def controller(self):
        class Controller:
            hops = 8
            sigmas = hop_sigmas(8)
            calls = []
            @contextmanager
            def activate(self, hop, sigma):
                self.calls.append((hop,sigma)); yield
        return Controller()

    def test_exact_eight_call_recurrence_and_fp32_carry(self):
        controller = self.controller()
        noise = torch.tensor([[1.,2.]], dtype=torch.float32)
        class Model:
            latent_image = torch.zeros_like(noise)
            def __init__(self): self.noise = noise
            def __call__(self, state, sigma, **kwargs):
                self.last_dtype = state.dtype
                return state-sigma.float().reshape(-1,1)*len(controller.calls)
        model = Model(); callbacks = []
        output = sample_ptd(model,noise,controller.sigmas.float(),controller=controller,callback=callbacks.append)
        expected = initial_state(noise)
        for i in range(8): expected = expected+(controller.sigmas[i+1]-controller.sigmas[i]).float()*(i+1)
        torch.testing.assert_close(output,expected)
        self.assertEqual(model.last_dtype,torch.float32)
        self.assertEqual([h for h,_ in controller.calls],list(range(8)))
        self.assertEqual(len(callbacks),8)

    def test_bad_grid_partial_denoise_and_mask_rejected(self):
        controller = self.controller(); x=torch.zeros(1,2)
        for grid in (controller.sigmas[:5],controller.sigmas.flip(0),controller.sigmas+1e-4):
            with self.assertRaises(ValueError): sample_ptd(None,x,grid,controller=controller)
        with self.assertRaises(ValueError):
            sample_ptd(None,x,controller.sigmas,{'denoise_mask':torch.ones_like(x)},controller=controller)
        with self.assertRaises(ValueError):
            sample_ptd(SimpleNamespace(latent_image=torch.ones_like(x)),x,controller.sigmas,controller=controller)

    def test_audio_chord_and_native_clock_agree_at_all_knots(self):
        from patches.ptd.runtime import audio_sigma,native_audio_shift
        for time in hop_sigmas(8)[:-1].tolist():
            shift = native_audio_shift(time)
            actual = shift*time/(1+(shift-1)*time)
            self.assertAlmostEqual(actual,audio_sigma(time),places=11)


class AdapterContracts(unittest.TestCase):
    def test_disabled_returns_native_model_sampler_and_sigmas_by_identity(self):
        from patches.ptd.runtime import adapt_model,PTDOptions
        native = object()
        self.assertEqual(adapt_model(native,None,None,PTDOptions()),(native,None))
        path=Path(__file__).resolve().parents[1]/'custom_nodes/minimax_h3_tp2/ptd_nodes.py'
        spec=importlib.util.spec_from_file_location('ptd_nodes_fixture',path)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        sampler,sigmas=object(),object()
        result=module.MiniMaxH3TP2PTDSampler().sampler(None,False,sampler,sigmas)
        self.assertIs(result[0],sampler);self.assertIs(result[1],sigmas)
        self.assertEqual(module.MiniMaxH3TP2PTDAdapter.INPUT_TYPES()['required']['session'],('MINIMAX_TP2_SESSION',))

    def test_raw_te_payload_retains_fp32_outliers(self):
        import comfy.conds
        from patches.ptd.runtime import Controller
        raw=torch.ones(1,2,5120)*100000
        conditions={'minimax_payload':comfy.conds.CONDConstant({'seed':7}),
                    'c_crossattn':comfy.conds.CONDRegular(raw.half())}
        result=Controller.conditions(None,conditions,{'cross_attn':raw})
        self.assertEqual(result['minimax_payload'].cond['ptd_text_states'].dtype,torch.float32)
        self.assertTrue(torch.isfinite(result['minimax_payload'].cond['ptd_text_states']).all())
        self.assertEqual(result['minimax_payload'].cond['seed'],7)
        self.assertTrue(torch.isfinite(result['c_crossattn'].cond).all())

    def test_pruned_approximation_requires_explicit_consent(self):
        from patches.ptd.runtime import WorkerAdapter,PTDOptions
        from patches.tp2_bundle import BundleReader
        base=object.__new__(BundleReader)
        base.path=Path('/tmp/base');base.header={'adaln_t_table':{}}
        runtime=SimpleNamespace(readers={'dit':base},config=SimpleNamespace(lora_stack=(),layers=2))
        bank=SimpleNamespace(base=base.path,layers=2)
        with patch('patches.ptd.runtime.PTDBank',return_value=bank):
            with self.assertRaisesRegex(ValueError,'approximation consent'):
                WorkerAdapter(runtime,'unused',PTDOptions(enabled=True))

    def test_native_aot_uses_partial_96_rotation(self):
        from patches.h3_operator_aot import plan
        from patches.h3_operator_policy import H3OperatorPolicy
        calls=plan(17,0,H3OperatorPolicy())
        rope=[args for module,name,args in calls if module=='tilelang_fp32_rms_rope']
        self.assertEqual(rope[0][3],96)

    def test_expert_identity_order_cleanup_and_base_restore(self):
        from patches.ptd.runtime import WorkerAdapter,PTDOptions
        from patches.tp2_bundle import BundleReader
        base=object.__new__(BundleReader);base.path=Path('/tmp/base');base.header={'adaln_t_table':{}}
        operations=SimpleNamespace(); flow=SimpleNamespace()
        cache_calls=[]; dropped=[]
        runtime=SimpleNamespace(readers={'dit':base},rank=0,sample_active=False,
            config=SimpleNamespace(lora_stack=(),layers=2,operators=operations,tensor_flow=flow),
            _drop_model=lambda:dropped.append(True),
            cache=SimpleNamespace(begin_request=lambda *a,**k:cache_calls.append((a,k))))
        bank=SimpleNamespace(base=base.path,layers=2,sigmas=hop_sigmas(8),check=lambda:None,
            reader=lambda hop,**kwargs:SimpleNamespace(identity=('fixture',hop)))
        with patch('patches.ptd.runtime.PTDBank',return_value=bank), \
                patch('patches.ptd.runtime.tp_specs',side_effect=lambda reader,*a:[reader.identity]):
            adapter=WorkerAdapter(runtime,'unused',PTDOptions(enabled=True,allow_pruned_adaln=True))
            self.assertEqual(runtime.readers['dit'].identity,('fixture',0))
            with self.assertRaises(ValueError):adapter.select(0,float(bank.sigmas[0]))
            runtime.sample_active=True
            adapter.calls.append(0)
            with self.assertRaises(ValueError):adapter.select(2,float(bank.sigmas[2]))
            adapter.select(1,float(bank.sigmas[1]))
            self.assertEqual(runtime.readers['dit'].identity,('fixture',1))
            self.assertTrue(cache_calls[-1][1]['discard_unwanted'])
            adapter.close()
            self.assertIs(runtime.readers['dit'],base)
            self.assertEqual(len(dropped),3)


if __name__=='__main__': unittest.main(verbosity=2)
