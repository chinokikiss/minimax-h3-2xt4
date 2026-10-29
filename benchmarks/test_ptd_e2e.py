"""Long-request admission and actual Native CFGGuider/PTD lifecycle contracts."""
from copy import deepcopy
from contextlib import contextmanager
import json
from pathlib import Path
from types import SimpleNamespace
import threading
import unittest
from unittest.mock import patch

import torch
from benchmarks.ptd_e2e_contract import from_previous,validate,resource_plan,RELEASE

ROOT=Path(__file__).resolve().parents[1]


def plan():
    return from_previous(json.loads((ROOT/'turbo_plan.json').read_text(encoding='utf-8')))


class RequestContracts(unittest.TestCase):
    def test_exact_previous_input_and_trained_ptd_switches(self):
        p=plan(); self.assertTrue(validate(p)['full_pipeline'])
        self.assertNotIn('turbo_lora',p)
        for change in (dict(sampler='euler'),dict(video_shift=6.),dict(seed=1),
                dict(auxiliary_layout='other'),dict(allow_pruned_adaln=False),
                dict(lora_stack=[('online.safetensors',1.)]),dict(prompt=p['prompt']+'changed')):
            with self.subTest(change=tuple(change)),self.assertRaises(ValueError): validate(p|change)

    def test_runtime_keeps_long_activation_reserve_without_online_lora(self):
        from benchmarks.run_ptd_e2e import runtime_config
        from patches.h3_operator_policy import H3OperatorPolicy
        from dataclasses import asdict
        runtime=runtime_config(plan()|dict(paths=dict(te='te.json',dit='base.json')))
        self.assertEqual(runtime.lora_stack,())
        self.assertEqual(runtime.reserve_bytes,9*2**30)
        self.assertEqual(runtime.cache_bytes,4*2**30)
        self.assertEqual(runtime.host_bytes,20*2**30)
        self.assertEqual(asdict(runtime.operators),asdict(H3OperatorPolicy()))
        self.assertEqual(runtime.tensor_flow.mode,'fp16')

    def test_disk_and_ram_admission_include_all_eight_experts(self):
        assets=json.loads((ROOT/'bundle_assets.json').read_text())
        vaes=json.loads((ROOT/'l1_model_identity.json').read_text())
        kwargs=dict(header=assets['auxiliary']['dit']['native_header'],assets=assets,vaes=vaes)
        resources=resource_plan(**kwargs,free_bytes=2**40,host_limit=30*2**30)
        self.assertGreater(resources['estimated_bank_bytes'],100*2**30)
        self.assertGreater(resources['required_free_bytes'],resources['estimated_bank_bytes']+RELEASE['bytes'])
        with self.assertRaises(OSError): resource_plan(**kwargs,free_bytes=1,host_limit=30*2**30)
        with self.assertRaises(MemoryError): resource_plan(**kwargs,free_bytes=2**40,host_limit=16*2**30)

    def test_production_bank_rejects_fixture_wrong_checkpoint_or_clock(self):
        from benchmarks.run_ptd_e2e import validate_bank
        bank=SimpleNamespace(layers=50,hops=[(None,dict(native_verified_targets=list(range(208))))]*8,
            recipe=dict(checkpoint_sha256=RELEASE['sha256'],video_shift=12.,audio_shift=3.),
            digest='a'*64,recipe_sha='b'*64)
        self.assertEqual(validate_bank(bank)['verified_targets_by_hop'],[208]*8)
        for key,value in (('layers',2),('recipe',bank.recipe|dict(checkpoint_sha256='0'*64)),
                ('recipe',bank.recipe|dict(video_shift=6.))):
            changed=deepcopy(bank); setattr(changed,key,value)
            with self.assertRaises(ValueError): validate_bank(changed)

    def test_p1_is_optional_and_disabled_in_every_default(self):
        from benchmarks.run_ptd_e2e import ptd_options
        from patches.ptd.runtime import PTDOptions
        from patches.ptd.nodes import MiniMaxH3TP2PTDAdapter
        p=plan()
        self.assertFalse(PTDOptions().prefetch_refiner)
        self.assertFalse(MiniMaxH3TP2PTDAdapter.INPUT_TYPES()['optional']['prefetch_refiner'][1]['default'])
        self.assertFalse(p['prefetch_refiner']); self.assertFalse(ptd_options(p).prefetch_refiner)
        self.assertTrue(ptd_options(p|dict(prefetch_refiner=True)).prefetch_refiner)
        with self.assertRaises(ValueError): ptd_options(p|dict(prefetch_refiner='true'))


class NativeSamplingContract(unittest.TestCase):
    def test_cfg_guider_runs_exactly_eight_hops_and_restores_session(self):
        # Real legacy MODEL/extra_conds/CFGGuider/KSAMPLER path with tiny CPU
        # latents. The fake service is the only seam: no real weights or GPU.
        from patches.comfy_sampler import create_model
        from patches.comfy_tp_runtime import RuntimeConfig
        from patches.ptd.runtime import adapt_model,PTDOptions
        from patches.ptd.sampler import make_sampler,initial_state
        from patches.ptd.schedule import hop_sigmas
        from comfy.nested_tensor import NestedTensor
        from comfy_extras.nodes_minimax_h3 import MiniMaxH3SigmaShift
        import comfy.samplers
        class Service:
            config=RuntimeConfig('te','dit')
            lock=threading.RLock()
            closed=False
            operator_aot=False
            def __init__(self): self.commands=[]; self.payloads=[]
            def call(self,command,*args,**kwargs):
                self.commands.append((command,args))
                if command=='denoise':
                    self.payloads.append(kwargs['minimax_payload'])
                    return [torch.zeros_like(x) for x in args[0]]
        service=Service()
        bank=SimpleNamespace(sigmas=hop_sigmas(8),digest='b'*64)
        native=create_model(service,device='cpu')
        native=MiniMaxH3SigmaShift.execute(native,12.,3.)[0]
        with patch('patches.ptd.runtime.PTDBank',return_value=bank):
            model,controller=adapt_model(native,service,'unused',PTDOptions(enabled=True,allow_pruned_adaln=True))
        # Model-only adaptation must leave the caller's original shifts alone.
        self.assertEqual(native.get_model_object('model_sampling').audio_shift,3.)
        self.assertEqual(model.get_model_object('model_sampling').audio_shift,12.)
        raw=torch.full((1,4,5120),100000.,dtype=torch.float32)
        positive=[[raw,{}]]
        noise=NestedTensor([torch.ones(1,24,1,4,4),torch.ones(1,32,2,8)])
        latent=NestedTensor([torch.zeros_like(v) for v in noise.unbind()])
        guider=comfy.samplers.CFGGuider(model); guider.set_conds(positive,positive); guider.set_cfg(1.)
        callbacks=[]
        result=guider.sample(noise,latent,make_sampler(controller),controller.sigmas.float(),
            callback=lambda *args:callbacks.append(args),disable_pbar=True,seed=20261020)
        commands=[name for name,_ in service.commands]
        self.assertEqual(commands[0:2],['ptd_configure','begin_sample'])
        self.assertEqual(commands[-2:],['end_sample','ptd_release'])
        self.assertEqual([args[0] for name,args in service.commands if name=='ptd_hop'],list(range(8)))
        self.assertEqual(commands.count('denoise'),8); self.assertEqual(len(callbacks),8)
        for payload in service.payloads:
            torch.testing.assert_close(payload['ptd_text_states'],raw)
        for actual,expected in zip(result.unbind(),initial_state(noise).unbind()):
            torch.testing.assert_close(actual,expected)


if __name__=='__main__': unittest.main(verbosity=2)
