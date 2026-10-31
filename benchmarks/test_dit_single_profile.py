"""CPU-only gates for one-NFE profiling; no full model or CUDA setup."""
import ast
import base64
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'ComfyUI'),str(ROOT/'benchmarks')]
import torch
from dit_performance_observer import Phases
from profile_dit_single import runtime,validate_attention_backend


class SingleDiT(unittest.TestCase):
    def test_worker_accepts_bundle_offline_stack_and_rejects_online_stack(self):
        from run_h3_l1_current_e2e import CurrentRank,observation
        def initialize(owner,config):
            owner.config=config;owner.readers={'dit':SimpleNamespace()}
        with patch.object(observation.ObservedRank,'__init__',initialize):
            CurrentRank(SimpleNamespace(bundle_offline_lora=True,lora_stack=('offline',)))
            CurrentRank(SimpleNamespace(bundle_offline_lora=False,lora_stack=()))
            with self.assertRaises(AssertionError):
                CurrentRank(SimpleNamespace(bundle_offline_lora=False,lora_stack=('online',)))

    def test_target_runtime_retains_current_operators_and_tp2_budgets(self):
        config=runtime({'paths':{'te':'te','dit':'baked'}})
        self.assertEqual(config.operators.dit_attention,'kitchen')
        self.assertEqual(config.operators.fc2_quantizer,'fused_stored')
        self.assertEqual(config.operators.row_linear_backend,'tilelang_scaled_fp16')
        self.assertEqual(config.layers,50);self.assertEqual(config.reserve_bytes,9*2**30)
        self.assertEqual(config.timeout_seconds,1500)
        self.assertEqual(config.lora_stack,());self.assertTrue(config.tensor_flow.enabled)

    def test_replacement_runs_sol_and_control_once_each(self):
        from run_h3_l1_current_batch import profile_modes
        modes=profile_modes('sol')
        self.assertEqual(modes,('sol','kitchen'))
        self.assertEqual(len(modes),len(set(modes)))

    def test_exact_native_51220_layout_without_activation_allocation(self):
        from comfy.ldm.minimax.model import PackedLayout
        layout=PackedLayout(1182,107,32,56,603,refs=[dict(kind='image',latent_t=1,latent_h=32,latent_w=56,ref_audio_t=0)]*2)
        self.assertEqual(layout.seq_len,51220)
        self.assertEqual(layout.segments[-1],(3284,51220,'video'))

    def test_sol_policy_preserves_other_operators_and_rejects_kitchen_fallback(self):
        config=runtime({'paths':{'te':'te','dit':'baked'},'operators':{'dit_attention':'sol'}})
        self.assertEqual(config.operators.dit_attention,'sol')
        self.assertEqual(config.operators.fc2_quantizer,'fused_stored')
        self.assertEqual(config.operators.row_linear_backend,'tilelang_scaled_fp16')
        validate_attention_backend('sol',{'DiT/sol_all_int8':50})
        with self.assertRaises(AssertionError):validate_attention_backend('sol',{'kitchen/int8_attention':50})
        with self.assertRaises(AssertionError):validate_attention_backend('sol',{'DiT/sol_all_int8':49})

    def test_sol_notebook_propagates_backend_and_uses_separate_kernel(self):
        from build_kaggle_light_job import main
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);pub=root/'pub.json';q=root/'cpu.json'
            pub.write_text(json.dumps(dict(published=True,capsule_sha256='abc',url='https://example.invalid/frozen',bytes=1,sha256='def')))
            q.write_text(json.dumps(dict(status='cpu_ready',capsule_sha256='abc')))
            with patch.object(sys,'argv',['builder','--mode','dit-profile','--dit-attention','sol',
                    '--publication',str(pub),'--qualification',str(q),'--output',str(root/'job')]):main()
            meta=json.loads((root/'job/kernel-metadata.json').read_text())
            self.assertEqual(meta['id'],'pamcks/h3-dit-sol-single-profile')
            code=''.join(json.loads((root/'job/experiment.ipynb').read_text())['cells'][0]['source'])
            workers=[base64.b64decode(ast.literal_eval(n.args[0])).decode() for n in ast.walk(ast.parse(code))
                if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='b64decode']
            self.assertTrue(any("'--dit-attention','sol'" in w.replace(' ','').replace('\n','') for w in workers))

    def test_first_native_lcm_interval_calls_denoiser_once(self):
        from comfy.k_diffusion.sampling import sample_lcm
        calls=[]
        def model(x,sigma,**kwargs):calls.append(sigma.clone());return x*.9
        sampling=SimpleNamespace(noise_scaling=lambda sigma,noise,x:x+noise*sigma)
        model.inner_model=SimpleNamespace(model_patcher=SimpleNamespace(get_model_object=lambda name:sampling))
        result=sample_lcm(model,torch.ones(1,2,3),torch.tensor([1.,.972972989]),disable=True,
            noise_sampler=lambda *args:torch.zeros(1,2,3))
        self.assertEqual(len(calls),1);self.assertTrue(torch.isfinite(result).all())

    def test_observer_runs_original_once_without_mutating_input(self):
        owner=SimpleNamespace();calls=[]
        def operator(x,**kwargs):calls.append(kwargs);return x+2
        owner.run=operator;phases=Phases(0);phases.wrap(owner,'run','test')
        x=torch.ones(3);actual=owner.run(x,flag=7)
        self.assertTrue(torch.equal(actual,x+2));self.assertTrue(torch.equal(x,torch.ones(3)))
        self.assertEqual(calls,[{'flag':7}]);self.assertFalse(phases.cpu)

    def test_profile_notebook_has_two_t4_and_zero_mounts(self):
        from build_kaggle_light_job import main
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);publication=root/'publication.json';q=root/'cpu.json'
            publication.write_text(json.dumps(dict(published=True,capsule_sha256='abc',url='https://example.invalid/immutable',bytes=1,sha256='def')))
            q.write_text(json.dumps(dict(status='cpu_ready',capsule_sha256='abc')))
            with patch.object(sys,'argv',['builder','--mode','dit-profile','--publication',str(publication),
                    '--qualification',str(q),'--output',str(root/'job'),'--setup-budget','240','--execution-budget','2400']):main()
            meta=json.loads((root/'job/kernel-metadata.json').read_text())
            self.assertEqual(meta['machine_shape'],'NvidiaTeslaT4');self.assertTrue(meta['is_private'])
            self.assertEqual(meta['dataset_sources'],[]);self.assertEqual(meta['kernel_sources'],[])
            code=''.join(json.loads((root/'job/experiment.ipynb').read_text())['cells'][0]['source'])
            tree=ast.parse(code)
            workers=[base64.b64decode(ast.literal_eval(n.args[0])).decode() for n in ast.walk(tree)
                if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='b64decode']
            self.assertTrue(any("'--mode','dit-profile'" in w.replace('\n','') for w in workers))


if __name__=='__main__':unittest.main(verbosity=2)
