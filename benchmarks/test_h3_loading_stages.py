"""Small offline-bake and idle/component ownership regression fixtures."""
from concurrent.futures import CancelledError
import hashlib
import json
from pathlib import Path
import sys
import os
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import weakref
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),os.environ.get('COMFY_ROOT',str(ROOT/'ComfyUI'))]
import torch
from safetensors.torch import save_file
from patches.windowed_tp import CheckpointReader
from patches import offline_lora as bake,staged_vae as staged
from patches.comfy_tp_runtime import PhysicalTPService


class OfflineBake(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.base=self.root/'base.safetensors'
        self.weights={f'blocks.{i}.{name}.weight':torch.arange(32,dtype=torch.float32).reshape(4,8)/100
                      for i in range(4) for name in ('attn.out_proj','mlp.fc2')}
        self.weights['untouched']=torch.tensor([14,9],dtype=torch.int32)
        save_file(self.weights,str(self.base));self.paths=[]
        for j,strength in ((0,.7),(1,-.25)):
            factors={}
            for key in self.weights:
                if not key.endswith('.weight'):continue
                target='diffusion_model.'+key.removesuffix('.weight')
                factors[target+'.lora_A.weight']=torch.full((2,8),.03+j*.01)
                factors[target+'.lora_B.weight']=torch.full((4,2),.04-j*.01)
                factors[target+'.alpha']=torch.tensor(3.)
            path=self.root/f'lora{j}.safetensors';save_file(factors,str(path));self.paths.append((str(path),strength))

    def test_spawned_layer_parallel_bake_matches_serial_and_preserves_base(self):
        before=hashlib.sha256(self.base.read_bytes()).hexdigest()
        one=bake.convert(self.base,self.paths,self.root/'one.safetensors',verify_native=True,devices=('cpu',))
        two=bake.convert(self.base,self.paths,self.root/'two.safetensors',verify_native=True,devices=('cpu','cpu'))
        self.assertEqual(Path(one).read_bytes(),Path(two).read_bytes())
        self.assertEqual(before,hashlib.sha256(self.base.read_bytes()).hexdigest())
        receipt=json.loads(Path(two+'.json').read_text());owners={}
        self.assertEqual(len(receipt['layer_partitions']),2)
        for rank,targets in enumerate(receipt['layer_partitions']):
            self.assertTrue(targets)
            for target in targets:
                layer='.'.join(target.split('.')[:2]);owners.setdefault(layer,set()).add(rank)
        self.assertTrue(all(len(ranks)==1 for ranks in owners.values()))
        self.assertEqual(receipt['verified_native_targets'],8)
        self.assertFalse(Path(two+'.partial').exists())
        with self.assertRaises(FileExistsError):bake.convert(self.base,self.paths,two,devices=('cpu',))

    def test_failed_worker_does_not_publish_checkpoint(self):
        output=self.root/'failed.safetensors'
        with patch.object(bake,'merged_weight',side_effect=RuntimeError('injected bake failure')):
            with self.assertRaisesRegex(RuntimeError,'injected'):bake.convert(self.base,self.paths,output,devices=('cpu',))
        self.assertFalse(output.exists());self.assertFalse(Path(str(output)+'.json').exists())

    def test_cancelled_base_copy_cleans_owned_scratch(self):
        output=self.root/'cancelled.safetensors'
        with patch.object(bake,'_check_cancel',side_effect=CancelledError('cancel fixture')):
            with self.assertRaises(CancelledError):bake.convert(self.base,self.paths,output,devices=('cpu',))
        self.assertFalse(output.exists());self.assertFalse(Path(str(output)+'.partial').exists())

    def test_cached_bake_reuse_and_corruption_rejection(self):
        real_convert=bake.convert;callbacks=[]
        def cpu_convert(*args,**kwargs):return real_convert(*args,**kwargs,devices=('cpu',))
        with patch.object(bake,'convert',side_effect=cpu_convert) as convert:
            path=bake.cached_bake(self.base,self.paths,before_bake=lambda:callbacks.append('cold'))
            again=bake.cached_bake(self.base,self.paths,before_bake=lambda:callbacks.append('warm'))
            self.assertEqual(path,again);self.assertEqual(convert.call_count,1);self.assertEqual(callbacks,['cold'])
            Path(path).touch()
            with self.assertRaisesRegex(RuntimeError,'identity changed'):bake.cached_bake(self.base,self.paths)

    def test_loader_offline_does_not_apply_adapters_twice(self):
        import custom_nodes.minimax_h3_tp2 as node
        class Service:
            def __init__(self,config):self.config=config;self.closed=False
            def close(self):self.closed=True
        with patch.object(node,'_service',None),patch.object(node,'PhysicalTPService',Service), \
                patch.object(node,'create_model'),patch.object(node,'create_clip'), \
                patch.object(bake,'cached_bake',return_value=str(self.base)) as merge:
            s=node.MiniMaxH3TP2Loader().load('te',str(self.base),lora_stack=tuple(self.paths),lora_mode='offline')[2]
            self.assertEqual(s.config.lora_stack,());self.assertEqual(s.config.dit_path,str(self.base))
            self.assertEqual(merge.call_count,1)


class ComponentAndIdle(unittest.TestCase):
    def service(self):
        s=PhysicalTPService.__new__(PhysicalTPService)
        s.config=SimpleNamespace(idle_residency=True);s.lock=threading.RLock();s.closed=False;s.idle_held=False
        s.staged_vaes=weakref.WeakSet();s.idle_vae_names=None;s.idle_vae_cancel=threading.Event()
        s.idle_vae_thread=None;s.idle_vae_stats=dict(loads=0,errors=0,last_error=None);s.commands=[]
        def call(cmd,*args,**kw):s.commands.append(cmd);return {'accepted':True}
        s.call=call
        return s

    def vae(self,s,name='video.safetensors',preload=None):
        class VAE:
            def __init__(self):self.stage='decode';self.reader=SimpleNamespace(path=Path(name));self.vae=SimpleNamespace(patcher=object());self.released=0
            def preload_idle(self,cancel):self.stage='encode';return True if preload is None else preload(cancel)
            def release(self):self.released+=1;self.stage=None
        v=VAE();s.register_staged_vae(v);return v

    def test_encoder_prefetch_before_te_and_sampling_release(self):
        s=self.service();order=[];v=self.vae(s,preload=lambda cancel:order.append('encoder') or True)
        call=s.call
        def observe(*a,**kw):order.append(a[0]);return call(*a,**kw)
        s.call=observe
        self.assertTrue(s.request_idle());s.idle_vae_thread.join(5)
        self.assertFalse(s.idle_vae_thread.is_alive());self.assertEqual(order,['encoder','idle_begin'])
        self.assertEqual(v.stage,'encode');self.assertEqual(s.idle_vae_stats['loads'],1)
        s.release_staged_encoders();self.assertIsNone(v.stage)

    def test_cancel_while_preloading_drops_encoder_and_stops_te(self):
        s=self.service();entered=threading.Event()
        def preload(cancel):
            entered.set();cancel.wait(5)
            if cancel.is_set():raise CancelledError()
        v=self.vae(s,preload=preload)
        s.request_idle();self.assertTrue(entered.wait(5))
        s.pause_idle(keep=False);s.idle_vae_thread.join(5)
        self.assertFalse(s.idle_vae_thread.is_alive());self.assertIsNone(v.stage)
        self.assertNotIn('idle_begin',s.commands);self.assertIn('idle_stop',s.commands)
        self.assertFalse(s.idle_held)

    def test_preload_failure_keeps_te_prefetch_and_filters_current_graph(self):
        s=self.service()
        def fail(cancel):raise MemoryError('fixture admission')
        old=self.vae(s,'old.safetensors');v=self.vae(s,preload=fail)
        s.configure_idle_vaes({'v':dict(class_type='MiniMaxH3ParallelVAELoader',inputs={'vae_name':'video.safetensors'})})
        self.assertEqual(s._idle_vaes(),[v]);s.request_idle();s.idle_vae_thread.join(5)
        self.assertIn('idle_begin',s.commands);self.assertEqual(s.idle_vae_stats['errors'],1)
        self.assertIsNone(old.stage)
        s.configure_idle_vaes({'v':dict(class_type='MiniMaxH3ParallelVAELoader',inputs={'vae_name':['primitive',0]})})
        self.assertEqual(s._idle_vaes(),[])

    def test_pressure_releases_both_encoder_and_te(self):
        s=self.service();v=self.vae(s);s.idle_held=True
        self.assertTrue(s.pause_idle(keep=False,pressure=True))
        self.assertIsNone(v.stage);self.assertEqual(s.commands,['idle_release'])

    def test_pressure_can_release_vae_after_te_consumed_idle_window(self):
        s=self.service();v=self.vae(s);s.idle_held=False
        self.assertTrue(s.pause_idle(keep=False,pressure=True));self.assertIsNone(v.stage)

    def test_abort_reconciles_parent_vae_lease(self):
        from patches.host_weight_cache import HostBudget
        s=self.service();s.budget=HostBudget(1024,reserve_bytes=0)
        s.processes=[];s.connections=[];s.store=object()
        v=self.vae(s);v.reservation=128;s.budget.acquire(128)
        def release():
            s.budget.release(v.reservation);v.reservation=0;v.stage=None
        v.release=release
        s.abort();self.assertEqual(s.budget.snapshot()['reserved_bytes'],0);self.assertIsNone(v.stage)

    def test_component_selection_is_disjoint_except_shared_norms(self):
        names=['encoder.w','quant_conv.w','decoder.w','post_quant_conv.w','latents_mean','latents_std',
               'pre_block.attn.zero_k_bias','mean_proj.w','logs_proj.w','dec_in_proj.w']
        r=SimpleNamespace(header=dict.fromkeys(names,{}))
        for kind in ('video','audio'):
            enc=set(staged.component_names(r,kind,'encode'));dec=set(staged.component_names(r,kind,'decode'))
            self.assertEqual(enc&dec,{'latents_mean','latents_std'})
            self.assertTrue('encoder.w' in enc and 'decoder.w' not in enc)
            self.assertTrue('decoder.w' in dec and 'encoder.w' not in dec)

    def test_staged_component_switch_releases_old_lease_and_never_overlaps(self):
        from patches.host_weight_cache import HostBudget
        s=self.service();s.budget=HostBudget(100*2**20,reserve_bytes=0)
        v=staged.StagedVAE.__new__(staged.StagedVAE)
        v.service=s;v.lock=threading.RLock();v.stage=None;v.reservation=0;v.events=[];v.kind='video'
        v.reader=SimpleNamespace(header={'encoder.w':{'data_offsets':[0,4]},'decoder.w':{'data_offsets':[4,8]}})
        v.vae=SimpleNamespace(device='cpu',first_stage_model=None,patcher=None,size=None)
        def model(*args):
            self.assertIsNone(v.stage);self.assertIsNone(v.vae.first_stage_model)
            return torch.nn.Linear(1,1).eval().requires_grad_(False)
        with patch.object(v,'check_identity'),patch.object(staged,'load_component',side_effect=model), \
                patch('comfy.model_patcher.CoreModelPatcher',new=lambda model,**kw:SimpleNamespace(model=model)), \
                patch('comfy.model_management.archive_model_dtypes',new=lambda m:None), \
                patch('comfy.model_management.unload_model_and_clones',new=lambda p:None):
            v._ensure('encode');first=weakref.ref(v.vae.first_stage_model)
            v._ensure('decode');self.assertIsNone(first());self.assertEqual(v.stage,'decode')
            self.assertEqual(s.budget.snapshot()['reserved_bytes'],v.reservation)
            v.release();self.assertEqual(s.budget.snapshot()['reserved_bytes'],0)


if __name__=='__main__':
    # ComfyUI's argument parser must not consume unittest flags or spawn args.
    unittest.main(verbosity=2)
