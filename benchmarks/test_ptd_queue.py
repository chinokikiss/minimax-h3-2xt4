"""CPU queue/ownership contracts; actual numerical execution remains on T4."""
from collections import Counter
from contextlib import contextmanager
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch
from torch import nn

from patches.weight_residency import ResidencyWindow, WeightSpec, _Entry, module_bytes, copy_module
from patches.ptd.queue import ExpertQueue, PlannedWindow, QueuedFinalLayer, refiner_group
from patches.h3_tensor_flow import TensorFlowPolicy, prepare_dit


def empty_window(specs=()):
    window=object.__new__(ResidencyWindow)
    window.specs={spec.key:spec for spec in specs}
    window.uses=();window.remaining=Counter();window.cursor=0
    window.closed=False;window.idle_mode=False;window.pinned_bytes=4096
    window.stats=dict(peak_live_weight_bytes=0,host_wait_ms=0.,h2d_bytes=0)
    return window


def native_tiny():
    import comfy.ops
    from comfy.ldm.minimax.model import MiniMaxH3Model
    from patches.h3_operator_policy import NativeRefinerAttention
    model=MiniMaxH3Model(hidden_size=8,num_layers=0,token_refiner_num_layers=2,
        num_attention_heads=2,attention_head_dim=4,ffn_hidden_size=16,latents_dim=2,audio_latents_dim=2,
        text_dim=6,time_embed_dim=4,adaln_curve_grid=4,dtype=torch.float32,device='cpu',
        operations=comfy.ops.disable_weight_init)
    with torch.no_grad():
        for i,value in enumerate(model.parameters()): value.fill_(.01+(i%7)*.001)
    for block in model.token_refiner.blocks:block.attn=NativeRefinerAttention(block.attn)
    return model


class QueueContracts(unittest.TestCase):
    def test_append_preserves_active_entry_and_rejects_rebinding_atomically(self):
        first=WeightSpec('a',lambda:nn.Linear(2,2),64)
        second=WeightSpec('b',lambda:nn.Linear(2,2),64)
        window=empty_window((first,));window.uses=('a',);window.remaining=Counter(a=1)
        active=SimpleNamespace(active=True,ready=object(),last_use=object())
        window.entries={'a':active};window.retired=[object()]
        window.append_plan((second,),('b',))
        self.assertEqual(window.uses,('a','b'));self.assertIs(window.entries['a'],active)
        self.assertEqual(len(window.retired),1);self.assertEqual(window.remaining,Counter(a=1,b=1))
        with self.assertRaises(ValueError):window.append_plan((WeightSpec('a',lambda:None,64),),('a',))
        with self.assertRaises(ValueError):window.append_plan((),('unknown',))
        self.assertEqual(window.uses,('a','b'))

    def test_pageable_reservation_is_owned_until_ready_without_double_pinning(self):
        host=nn.Linear(2,2,bias=False);size=module_bytes(host)
        spec=WeightSpec('refiner',lambda:host,size,pin_backing=False)
        window=empty_window((spec,));window.pending={};window.unmaterialized_staging={}
        window.host_reservations={};window.observations=[];window.retired=[];window.idle_pending=set()
        release=Mock();cache=SimpleNamespace(take=Mock(return_value=(host,size)),pin=Mock(),
            budget=SimpleNamespace(release=release))
        window.host_cache=cache
        result,actual=window._prepare(spec)
        self.assertIs(result,host);self.assertEqual(actual,size);cache.pin.assert_not_called()
        ready=SimpleNamespace(query=Mock(return_value=False))
        entry=_Entry(host,host,size,ready,ready,copy_reported=True,
                     host_reservation=window.host_reservations.pop('refiner'),pinned=False)
        window.entries={'refiner':entry}
        self.assertEqual(window._host_bytes(),0)
        window._reap();release.assert_not_called();self.assertIs(entry.host,host)
        ready.query.return_value=True
        window._reap();release.assert_called_once_with(size);self.assertIsNone(entry.host)
        window._reap();release.assert_called_once_with(size)

    def test_resident_or_pageable_group_is_not_rejected_by_a_pin_only_bound(self):
        for options in ({'resident_backing':True},{'pin_backing':False}):
            spec=WeightSpec('large',lambda:nn.Linear(2,2),8192,**options)
            with patch('torch.cuda.Stream',return_value=object()):
                window=ResidencyWindow((spec,),(),'cuda:0',max_weight_bytes=16384,pinned_bytes=4096)
            window.pool.shutdown();window.closed=True
            self.assertEqual(window._staging_bytes(spec),0)
        with self.assertRaises(ValueError):
            ResidencyWindow((WeightSpec('large',lambda:None,8192),),(),'cuda:0',
                            max_weight_bytes=16384,pinned_bytes=4096)

    def test_refiner_matches_fresh_native_loading_and_fp16_folding(self):
        source,oracle=native_tiny(),native_tiny()
        values={}
        for index,block in enumerate(oracle.token_refiner.blocks):
            for part,projection in (('attn.qkv_proj',block.attn.base.qkv_proj),
                    ('attn.out_proj',block.attn.base.out_proj),('mlp.fc1',block.mlp.fc1),('mlp.fc2',block.mlp.fc2)):
                name=f'token_refiner.blocks.{index}.{part}'
                value=(torch.arange(projection.weight.numel()).reshape(projection.weight.shape)*.0001+.02).half()
                values[name+'.weight']=value
                with torch.no_grad():projection.weight.copy_(value)
        flow=TensorFlowPolicy('fp16')
        prepare_dit(source,flow,rank=0);prepare_dit(oracle,flow,rank=0)
        reader=SimpleNamespace(hop=1,bank=SimpleNamespace(hops=[None,(None,{
            'auxiliary':{'verified_targets':[key.removesuffix('.weight') for key in values]}})]),
            read_tensor=lambda name:values[name].clone())
        group=refiner_group({'condition_proj':source.condition_proj,'token_refiner':source.token_refiner},reader,flow)
        expected=nn.ModuleDict({'condition_proj':oracle.condition_proj,'token_refiner':oracle.token_refiner})
        self.assertEqual(set(group.state_dict()),set(expected.state_dict()))
        for name,value in expected.state_dict().items():self.assertTrue(torch.equal(group.state_dict()[name],value),name)
        self.assertEqual(group['token_refiner'].blocks[0].norm1.eps,oracle.token_refiner.blocks[0].norm1.eps)

    def test_final_layer_uses_native_math_and_restores_heads_after_failure(self):
        native=native_tiny().final_layer
        heads=nn.ModuleDict({name:copy_module(getattr(native,name),lambda value:value.clone())
                             for name in ('video_out','audio_out')})
        reference=copy_module(native,lambda value:value.clone())
        factor=1.7
        reference.audio_out.register_forward_hook(lambda module,args,output:output*factor)
        entered=[]
        class Window:
            @contextmanager
            def use(self,key):
                entered.append(key)
                try:yield heads
                finally:entered.append('retired')
        wrapper=QueuedFinalLayer(native,Window());wrapper.head_key='hop1';wrapper.audio_factor=factor
        args=(torch.full((4,8),.05),torch.full((2,4),.1),(0,2,0),(2,4,1),torch.tensor(.7),None,(1.,1.))
        expected=reference(*args);actual=wrapper(*args)
        for left,right in zip(actual,expected):self.assertTrue(torch.equal(left,right))
        with patch.object(heads['audio_out'],'forward',side_effect=RuntimeError('head failure')):
            with self.assertRaisesRegex(RuntimeError,'head failure'):wrapper(*args)
        self.assertIsInstance(native.video_out,nn.Identity);self.assertIsInstance(native.audio_out,nn.Identity)
        self.assertEqual(entered,['hop1','retired','hop1','retired'])

    def test_expert_queue_keeps_shell_window_and_appends_only_the_next_hop(self):
        model=native_tiny();flow=TensorFlowPolicy('fp16');prepare_dit(model,flow,rank=0)
        model.blocks=nn.ModuleList([nn.Identity(),nn.Identity()])
        canonical={name:getattr(model,name) for name in ('condition_proj','token_refiner','video_patch_proj','audio_patch_proj')}
        ref=WeightSpec('ref0',lambda:nn.ModuleDict(),1024,resident_backing=True)
        project=WeightSpec('project',lambda:nn.ModuleDict(),128,resident_backing=True)
        blocks=tuple(WeightSpec(f'b0_{i}',lambda:nn.Identity(),64) for i in range(2))
        window=empty_window((ref,project,*blocks))
        window.pinned_bytes=4*2**30
        header={f'final_layer.{name}.{attribute}':{'shape':list(getattr(getattr(model.final_layer,name),attribute).shape)}
            for name in ('video_out','audio_out') for attribute in ('weight','bias')}
        def reader(hop,**kwargs):return SimpleNamespace(identity=('hop',hop),header=header)
        cache=SimpleNamespace(begin_request=Mock())
        preprocessing=SimpleNamespace(canonical=canonical,refine_key='ref0',project_key='project',project_uses=1)
        runtime=SimpleNamespace(window=window,preprocessing=preprocessing,keys=[spec.key for spec in blocks],
            model=model,rank=0,readers={'dit':reader(0)},cache=cache,
            config=SimpleNamespace(layers=2,operators=object(),tensor_flow=flow))
        adapter=SimpleNamespace(runtime=runtime,original=object(),bank=SimpleNamespace(reader=reader),
            options=SimpleNamespace(use_hop_refiner=True,use_pdd_heads=True,future_cache=False))
        def block_specs(reader,*args):
            return [WeightSpec(f'b{reader.identity[1]}_{i}',lambda:nn.Identity(),64) for i in range(2)]
        with patch('patches.ptd.queue.tp_specs',side_effect=block_specs):
            queue=ExpertQueue(adapter)
            shell_id=id(runtime.model);window_id=id(queue.window)
            self.assertEqual(set(queue.plans),{0,1})
            window.cursor=queue.starts[1];queue.select(1)
            self.assertEqual(set(queue.plans),{0,1,2})
            self.assertEqual(id(runtime.model),shell_id);self.assertEqual(id(queue.window),window_id)
            self.assertEqual([block.key for block in model.blocks],['b1_0','b1_1'])
            self.assertFalse(cache.begin_request.call_args.kwargs['discard_unwanted'])
            old_plan=window.uses
            queue.view.extend_uses((queue.refine_keys[1],))
            self.assertEqual(window.uses,old_plan)
            with self.assertRaises(RuntimeError):queue.view.extend_uses(('wrong',))

    def test_finish_drains_once_before_releasing_auxiliary_owners(self):
        order=[];queue=object.__new__(ExpertQueue);queue.finished=False
        queue.adapter=SimpleNamespace(options=SimpleNamespace(future_cache=False))
        queue.window=SimpleNamespace(copy_stream=SimpleNamespace(synchronize=lambda:order.append('copy_fence')),
            _reap=lambda:order.append('reap'),close=lambda:order.append('close'),specs={'owned':object()})
        queue.runtime=SimpleNamespace(config=SimpleNamespace(timeout_seconds=7),cache=SimpleNamespace(
            begin_request=lambda *a,**k:order.append('cancel_cpu'),wait_idle=lambda **k:order.append('cpu_drained')))
        queue.canonical=queue.preprocessing=queue.final=object()
        queue.head_templates={'head':object()};queue.readers={0:object()}
        queue.finish();queue.finish()
        self.assertEqual(order,['copy_fence','reap','close','cancel_cpu','cpu_drained'])
        self.assertEqual(queue.window.specs,{})
        self.assertIsNone(queue.canonical);self.assertIsNone(queue.preprocessing);self.assertIsNone(queue.final)


if __name__=='__main__':unittest.main(verbosity=2)
