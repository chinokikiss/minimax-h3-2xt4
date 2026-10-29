"""Known-sequence eviction, concurrent preparation and static-fold contracts."""
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
import tempfile
import threading
import unittest
from unittest.mock import patch

import torch
from torch import nn

from patches.host_weight_cache import HostBudget, HostWeightCache, _Cached
from patches.weight_residency import WeightSpec, ResidencyWindow
from patches.ptd.refiner_layout import write_layout, read_layout_tensor
from patches.h3_tensor_flow import TensorFlowPolicy


def item(key, prepare=None):
    def create():
        module=nn.Module();module.register_buffer('weight',torch.ones(16,dtype=torch.float32))
        return module
    return WeightSpec(key,prepare or create,64,prepare_peak_bytes=128)


class ScheduleContracts(unittest.TestCase):
    def cache(self, capacity=512):
        budget=HostBudget(4096,reserve_bytes=0)
        cache=HostWeightCache(budget,capacity,0)
        self.addCleanup(cache.close)
        return cache

    def load(self,cache,keys):
        cache.prefetch([item(k) for k in keys]);cache.wait_idle(5)

    def test_repeat_use_survives_hop_boundary_and_safe_hold(self):
        cache=self.cache();cache.set_access_plan(('expert','other','expert'))
        self.load(cache,['expert','other'])
        cache.set_access_plan(('expert','other','expert'),1)
        self.assertEqual(cache.remaining_uses['expert'],1)
        self.assertIn('expert',cache.entries)
        cache.hold('expert');cache.set_access_plan(('expert','other','expert'),3)
        self.assertIn('expert',cache.entries);self.assertNotIn('other',cache.entries)
        cache.release_hold('expert')
        self.assertFalse(cache.entries);self.assertEqual(cache.stats['consumed_releases'],2)
        cache.reset_access_plan();self.assertEqual(cache.budget.snapshot()['reserved_bytes'],0)

    def test_farthest_future_eviction_protects_near_and_active(self):
        cache=self.cache();sequence=('a','b','c','a','d','b')
        cache.set_access_plan(sequence);self.load(cache,['a','b','c','d'])
        cache.near_keys={'a','b'};cache.hold('d')
        with cache.condition:self.assertTrue(cache._evict_one('new',True))
        self.assertNotIn('c',cache.entries);self.assertIn('d',cache.entries)
        cache.release_hold('d')
        with cache.condition:self.assertTrue(cache._evict_one('new',True))
        self.assertNotIn('d',cache.entries)
        with cache.condition:self.assertFalse(cache._evict_one('new',False))

    def test_pin_borrow_defers_consumed_reclamation(self):
        cache=self.cache();cache.set_access_plan(('a',));self.load(cache,['a'])
        entered,finish=threading.Event(),threading.Event();errors=[]
        def pin(module):
            entered.set();self.assertTrue(finish.wait(5));return nn.Identity()
        def consume():
            try:cache.pin(item('a'))
            except BaseException as error:errors.append(error)
        with patch('patches.host_weight_cache.pinned_module',side_effect=pin):
            thread=threading.Thread(target=consume);thread.start()
            try:
                self.assertTrue(entered.wait(5));cache.set_access_plan(('a',),1)
                self.assertIn('a',cache.entries)
            finally:finish.set();thread.join(5)
        self.assertFalse(errors);self.assertFalse(cache.entries)

    def test_window_event_owns_protection_without_worker_cuda_queries(self):
        cache=self.cache();cache.set_access_plan(('a',));self.load(cache,['a']);cache.hold('a')
        cache.set_access_plan(('a',),1)
        event=SimpleNamespace(done=False,query=lambda:event.done)
        window=object.__new__(ResidencyWindow)
        window.host_cache=cache;window.cpu_fences=[('a',event)]
        window.entries={};window.retired=[];window.observations=[];window.idle_pending=set()
        window._reap();self.assertIn('a',cache.entries)
        event.done=True;window._reap();window._reap()
        self.assertFalse(cache.entries);self.assertFalse(cache.holds)

    def test_teardown_releases_consumer_holds_before_joining_preparation(self):
        from contextlib import nullcontext
        order=[];event=SimpleNamespace(done=False)
        def complete():event.done=True;order.append('consumer_complete')
        event.query=lambda:event.done;event.synchronize=complete
        window=object.__new__(ResidencyWindow)
        window.closed=False;window.idle_cancel=None;window.future_cpu=True;window.device='cuda:0'
        window.host_cache=SimpleNamespace(release_hold=lambda key:order.append('release_hold'))
        window.cpu_fences=[('a',event)];window.entries={};window.retired=[];window.observations=[]
        window.idle_pending=set();window.pending={};window.unmaterialized_staging={};window.host_reservations={}
        window.stats={};window.pool=SimpleNamespace(shutdown=lambda **kw:order.append('join_prepare'))
        window.copy_stream=SimpleNamespace(synchronize=lambda:order.append('copy_complete'))
        with patch('torch.cuda.device',return_value=nullcontext()):window.close()
        self.assertLess(order.index('consumer_complete'),order.index('release_hold'))
        self.assertLess(order.index('release_hold'),order.index('join_prepare'))

    def test_pending_copy_source_is_not_prepared_again_after_cache_take(self):
        from concurrent.futures import Future
        from contextlib import nullcontext
        from unittest.mock import Mock
        cache=SimpleNamespace(budget=SimpleNamespace(pressure_aware=False),
            set_access_plan=Mock(),prefetch=Mock())
        with patch('torch.cuda.Stream',return_value=object()):
            window=ResidencyWindow((item('a'),item('refiner')),('a','refiner'),'cuda:0',
                max_weight_bytes=4096,pinned_bytes=4096,host_cache=cache)
        window.future_cpu=True;window.cpu_priority_by_use={1:'refiner'}
        window.pending={'a':Future(),'refiner':Future()}
        try:
            with patch('torch.cuda.device',return_value=nullcontext()):window.prefetch()
            self.assertEqual(cache.prefetch.call_args.args[0],[])
        finally:window.pool.shutdown()

    def test_priority_refiner_prepares_concurrently_with_current_block(self):
        cache=self.cache(128);cache.set_access_plan(('block','refiner'))
        cache.enable_priority_lane({'refiner'})
        block_started,refiner_started,finish=threading.Event(),threading.Event(),threading.Event()
        def prepare(started):
            started.set()
            if not finish.wait(5):raise TimeoutError('producer test timed out')
            return item('value').prepare()
        try:
            cache.prefetch([item('block',lambda:prepare(block_started)),item('refiner',lambda:prepare(refiner_started))])
            self.assertTrue(block_started.wait(5));self.assertTrue(refiner_started.wait(5))
            self.assertLessEqual(cache.budget.snapshot()['reserved_bytes'],256)
        finally:finish.set();cache.wait_idle(5)
        self.assertEqual(cache._bytes(),128)
        cache.set_access_plan(('block','refiner'),2);cache.reset_access_plan()
        # The same service can start another request; no abandoned lane/thread.
        cache.set_access_plan(('second',));cache.enable_priority_lane({'second'})
        self.load(cache,['second']);self.assertIn('second',cache.entries)
        cache.set_access_plan(('second',),1);cache.reset_access_plan()

    def test_two_lanes_cannot_overbook_final_cache_capacity(self):
        cache=self.cache(64);cache.set_access_plan(('block','refiner'))
        cache.enable_priority_lane({'refiner'})
        cache.prefetch([item('block')]);cache.wait_idle(5)
        # A farther speculative producer cannot displace the near block.
        with cache.condition:
            future=cache._queue(item('refiner'),False,deadline=1);cache.condition.notify_all()
        self.assertFalse(future.done());self.assertEqual(cache._bytes(),64)
        cache.set_access_plan(('block','refiner'),1)
        future.result(5);cache.wait_idle(5)
        self.assertEqual(cache._bytes(),64);self.assertIn('refiner',cache.entries)

    def test_invalid_rewind_and_unbalanced_protection_fail(self):
        cache=self.cache();cache.set_access_plan(('a','a'),1)
        with self.assertRaises(ValueError):cache.set_access_plan(('a','a'),0)
        with self.assertRaises(ValueError):cache.set_access_plan(('a','b'),1)
        with self.assertRaises(RuntimeError):cache.release_hold('a')

    def test_refiner_lane_and_other_rank_share_preparation_slots(self):
        budget=HostBudget(4096,reserve_bytes=0,pressure_aware=True,preparation_slots=2)
        budget.memory=SimpleNamespace(snapshot=lambda:{'available':False})
        left=HostWeightCache(budget,256,0);right=HostWeightCache(budget,128,0)
        self.addCleanup(right.close);self.addCleanup(left.close)
        left.set_access_plan(('block','refiner'));right.set_access_plan(('other_rank',))
        left.enable_priority_lane({'refiner'})
        finish=threading.Event();condition=threading.Condition();started=[]
        def prepare(name):
            with condition:started.append(name);condition.notify_all()
            if not finish.wait(5):raise TimeoutError('slot test timed out')
            return item(name).prepare()
        try:
            left.prefetch([item('block',lambda:prepare('block')),item('refiner',lambda:prepare('refiner'))])
            right.prefetch([item('other_rank',lambda:prepare('other_rank'))])
            with condition:self.assertTrue(condition.wait_for(lambda:len(started)==2,5))
            self.assertEqual(budget.snapshot()['preparing'],2)
            with condition:self.assertFalse(condition.wait_for(lambda:len(started)>2,.1))
        finally:finish.set();left.wait_idle(5);right.wait_idle(5)
        self.assertEqual(set(started),{'block','refiner','other_rank'})
        self.assertEqual(budget.snapshot()['preparing'],0)

    def test_complete_plan_is_installed_before_refiner_consumption(self):
        from unittest.mock import Mock
        from benchmarks.test_ptd_queue import native_tiny, empty_window
        from patches.h3_tensor_flow import prepare_dit
        from patches.ptd.queue import ExpertQueue
        model=native_tiny();flow=TensorFlowPolicy('fp16');prepare_dit(model,flow,rank=0)
        model.blocks=nn.ModuleList([nn.Identity()])
        ref=WeightSpec('ref0',lambda:nn.ModuleDict(),1024,resident_backing=True)
        project=WeightSpec('project',lambda:nn.ModuleDict(),128,resident_backing=True)
        block=WeightSpec('block0',lambda:nn.Identity(),64)
        window=empty_window((ref,project,block));window.cpu_fences=[];window.pinned_bytes=4*2**30
        cache=SimpleNamespace(begin_request=Mock(),set_access_plan=Mock(),enable_priority_lane=Mock())
        window.host_cache=cache
        header={f'final_layer.{name}.{attribute}':{'shape':list(getattr(getattr(model.final_layer,name),attribute).shape)}
            for name in ('video_out','audio_out') for attribute in ('weight','bias')}
        def reader(hop,**kw):return SimpleNamespace(identity=('hop',hop),header=header)
        preprocessing=SimpleNamespace(canonical={name:getattr(model,name) for name in
            ('condition_proj','token_refiner','video_patch_proj','audio_patch_proj')},refine_key='ref0',
            project_key='project',project_uses=1)
        runtime=SimpleNamespace(window=window,preprocessing=preprocessing,keys=['block0'],model=model,rank=0,
            readers={'dit':reader(0)},cache=cache,config=SimpleNamespace(layers=1,operators=None,tensor_flow=flow))
        adapter=SimpleNamespace(runtime=runtime,original=None,bank=SimpleNamespace(reader=reader),options=SimpleNamespace(
            future_cache=True,prefetch_refiner=True,baked_refiner=False,use_hop_refiner=True,use_pdd_heads=True))
        with patch('patches.ptd.queue.tp_specs',side_effect=lambda r,*a:[WeightSpec(('block',r.identity[1]),lambda:nn.Identity(),64)]):
            queue=ExpertQueue(adapter)
        self.assertEqual(set(queue.plans),set(range(8)))
        self.assertEqual(window.cursor,0);self.assertEqual(sum(window.remaining.values()),len(window.uses))
        cache.set_access_plan.assert_called_once_with(window.uses,0)
        self.assertEqual(cache.enable_priority_lane.call_args.args[0],{queue.refine_keys[i] for i in range(1,8)})

    def test_runtime_sidecar_matches_native_then_rejects_corruption_and_wrong_flow(self):
        flow=TensorFlowPolicy('fp16')
        values={f'token_refiner.blocks.0.{name}.weight':torch.linspace(-3,3,32).reshape(4,8).half()
            for name in ('attn.qkv_proj','attn.out_proj','mlp.fc1','mlp.fc2')}
        with tempfile.TemporaryDirectory() as directory:
            proof=write_layout(values,directory,'source-digest',flow)
            reader=SimpleNamespace(root=Path(directory),hop=0,bank=SimpleNamespace(hops=[(None,{
                'auxiliary':{'sha256':'source-digest','runtime_layout':proof}})]))
            for name,source in values.items():
                expected=source.float()
                if name.endswith(('attn.out_proj.weight','mlp.fc2.weight')):expected=expected/flow.divisor
                self.assertTrue(torch.equal(read_layout_tensor(reader,name,flow),expected))
            with self.assertRaises(ValueError):read_layout_tensor(reader,next(iter(values)),TensorFlowPolicy('legacy'))
            name=next(iter(values));proof['tensors'][name]['sha256']='0'*64
            with self.assertRaisesRegex(ValueError,'integrity'):read_layout_tensor(reader,name,flow)

    def test_baked_refiner_preserves_full_native_norm_and_projection_state(self):
        from benchmarks.test_ptd_queue import native_tiny
        from patches.h3_tensor_flow import prepare_dit
        from patches.ptd.queue import refiner_group
        from patches.ptd.nodes import MiniMaxH3TP2PTDAdapter
        self.assertIn('future_cache',MiniMaxH3TP2PTDAdapter.INPUT_TYPES()['optional'])
        flow=TensorFlowPolicy('fp16');model=native_tiny();prepare_dit(model,flow,rank=0)
        values={}
        for i,block in enumerate(model.token_refiner.blocks):
            for part,module in [('attn.qkv_proj',block.attn.base.qkv_proj),('attn.out_proj',block.attn.base.out_proj),
                                ('mlp.fc1',block.mlp.fc1),('mlp.fc2',block.mlp.fc2)]:
                values[f'token_refiner.blocks.{i}.{part}.weight']=torch.linspace(-.2,.3,module.weight.numel()).reshape(module.weight.shape).half()
        with tempfile.TemporaryDirectory() as directory:
            proof=write_layout(values,directory,'original-overlay',flow)
            aux=dict(sha256='original-overlay',runtime_layout=proof,verified_targets=[k.removesuffix('.weight') for k in values])
            reader=SimpleNamespace(root=Path(directory),hop=0,bank=SimpleNamespace(hops=[(None,{'auxiliary':aux})]),
                read_tensor=lambda name:values[name].clone())
            canonical={'condition_proj':model.condition_proj,'token_refiner':model.token_refiner}
            raw=refiner_group(canonical,reader,flow);baked=refiner_group(canonical,reader,flow,baked=True)
            self.assertEqual(set(raw.state_dict()),set(baked.state_dict()))
            for name,value in raw.state_dict().items():self.assertTrue(torch.equal(value,baked.state_dict()[name]),name)
            self.assertEqual(raw['token_refiner'].blocks[0].norm1.eps,baked['token_refiner'].blocks[0].norm1.eps)


if __name__=='__main__':unittest.main(verbosity=2)
