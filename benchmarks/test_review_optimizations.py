"""Review follow-up: numerical, integrity, ownership and admission contracts."""
from contextlib import nullcontext
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch
from safetensors.torch import save_file
import comfy_kitchen as ck
from patches.dit_column_linear import column_linear
from patches.windowed_tp import CheckpointReader
from patches.comfy_tp_runtime import RuntimeConfig, worker_reply
from patches.h3_lora import H3LoRAStore


class ColumnDestination(unittest.TestCase):
    @torch.no_grad()
    def test_te_shares_only_identical_preparation_and_preserves_projection_hooks(self):
        from comfy_kitchen.backends.eager import quantization as eager
        from patches.tp_qwen3vl import ColumnParallelLinearINT8,project_columns
        torch.manual_seed(2401)
        modules=[ColumnParallelLinearINT8(5120,n) for n in (8192,1024,1024)]
        for module in modules:
            module.weight.random_(-127,127);module.weight_scale.fill_(.0002)
        x=torch.randn(1,5,5120,dtype=torch.float16)
        with ck.registry.use_backend('eager'):
            expected=tuple(module(x) for module in modules)
            with patch.object(eager,'_rotate_activation',wraps=eager._rotate_activation) as rotate, \
                    patch.object(eager,'quantize_int8_rowwise',wraps=eager.quantize_int8_rowwise) as quant:
                actual=project_columns(x,*modules)
                self.assertEqual(rotate.call_count,1);self.assertEqual(quant.call_count,1)
                self.assertTrue(all(torch.equal(a,b) for a,b in zip(actual,expected)))
            calls=[];hook=modules[1].register_forward_hook(lambda *args:calls.append(True))
            try:
                with patch.object(eager,'quantize_int8_rowwise',wraps=eager.quantize_int8_rowwise) as quant:
                    actual=project_columns(x,*modules)
                    self.assertEqual(quant.call_count,3)
                    self.assertEqual(calls,[True])
                    self.assertTrue(all(torch.equal(a,b) for a,b in zip(actual,expected)))
            finally:hook.remove()

    @torch.no_grad()
    def test_chunked_destination_is_exact_without_output_concatenation(self):
        from comfy_kitchen.backends.eager import quantization as eager
        from patches import dit_column_linear as column
        torch.manual_seed(2410)
        x=torch.randn(29,64);weight=torch.randint(-127,128,(96,64),dtype=torch.int8)
        # Nonuniform scales exercise both FP32 multiplications and FP16 rounding.
        scale=torch.logspace(-5,-1,96).reshape(-1,1)
        for dtype in (torch.float16,torch.bfloat16,torch.float32):
            with ck.registry.use_backend('eager'):
                expected=ck.int8_linear(x,weight,scale,None,dtype,convrot=False)
                original=eager._int8_matmul_accumulate
                def multiply(*args,**kwargs):
                    value=original(*args,**kwargs)
                    # Quantization/GEMM may legitimately concatenate internally.
                    # The dequantization/output stage must not retain/cat parts.
                    self.enterContext(patch('torch.cat',side_effect=AssertionError('output cat')))
                    return value
                with patch.object(column,'_DEQUANT_CHUNK_BYTES',96*4*3), \
                        patch.object(eager,'_int8_matmul_accumulate',side_effect=multiply):
                    actual=column_linear(x,weight,scale,convrot=False,out_dtype=dtype)
                self.assertTrue(torch.equal(actual,expected),dtype)
                self.doCleanups()


class CheckpointRanges(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.path=Path(self.temp.name)/'weights.safetensors'
        self.tensor=torch.arange(256*512,dtype=torch.int32).reshape(256,512)
        save_file({'w':self.tensor},str(self.path))

    def test_owned_reads_preserve_order_without_read_amplification(self):
        reader=CheckpointReader(self.path)
        rows=[(180,256),(3,11),(0,3)]
        result=reader.read_tensor('w',rows=rows,columns=(128,384))
        expected=torch.cat([self.tensor[a:b,128:384] for a,b in rows])
        self.assertTrue(torch.equal(result,expected))
        self.assertEqual(reader.logical_read_bytes,result.numel()*4)
        self.assertEqual(reader.source_read_bytes,reader.logical_read_bytes)
        self.assertEqual(reader.range_read_calls,87)

    def test_full_width_columns_need_no_extra_source_bytes_or_row_syscalls(self):
        reader=CheckpointReader(self.path)
        self.assertTrue(torch.equal(reader.read_tensor('w',columns=(0,512)),self.tensor))
        self.assertEqual(reader.range_read_calls,1)
        self.assertEqual(reader.source_read_bytes,reader.logical_read_bytes)

    def test_short_positional_reads_then_seek_fallback_and_eof(self):
        reader=CheckpointReader(self.path)
        calls=0
        raw=self.path.read_bytes()
        def positional(fd,size,offset):
            nonlocal calls
            calls+=1
            if calls==2:raise OSError('non-positional descriptor')
            return raw[offset:offset+min(size,7)]
        with patch('patches.windowed_tp.os.pread',side_effect=positional,create=True):
            result=reader.read_tensor('w',rows=[(1,9)],columns=(256,512))
        self.assertTrue(torch.equal(result,self.tensor[1:9,256:512]))
        with patch('patches.windowed_tp.os.pread',return_value=b'',create=True):
            with self.assertRaises(IOError):reader.read_tensor('w',columns=(256,512))

    def test_removed_chunk_option_and_mutation_are_rejected(self):
        with self.assertRaises(TypeError):CheckpointReader(self.path,column_read_chunk_bytes=8192)
        reader=CheckpointReader(self.path)
        self.path.write_bytes(self.path.read_bytes()+b'changed')
        with self.assertRaisesRegex(RuntimeError,'changed'):reader.read_tensor('w',columns=(256,512))


class ReplyOwnership(unittest.TestCase):
    def test_unused_rank_has_no_conversion_diagnostics_still_return(self):
        with patch('patches.comfy_tp_runtime.packet',side_effect=lambda value,device:value) as convert:
            value=object()
            for command in ('te','refine','denoise','close','video_pixels'):
                self.assertIsNone(worker_reply(value,1,command))
            convert.assert_not_called()
            self.assertIs(worker_reply(value,1,'stats'),value)
            self.assertIs(worker_reply(value,1,'idle_stats'),value)
            self.assertIs(worker_reply(value,0,'te'),value)
            self.assertEqual(convert.call_count,3)

    def test_returned_cpu_tensor_is_owned_copy(self):
        x=torch.arange(7)
        out=worker_reply({'x':x},0,'te')
        self.assertTrue(torch.equal(out['x'],x))
        self.assertNotEqual(out['x'].data_ptr(),x.data_ptr())


class LoaderControls(unittest.TestCase):
    def test_column_only_policy_still_requires_the_pinned_compiler_library(self):
        from patches.h3_operator_policy import H3OperatorPolicy,configure_block
        from patches.h3_operator_dependencies import validate_dependencies
        policy=H3OperatorPolicy(te_attention='native',dit_attention='native',row_linear_backend='native',
                               column_linear_backend='cuda_int8_tilelang')
        with patch('patches.h3_operator_dependencies.version',side_effect=lambda name:
                   {'comfy-kitchen':'0.2.37','tilelang':'0.1.15'}[name]):
            with self.assertRaisesRegex(RuntimeError,'0.1.13'):validate_dependencies(policy,t4=True)
        block=SimpleNamespace(**{name:SimpleNamespace() for name in ('qkv_proj','fc1','out_proj','fc2')})
        configure_block(block,'dit',policy)
        self.assertEqual(block.fc1.operator_policy.column_linear_backend,'cuda_int8_tilelang')
        configure_block(block,'dit',H3OperatorPolicy())
        self.assertEqual(block.fc1.operator_policy.column_linear_backend,'cuda_int8_tilelang')

    def test_loader_defaults_and_explicit_resource_controls(self):
        import custom_nodes.minimax_h3_tp2 as nodes
        options=nodes.MiniMaxH3TP2Loader.INPUT_TYPES()['optional']
        for removed in ('tp_quantization','swiglu_padding','selective_b','checkpoint_read_chunk_mib'):
            self.assertNotIn(removed,options)
        def service(config):return SimpleNamespace(config=config,closed=False,close=Mock())
        with patch.object(nodes,'_service',None),patch.object(nodes,'PhysicalTPService',side_effect=service), \
                patch.object(nodes,'create_model'),patch.object(nodes,'create_clip'):
            loader=nodes.MiniMaxH3TP2Loader();one=loader.load('te','dit')[2]
            self.assertEqual(one.config,RuntimeConfig(str(Path('te').resolve()),str(Path('dit').resolve())))
            two=loader.load('te','dit',prefetch_lifecycle=True,memory_pressure_aware=True,
                dit_reserve_gib=8.,command_timeout_seconds=1200,
                lora_selective_qkv_b=True,rank_checkpoint_dir='rank_blocks')[2]
            self.assertFalse(two.config.idle_residency)
            self.assertTrue(two.config.prefetch_lifecycle and two.config.memory_pressure_aware)
            self.assertEqual(two.config.reserve_bytes,8*2**30)
            self.assertEqual(two.config.timeout_seconds,1200)
            self.assertTrue(two.config.lora_selective_qkv_b)
            self.assertEqual(two.config.rank_checkpoint_dir,str(Path('rank_blocks').resolve()))
            one.close.assert_called_once()

    def test_invalid_resource_controls_rejected(self):
        config=RuntimeConfig('te','dit')
        for kwargs in [dict(timeout_seconds=0),dict(reserve_bytes=-1)]:
            with self.assertRaises(ValueError):replace(config,**kwargs)

    def test_qkv_owned_rows_are_explicit_and_geometry_guarded(self):
        store=H3LoRAStore.__new__(H3LoRAStore);store.selective_b=True;store.selective_qkv_b=False
        target='blocks.0.attn.qkv_proj'
        header={'a':{'shape':[128,5376]},'b':{'shape':[21504,128]}}
        store.sources=[(SimpleNamespace(header=header),1.,{target:('a','b',1.)})]
        self.assertIsNone(store.selective_rows(target,0))
        store.selective_qkv_b=True
        for rank in (0,1):
            self.assertEqual(store.selective_rows(target,rank),
                [(c*7168+rank*3584,c*7168+(rank+1)*3584) for c in range(3)])
        header['a']['shape']=[64,5376]
        self.assertIsNone(store.selective_rows(target,0))
        self.assertIsNone(store.selective_rows('token_refiner.blocks.0.attn.qkv_proj',0))

    @torch.no_grad()
    def test_selected_qkv_adapter_packs_each_component_and_preserves_scale_order(self):
        from patches.h3_lora import NativeAlignedLoRA
        torch.manual_seed(1901)
        target='blocks.0.attn.qkv_proj'
        values={'a':torch.randn(128,5376)*.02,'b':torch.randn(21504,128)*.02}
        header={key:{'shape':list(t.shape)} for key,t in values.items()}
        def read(name,*,columns=None,rows=None):
            value=values[name]
            if rows is not None:value=torch.cat([value[a:b] for a,b in rows])
            return value
        reader=SimpleNamespace(header=header,read_tensor=read)
        store=H3LoRAStore.__new__(H3LoRAStore)
        store.selective_b=True;store.selective_qkv_b=True;store.row_partial=False;store.check=lambda:None
        store.sources=[(reader,.7,{target:('a','b',.7)}),(reader,-.25,{target:('a','b',-.25)})]
        x=torch.randn(11,5376)
        for rank in (0,1):
            rows=store.selective_rows(target,rank)
            selected=read('b',rows=rows)
            reference=NativeAlignedLoRA([(values['a'],selected,.7),(values['a'],selected,-.25)])
            adapter=store.adapter(target,rank)
            initial=torch.randn(11,10752)
            self.assertTrue(torch.equal(adapter.add_to(x,initial.clone()),reference.add_to(x,initial.clone())))
            self.assertEqual(store.prefix_bytes('blocks.0.',rank),2*4*(128*5376+10752*128))
            self.assertTrue(adapter.selective_b)


class LazyVideoParent(unittest.TestCase):
    def setUp(self):
        from benchmarks.test_video_decode_memory import native_fixture
        from patches.staged_vae import StagedVAE,video_planning_template
        self.real_template=video_planning_template
        self.source,self.raw=native_fixture()
        self.service=SimpleNamespace(lock=threading.RLock(),closed=False,call=Mock(),
                                     config=SimpleNamespace(idle_residency=False))
        self.staged=StagedVAE.__new__(StagedVAE)
        self.staged.kind='video';self.staged.service=self.service;self.staged.lock=threading.RLock()
        self.staged.stage=None;self.staged.reservation=0;self.staged.reader=SimpleNamespace()
        self.staged.events=[];self.staged.check_identity=Mock()
        self.staged.vae=SimpleNamespace(first_stage_model=None,patcher=None,size=None,
            device=torch.device('cuda:0'),vae_dtype=torch.float16,memory_used_decode=lambda *args:1024)
        self.staged.vae.decode=lambda z:self.staged.vae.first_stage_model.decode(z)
        self.parent_loads=[]
        def ensure(stage):
            if self.staged.stage==stage:return
            self.assertIn('vae_prepare',[c.args[0] for c in self.service.call.call_args_list])
            self.parent_loads.append(stage)
            self.staged.stage=stage;self.staged.vae.first_stage_model=self.source
        self.ensure=Mock(side_effect=ensure);self.staged._ensure=self.ensure
        self.enterContext(patch('patches.staged_vae.video_planning_template',return_value=self.source))
        self.enterContext(patch('patches.parallel_vae.query_partitioning_status',return_value={'available':True}))
        self.enterContext(patch('comfy.model_management.cuda_device_context',side_effect=lambda *args:nullcontext()))
        self.enterContext(patch('comfy.model_management.unload_model_and_clones'))
        self.enterContext(patch('comfy.model_management.archive_model_dtypes'))
        self.enterContext(patch('comfy.model_patcher.CoreModelPatcher',
            side_effect=lambda model,**kwargs:SimpleNamespace(model=model)))
        self.enterContext(patch('comfy.model_management.intermediate_device',return_value=torch.device('cpu')))
        self.enterContext(patch('torch.cuda.synchronize'))

    def wrapper(self,accepted=True,fail=False):
        from patches.comfy_sampler import ParallelDecodeHandoff
        def call(command,*args):
            if command=='vae_prepare':
                self.ensure.assert_not_called()
                self.assertEqual(self.staged.reservation,0)
                self.assertEqual(sum(p.numel() for p in self.staged.vae.patcher.model.parameters()),0)
                return dict(accepted=accepted,reason='fixture host admission')
            if command=='video_pixels':
                if fail:raise RuntimeError('remote failure')
                return self.raw(args[0])
        self.service.call.side_effect=call
        return ParallelDecodeHandoff(self.staged,self.service,'fixture')

    def test_success_never_loads_parent_component_and_keeps_native_plan_exact(self):
        wrapper=self.wrapper();z=torch.randn(1,24,7,2,2)
        expected=self.source.decode(z)
        self.assertTrue(torch.equal(wrapper.decode(z),expected))
        self.ensure.assert_not_called()
        self.assertIsNone(self.staged.vae.first_stage_model)
        self.assertIsNone(self.staged.stage)
        self.assertFalse(self.staged.remote_decode_active)
        self.assertEqual(wrapper.last_strategy,'native_aligned_query')

    def test_refusal_loads_parent_only_after_remote_lease_is_released(self):
        wrapper=self.wrapper(accepted=False);z=torch.randn(1,24,1,2,2)
        self.assertTrue(torch.equal(wrapper.decode(z),self.source.decode(z)))
        self.ensure.assert_called_with('decode')
        self.assertEqual(self.parent_loads,['decode'])
        self.assertEqual(wrapper.last_strategy,'native')
        self.assertNotIn('video_pixels',[c.args[0] for c in self.service.call.call_args_list])

    def test_error_releases_workers_and_restores_empty_parent_without_fallback(self):
        wrapper=self.wrapper(fail=True)
        with self.assertRaisesRegex(RuntimeError,'remote failure'):wrapper.decode(torch.randn(1,24,1,2,2))
        self.ensure.assert_not_called()
        self.assertIsNone(self.staged.vae.first_stage_model)
        self.assertFalse(self.staged.remote_decode_active)
        self.assertEqual(self.service.call.call_args_list[-1].args[0],'vae_release')

    def test_real_metadata_template_reads_no_learned_payload(self):
        reads=[]
        reader=SimpleNamespace(header={'latents_mean':{},'latents_std':{}})
        def read(name):
            reads.append(name)
            return torch.arange(24,dtype=torch.float32)/100+1
        reader.read_tensor=read
        model=self.real_template(reader)
        self.assertEqual(reads,['latents_mean','latents_std'])
        self.assertTrue(all(p.is_meta for p in model.parameters()))
        self.assertEqual(model.latents_std.dtype,torch.float16)
        self.assertEqual(model.pixel_mean.device.type,'cpu')


if __name__=='__main__':unittest.main(verbosity=2)
