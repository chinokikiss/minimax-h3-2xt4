"""CPU contract gates for the production Sol attention integration."""
import math
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from patches.attention_schedule import parse_attention_schedule,select_attention,select_attention_policy,schedule_uses_backend
from patches import h3_operator_policy as policy
from patches import sol_attention
from patches import parallel_vae


ROOT=Path(__file__).resolve().parents[1]


class SolProductionContracts(unittest.TestCase):
    def test_nfe_schedule_is_one_based_sparse_and_uses_default_for_gaps(self):
        schedule=parse_attention_schedule("1-2:kitchen,4:sol@0.75,6+:native")
        self.assertEqual(schedule,((1,2,'kitchen',None),(4,4,'sol',0.75),(6,None,'native',None)))
        self.assertEqual(select_attention(schedule,1,'sol'),'kitchen')
        self.assertEqual(select_attention(schedule,3,'sol'),'sol')
        self.assertEqual(select_attention(schedule,4,'kitchen'),'sol')
        self.assertEqual(select_attention(schedule,9,'kitchen'),'native')
        self.assertEqual(select_attention_policy(schedule,4,'kitchen',1.0),('sol',0.75))
        self.assertEqual(select_attention_policy(parse_attention_schedule('2-7:sol'),3,'kitchen',1.25),('sol',1.25))
        self.assertTrue(schedule_uses_backend(schedule,'sol'))
        self.assertTrue(schedule_uses_backend(schedule,'kitchen'))
        self.assertTrue(schedule_uses_backend(schedule,'native'))
        for bad in ("0:sol","2-1:sol","1-3:sol,3-4:kitchen","1:missing","1:kitchen@1.0","1:sol@-1"):
            with self.assertRaises((ValueError,TypeError)):
                parse_attention_schedule(bad)

    def test_runtime_rejects_overlapping_direct_schedule_tuple(self):
        from patches.comfy_tp_runtime import RuntimeConfig
        with self.assertRaises(ValueError):
            RuntimeConfig('te','dit',dit_attention_schedule=((1,3,'sol',None),(3,4,'kitchen',None)))

    def test_dit_protects_conditioning_but_not_generated_audio_or_video(self):
        layout=SimpleNamespace(seq_len=321,segments=[
            (0,7,'text'),(7,53,'ref_img'),(53,61,'audio'),(61,321,'video')])
        self.assertEqual(policy.protected_prefix({'minimax_h3_layout':layout},321),53)
        with self.assertRaises(ValueError):
            policy.protected_prefix({'minimax_h3_layout':layout},320)

    def test_native_output_buffer_is_contiguous_bhld(self):
        source=(ROOT/'patches/sol_sm75_backend.py').read_text(encoding='utf-8')
        run=source[source.index('def run('):]
        self.assertNotIn('torch.empty_like(v)',run)
        self.assertGreaterEqual(run.count('torch.empty((batch, heads, length, head_dim)'),2)

    def test_native_route_protection_is_bidirectional(self):
        source=(ROOT/'patches/sol_attention_native/sol_sm75_preprocess.cuh').read_text()
        self.assertIn('protected_key',source)
        self.assertIn('protected_query',source)
        self.assertIn('neighbor || protected_key || protected_query',source)

    def test_sol_workspace_pool_is_stream_local_and_does_not_pin_qkv_payloads(self):
        backend=(ROOT/'patches/sol_sm75_backend.py').read_text(encoding='utf-8')
        attention=(ROOT/'patches/sol_attention.py').read_text(encoding='utf-8')
        self.assertIn('_WORKSPACE_POOL = {}',backend)
        self.assertIn('torch.cuda.current_stream(device).cuda_stream',backend)
        self.assertIn('workspace, "exact_mask"',backend)
        self.assertIn('workspace, "lut"',backend)
        # Dominant Q/K/V INT8 payloads deliberately remain call-owned.
        partition=backend[backend.index('def prepare_partition('):backend.index('def quantize(')]
        self.assertIn('q8=torch.empty(q.shape',partition)
        self.assertIn('k8=torch.empty(k.shape',partition)
        self.assertIn('class _PreparedSol(dict):',attention)
        self.assertIn('workspace_lease=workspace',attention)

    def test_workspace_lease_reentrance_and_reuse(self):
        from patches import sol_sm75_backend as backend
        # CPU tensors exercise ownership and allocation invariants without
        # pretending to qualify the SM75 native GPU path.
        backend.clear_workspace_pool()
        with patch.object(backend.torch.cuda, "current_stream",
                          return_value=SimpleNamespace(cuda_stream=493)):
            first=backend._acquire_workspace(torch.device("cpu"))
            first_owner=first.entry
            a=backend._workspace_tensor(
                first,"exact_mask",(2,4),torch.uint8,torch.device("cpu"))
            second=backend._acquire_workspace(torch.device("cpu"))
            b=backend._workspace_tensor(
                second,"exact_mask",(2,4),torch.uint8,torch.device("cpu"))
            self.assertIsNot(a,b)
            self.assertIsNot(first.entry,second.entry)
            first.release()
            reused=backend._acquire_workspace(torch.device("cpu"))
            self.assertIs(reused.entry,first_owner)
            self.assertIs(
                backend._workspace_tensor(
                    reused,"exact_mask",(2,4),torch.uint8,torch.device("cpu")),a)
            self.assertEqual(reused.entry["reuses"],1)
            backend.clear_workspace_pool()
            self.assertEqual(len(backend.workspace_pool_snapshot()),2)
            second.release()
            reused.release()
            backend.clear_workspace_pool()
            self.assertEqual(backend.workspace_pool_snapshot(),[])

    def test_workspace_same_stream_execution_and_released_lease(self):
        from patches import sol_sm75_backend as backend
        backend.clear_workspace_pool()
        with patch.object(backend.torch.cuda,"current_stream",
                          return_value=SimpleNamespace(cuda_stream=731)):
            lease=backend._acquire_workspace(torch.device("cpu"))
            backend.validate_workspace_execution(lease,torch.device("cpu"))
        with patch.object(backend.torch.cuda,"current_stream",
                          return_value=SimpleNamespace(cuda_stream=732)):
            with self.assertRaisesRegex(RuntimeError,"preparation CUDA stream"):
                backend.validate_workspace_execution(lease,torch.device("cpu"))
        lease.release()
        with patch.object(backend.torch.cuda,"current_stream",
                          return_value=SimpleNamespace(cuda_stream=731)):
            with self.assertRaisesRegex(RuntimeError,"already released"):
                backend.validate_workspace_execution(lease,torch.device("cpu"))
        backend.clear_workspace_pool()

    def test_prepared_sol_rejects_execution_after_release(self):
        from patches import sol_sm75_backend as backend
        with patch.object(backend.torch.cuda,"current_stream",
                          return_value=SimpleNamespace(cuda_stream=731)):
            lease=backend._acquire_workspace(torch.device("cpu"))
        prepared=sol_attention._PreparedSol(
            {"_h3_backend":"sol","q_int8":torch.empty((1,1,2,64),dtype=torch.int8)},
            workspace_lease=lease)
        prepared.release()
        with self.assertRaisesRegex(RuntimeError,"already been released"):
            sol_attention.execute_prepared(prepared)
        backend.clear_workspace_pool()

    def test_bitmap_lut_encodes_identical_ordered_indices(self):
        from benchmarks.sol_bitmap_route_probe import valid_lut_sha256
        for key_blocks in (1, 31, 32, 33, 64, 1019):
            words=(key_blocks+31)//32
            rows=3
            mask=torch.zeros((rows,key_blocks),dtype=torch.uint8)
            mask[0]=1 # All exact, including final partial word.
            mask[1,::3]=1
            mask[2,-1]=1 # Includes final bit of last word.
            counts=mask.sum(dim=1).to(torch.int32)
            bitmap=torch.zeros((rows,words),dtype=torch.int32)
            ordered=torch.zeros((rows,key_blocks),dtype=torch.int32)
            for row in range(rows):
                positions=torch.nonzero(mask[row]).flatten().tolist()
                ordered[row,:len(positions)]=torch.tensor(positions,dtype=torch.int32)
                for pos in positions:
                    word=pos//32
                    value=int(bitmap[row,word]) & 0xffffffff
                    value|=1<<(pos%32)
                    bitmap[row,word]=value if value<2**31 else value-2**32
            self.assertEqual(
                valid_lut_sha256(ordered,counts),
                valid_lut_sha256(bitmap,counts,bitmap=True,key_blocks=key_blocks))
            self.assertLessEqual(bitmap.numel(),ordered.numel())

    def test_bitmap_kernel_bit_enumerator_preserves_key_order(self):
        # Simulates the CUDA bitmap_word_index / bitmap_pending / __ffs loop.
        for length in (31,32,33,64,1019):
            words=(length+31)//32
            for expected in (
                list(range(length)), list(range(0,length,7)), [length-1],
                [0,length-1] if length>1 else [0],
            ):
                bitmap=[0]*words
                for key in expected:
                    bitmap[key//32]|=1<<(key%32)
                word_index=0
                pending=0
                decoded=[]
                for _ in range(len(expected)):
                    while pending==0 and word_index<words:
                        pending=bitmap[word_index]
                        word_index+=1
                    self.assertNotEqual(pending,0)
                    low_bit=(pending&-pending).bit_length()-1
                    decoded.append((word_index-1)*32+low_bit)
                    pending&=pending-1
                self.assertEqual(decoded,expected)

    def test_bitmap_mask_is_exactly_recoverable(self):
        from benchmarks.sol_bitmap_route_probe import decode_bitmap_mask
        key_blocks=1019
        rows=5
        torch.manual_seed(142)
        original=(torch.rand(rows,key_blocks)>.59).to(torch.uint8)
        original[0]=1
        original[1]=0
        words=torch.zeros((rows,(key_blocks+31)//32),dtype=torch.int32)
        for row in range(rows):
            for pos in torch.nonzero(original[row]).flatten().tolist():
                slot=pos//32
                value=(int(words[row,slot])&0xffffffff)|(1<<(pos%32))
                words[row,slot]=value if value<2**31 else value-2**32
        torch.testing.assert_close(decode_bitmap_mask(words,key_blocks),original,
                                   rtol=0,atol=0)

    def test_bitmap_lut_native_dispatch_is_separate_from_abi12_baseline(self):
        backend=(ROOT/'patches/sol_sm75_backend.py').read_text(encoding='utf-8')
        native=(ROOT/'patches/sol_attention_native/sla_sm75_sparse.cu').read_text()
        kernel=(ROOT/'patches/sol_attention_native/sla_sm75_fp16_kernel.cuh').read_text()
        pack=(ROOT/'patches/sol_attention_native/sol_sm75_preprocess.cuh').read_text()
        self.assertIn('star7_sol_sm75_pack_bitmap_lut',native)
        self.assertIn('star7_sol_sm75_launch_all_int8_complete_bitmap',native)
        self.assertIn('star7_sol_sm75_launch_all_int8_complete(',native)
        self.assertIn('bool LUT_BITMAP = false',kernel)
        self.assertIn('while (bitmap_pending == 0 && bitmap_word_index < lut_stride)',kernel)
        self.assertIn('bitmap_pending &= bitmap_pending - 1u;',kernel)
        self.assertIn('__global__ void pack_bitmap_lut(',pack)
        self.assertIn('bitmap_lut: bool = False',backend)
        wrapper=(ROOT/'patches/sol_attention.py').read_text(encoding='utf-8')
        self.assertIn('bitmap_lut=bool(prepared.get("bitmap_lut",False))',wrapper)
        self.assertIn('None if routing.get("bitmap_lut",False)',wrapper)
        self.assertIn('0 if bitmap_lut else exact_mask.data_ptr()',backend)
        self.assertIn('None if bitmap_lut else workspace',backend)

    def test_kc_fusion_has_been_retired(self):
        kernel=(ROOT/'patches/sol_attention_native/sol_sm75_preprocess.cuh').read_text()
        self.assertNotIn('compute_kc_stats_threshold_fused',kernel)
        self.assertNotIn('H3_SOL_FUSED_KC_THRESHOLD',kernel)
        self.assertGreaterEqual(kernel.count('reduce_kc_stats<HEAD_DIM><<<'),2)

    def test_d64_uses_native_shape_without_padding(self):
        q=torch.randn(1,2,11,64,dtype=torch.float16)
        k=torch.randn_like(q);v=torch.randn_like(q)
        captured={}
        def fake(q64,k64,v64,**kwargs):
            captured.update(q=q64,k=k64,v=v64,**kwargs)
            return v64+torch.ones_like(v64)
        with patch.object(sol_attention,'_sm75',return_value=None),patch.object(sol_attention,'_native',side_effect=fake):
            out=sol_attention.attention(q,k,v,protected_tokens=5,protected_start=6,allow_d64=True)
        self.assertEqual(tuple(captured['q'].shape),(1,2,11,64))
        self.assertIs(captured['q'],q);self.assertIs(captured['k'],k);self.assertIs(captured['v'],v)
        self.assertEqual(captured['protected_tokens'],5)
        self.assertEqual(captured['protected_start'],6)
        self.assertIsNone(captured['topology'])
        torch.testing.assert_close(out,v+1,rtol=0,atol=0)

    def test_sol_v2_uses_exact_836_token_frame_stride_and_audio_prefix(self):
        layout=SimpleNamespace(
            seq_len=65166,
            segments=[(0,2528,'text'),(2528,3344,'ref_img'),(3344,4164,'ref_img'),
                      (4164,4974,'audio'),(4974,65166,'video')],
            signature=(2528,72,44,76,405),
        )
        prefix,topology=policy.sol_v2_routing(
            {'minimax_h3_layout':layout,'_h3_sol_temporal_radius':1,
             '_h3_sol_local_radius_tokens':64},65166)
        self.assertEqual(prefix,4974)
        self.assertEqual(topology['start_token'],4974)
        self.assertEqual(topology['end_token'],65166)
        self.assertEqual(topology['frame_stride_tokens'],836)
        self.assertEqual(topology['temporal_radius'],1)
        self.assertEqual(topology['local_radius_tokens'],64)
        self.assertNotEqual(topology['frame_stride_tokens'],13*64)

    def test_query_partitioned_video_uses_local_q_full_kv_without_sol(self):
        source=(ROOT/'patches/parallel_vae.py').read_text()
        self.assertNotIn("run.exchange(q,sizes,'q',2)",source)
        self.assertNotIn("run.exchange(q,sizes,'q',3)",source)
        self.assertNotIn("full_q=torch.zeros_like(full_k)",source)
        self.assertNotIn("full_q[:,:,lo:hi].copy_(q)",source)
        self.assertNotIn("from patches.sol_attention import attention_partition",source)
        self.assertIn("out=int8_attention(q,full_k,full_v)",source)

    def test_sol_partition_tail_global_suffix_contract(self):
        # 1024 content tokens + five global suffix tokens. The physical split
        # must remain Q64-aligned while rank 1 owns the final partial block.
        ranges=parallel_vae.aligned_ranges(1029)
        self.assertEqual(ranges,((0,640),(640,1029)))
        first=(ranges[1][0]//64)
        count=(ranges[1][1]-ranges[1][0]+63)//64
        self.assertEqual((first,count,first+count-1),(10,7,16))
        source=(ROOT/'patches/sol_attention_native/sol_sm75_preprocess.cuh').read_text()
        self.assertIn("const int global_q_block=q_block_base+q_block;",source)
        self.assertIn("global_q_block>=sink_start_block&&global_q_block<sink_end_block",source)
        kernel=(ROOT/'patches/sol_attention_native/sla_sm75_fp16_kernel.cuh').read_text()
        self.assertIn("const uint32_t storage_q_block = q_block - q_storage_block_base;",kernel)
        self.assertIn("const uint32_t route_q_block = q_block - route_block_base;",kernel)

    def test_removed_te_group32_cannot_be_selected(self):
        self.assertFalse((ROOT/'patches/tilelang_group32_masked.py').exists())
        self.assertFalse((ROOT/'patches/tilelang_te_preparation.py').exists())
        with self.assertRaises(ValueError):
            policy.H3OperatorPolicy(te_attention='tilelang_group32')

    def test_video_sol_has_been_retired(self):
        from patches.comfy_tp_runtime import RuntimeConfig
        with self.assertRaisesRegex(ValueError,'Unknown Video VAE attention'):
            RuntimeConfig('te','dit',video_attention='sol')

    def test_video_native_forces_sdpa(self):
        q=torch.randn(1,2,19,64,dtype=torch.float16)
        k=torch.randn_like(q);v=torch.randn_like(q)
        owner=SimpleNamespace(_h3_attention_backend='native')
        with patch('torch.nn.functional.scaled_dot_product_attention',return_value=v) as call:
            out=parallel_vae._video_attention(owner,q,k,v)
        self.assertIs(out,v)
        call.assert_called_once_with(q,k,v)

    def test_complete_sol_native_launcher_supports_query_block_subset(self):
        source=(ROOT/'patches/sol_attention_native/sla_sm75_sparse.cu').read_text()
        self.assertIn("int q_block_base, int q_block_count",source)
        self.assertIn("dim3 grid(q_block_count, heads, batch)",source)
        self.assertIn("length, lut_stride, q_block_base",source)

    def test_sol_prepared_and_fallback_defaults_match_v2(self):
        source=(ROOT/'patches/h3_operator_policy.py').read_text()
        self.assertGreaterEqual(source.count("options.get('_h3_sol_topology_v2',True)"),2)
        self.assertGreaterEqual(source.count("options.get('_h3_sol_sync_free_lut',True)"),2)
        self.assertGreaterEqual(source.count("options.get('_h3_sol_collect_route_stats',False)"),2)

    def test_runtime_routes_sol_through_aot_prepare(self):
        source=(ROOT/'patches/comfy_tp_runtime.py').read_text(encoding='utf-8')
        self.assertIn("sol=use_sol",source)
        self.assertNotIn("stage='video',sol=True",source)
        self.assertNotIn("from patches.sol_attention import prepare_native",source)

    def test_sol_aot_prepare_records_native_library(self):
        from patches import h3_operator_aot as aot
        aot.prepare.cache_clear()
        fake_env=dict(compiler='nvcc',compiler_version='release 12.8',python='x',
            torch='x',cuda='12.8',tilelang='0.1.13',arch='sm_75')
        with patch.object(aot.torch.cuda,'is_available',return_value=True),                 patch.object(aot.torch.cuda,'get_device_capability',return_value=(7,5)),                 patch.object(aot,'environment',return_value=fake_env),                 patch.object(aot,'plan',return_value=[]),                 patch.object(sol_attention,'prepare_native',return_value='/tmp/sol.so') as prepare:
            result=aot.prepare(1,0,policy.H3OperatorPolicy(),stage='dit',sol=True)
        self.assertTrue(result['sol_precompiled'])
        self.assertEqual(result['sol_library'],'/tmp/sol.so')
        self.assertTrue(result['complete_default_custom_coverage'])
        prepare.assert_called_once_with()
        aot.prepare.cache_clear()

    def test_sol_is_opt_in_not_default(self):
        from patches.comfy_tp_runtime import RuntimeConfig
        p=policy.H3OperatorPolicy()
        self.assertEqual(p.dit_attention,'kitchen')
        self.assertEqual(RuntimeConfig('te','dit',video_attention='native').video_attention,'native')
        self.assertEqual(RuntimeConfig('te','dit',sol_tau=.8).sol_tau,.8)
        production=RuntimeConfig('te','dit',sol_topology_v2=False,sol_sync_free_lut=False,
            sol_collect_route_stats=False,sol_temporal_radius=2,sol_local_radius_tokens=32)
        self.assertTrue(production.sol_topology_v2 and production.sol_sync_free_lut)
        self.assertEqual((production.sol_temporal_radius,production.sol_local_radius_tokens),(2,32))
        with self.assertRaises(ValueError):RuntimeConfig('te','dit',sol_tau=-.1)
        self.assertNotIn('sol',('kitchen','native'))


if __name__=='__main__':
    unittest.main(verbosity=2)

