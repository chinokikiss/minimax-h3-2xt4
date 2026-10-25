"""Native full-width merge vs bounded bundle preparation; small CPU fixtures."""
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import torch
from torch import nn
from safetensors.torch import save_file
from patches.tp2_bundle import BundleReader,DESCRIPTOR,split_matrix,join_matrix,sha,DIT_LINEAR
from patches.rank_local_int8 import CONFIG,FORMAT,state_hashes
from patches.windowed_tp import CheckpointReader
from patches.h3_lora import H3LoRAStore
from patches.offline_lora import merged_weight
from patches.offline_lora_reference import verify_weight


def block(state):
    module=nn.Module()
    for name,tensor in state.items():
        owner=module; bits=name.split('.')
        for bit in bits[:-1]:
            if not hasattr(owner,bit):owner.add_module(bit,nn.Module())
            owner=getattr(owner,bit)
        owner.register_parameter(bits[-1],nn.Parameter(torch.zeros_like(tensor),requires_grad=False))
    module.load_state_dict(state,strict=True,assign=True)
    return module


class Bundle(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name); self.source=self.root/'base.safetensors'
        generator=torch.Generator().manual_seed(123)
        from comfy_kitchen.tensor.int8 import TensorWiseINT8Layout as L
        self.native={};self.local=[{},{}];self.quant={}
        for name,local in DIT_LINEAR.items():
            shape=(12 if local=='qkv_proj' else 8,512) if local!='adaln_proj' else (48,8)
            weight=torch.randn(shape,generator=generator)*.1
            if local!='adaln_proj':
                q,params=L.quantize(weight,per_channel=True,convrot=True,convrot_groupsize=256,stochastic_rounding=0)
                config=dict(format='int8_tensorwise',convrot=True,convrot_groupsize=256)
                self.quant['blocks.0.'+name+'.comfy_quant']=config
                self.native['blocks.0.'+name+'.comfy_quant']=torch.tensor(list(json.dumps(config).encode()),dtype=torch.uint8)
                self.native['blocks.0.'+name+'.weight_scale']=params.scale
                for rank in (0,1):self.local[rank][local+'.weight_scale']=split_matrix(params.scale,local,rank,scale=True)
            else:q=weight
            self.native['blocks.0.'+name+'.weight']=q
            for rank in (0,1):self.local[rank][local+'.weight']=split_matrix(q,local,rank)
        self.native['token_refiner.blocks.0.attn.out_proj.weight']=torch.randn((8,8),generator=generator).bfloat16()
        self.native['untouched']=torch.tensor([19,5],dtype=torch.int32)
        save_file(self.native,str(self.source))
        self.aux=self.root/'aux.safetensors'
        save_file({k:v for k,v in self.native.items() if not k.startswith('blocks.')},str(self.aux))
        source=CheckpointReader(self.source);self.states=[state_hashes(block(s)) for s in self.local]
        self.bundle=self.root/'bundle';self.bundle.mkdir();plans={}
        for kind in ('te','dit'):
            for index in range(50):
                for rank in (0,1):
                    stem=f'{kind}.block{index:02d}.rank{rank}'
                    plans[stem]=dict(kind=kind,index=index,rank=rank,state_sha256=self.states[rank],quant_metadata=self.quant)
        sources={k:dict(bytes=self.source.stat().st_size,sha256=sha(self.source)) for k in ('te','dit')}
        plan=dict(format=FORMAT,configuration=CONFIG,blocks=plans,sources=sources)
        for rank in (0,1):
            stem=f'dit.block00.rank{rank}';path=self.bundle/(stem+'.safetensors');save_file(self.local[rank],str(path))
            (self.bundle/(stem+'.json')).write_text(json.dumps(dict(format=FORMAT,configuration=CONFIG,
                **plans[stem],native_tensor_bit_exact=True,file_sha256=sha(path))))
        (self.bundle/'bundle.json').write_text(json.dumps(dict(complete=True,plan=plan,blocks=list(plans))))
        self.descriptor=self.root/'dit.json'
        self.descriptor.write_text(json.dumps(dict(format=DESCRIPTOR,kind='dit',bundle_root='bundle',
            bundle_sha256=sha(self.bundle/'bundle.json'),source=sources['dit'],native_header=source.header,
            native_header_sha256=hashlib.sha256(json.dumps(source.header,sort_keys=True).encode()).hexdigest(),
            auxiliary_file='aux.safetensors',auxiliary_sha256=sha(self.aux))))
        lora={}
        for key,value in self.native.items():
            if not key.endswith('.weight'):continue
            name='diffusion_model.'+key.removesuffix('.weight')
            lora[name+'.lora_A.weight']=torch.randn((2,value.shape[1]),generator=generator)*.03
            lora[name+'.lora_B.weight']=torch.randn((value.shape[0],2),generator=generator)*.02
            lora[name+'.alpha']=torch.tensor(3.)
        self.lora=self.root/'choice.safetensors';save_file(lora,str(self.lora))

    def test_component_packing_and_shared_row_scales_round_trip(self):
        for name in DIT_LINEAR.values():
            shape=(12 if name=='qkv_proj' else 8,512) if name!='adaln_proj' else (48,8)
            full=torch.arange(shape[0]*shape[1]).reshape(shape)
            self.assertTrue(torch.equal(full,join_matrix(split_matrix(full,name,0),split_matrix(full,name,1),name)))
        with self.assertRaises(ValueError):join_matrix(torch.ones(4,1),torch.zeros(4,1),'fc2',scale=True)

    def test_bundle_offline_matches_independent_full_native_matrix_and_keeps_base(self):
        before={p:sha(p) for p in self.bundle.iterdir()}
        reader=BundleReader(self.descriptor);reader.set_offline_lora(((str(self.lora),.7),))
        native=CheckpointReader(self.source);store=H3LoRAStore(((str(self.lora),.7),),native)
        with patch('patches.tp2_bundle.shell',side_effect=lambda kind,rank:block({k:torch.zeros_like(v) for k,v in self.local[rank].items()})):
            prepared=[reader.load_rank(0,r).state_dict() for r in (0,1)]
        for name,local in DIT_LINEAR.items():
            target='blocks.0.'+name;q,scale=merged_weight(native,store,target,'cpu')
            self.assertTrue(verify_weight(native,store,target,q,scale,'cpu'))
            for rank in (0,1):
                self.assertTrue(torch.equal(prepared[rank][local+'.weight'],split_matrix(q,local,rank)))
                if scale is not None:self.assertTrue(torch.equal(prepared[rank][local+'.weight_scale'],split_matrix(scale,local,rank,scale=True)))
        global_target='token_refiner.blocks.0.attn.out_proj'
        expected,_=merged_weight(native,store,global_target,'cpu')
        self.assertTrue(torch.equal(reader.read_tensor(global_target+'.weight'),expected))
        self.assertEqual(before,{p:sha(p) for p in self.bundle.iterdir()})
        self.assertEqual(reader.stats['persistent_baked_bytes'],0)
        self.assertFalse(hasattr(reader,'lora_store'))
        self.assertGreater(reader.workspace_bytes(0),0)

    def test_corruption_and_changed_lora_are_rejected(self):
        reader=BundleReader(self.descriptor);reader.set_offline_lora(((str(self.lora),1.),))
        reader._shard(0,0)
        (self.bundle/'dit.block00.rank0.safetensors').touch()
        with self.assertRaisesRegex(RuntimeError,'changed'):reader.check()
        fresh=BundleReader(self.descriptor);self.lora.touch()
        fresh.set_offline_lora(((str(self.lora),1.),))
        stat=self.lora.stat();os.utime(self.lora,ns=(stat.st_atime_ns,stat.st_mtime_ns+10**9))
        with self.assertRaisesRegex(RuntimeError,'LoRA changed'):fresh.check()

    def test_immutable_state_proof_is_once_per_owner_without_changing_merge_count(self):
        from patches import tp2_bundle as bundle
        reader=BundleReader(self.descriptor);reader.set_offline_lora(((str(self.lora),.7),))
        with patch.object(bundle,'shell',side_effect=lambda kind,rank:block({k:torch.zeros_like(v) for k,v in self.local[rank].items()})), \
                patch.object(bundle,'state_hashes',wraps=state_hashes) as hashes:
            for _ in range(2):
                for rank in (0,1):reader.load_rank(0,rank)
        self.assertEqual(hashes.call_count,2)
        self.assertEqual(reader.stats['prepared_blocks'],4)
        # The explicitly deferred full-width CPU merge remains unchanged.
        self.assertEqual(reader.stats['merged_matrices'],20)
        self.assertEqual(reader.stats['persistent_baked_bytes'],0)
        path=self.bundle/'dit.block00.rank0.safetensors'
        stat=path.stat();os.utime(path,ns=(stat.st_atime_ns,stat.st_mtime_ns+10**9))
        with self.assertRaisesRegex(RuntimeError,'changed'):reader.load_rank(0,0)

    def test_cached_proof_rejects_mutation_during_last_tensor_read(self):
        reader=BundleReader(self.descriptor)
        with patch('patches.tp2_bundle.shell',side_effect=lambda kind,rank:block({k:torch.zeros_like(v) for k,v in self.local[rank].items()})):
            reader.load_rank(0,0)
            source=reader._shard(0,0);original=source.read_tensor;calls=0
            def read(*args,**kwargs):
                nonlocal calls
                value=original(*args,**kwargs);calls+=1
                if calls==len(self.local[0]):
                    stat=source.path.stat();os.utime(source.path,ns=(stat.st_atime_ns,stat.st_mtime_ns+10**9))
                return value
            with patch.object(source,'read_tensor',side_effect=read):
                with self.assertRaisesRegex(RuntimeError,'changed'):reader.load_rank(0,0)
        self.assertEqual(reader.stats['prepared_blocks'],1)

    def test_different_lora_strength_changes_identity_without_new_base_files(self):
        first=BundleReader(self.descriptor);first.set_offline_lora(((str(self.lora),1.),))
        second=BundleReader(self.descriptor);second.set_offline_lora(((str(self.lora),.5),))
        self.assertNotEqual(first.identity,second.identity)

    def test_auxiliary_range_plan_contains_only_non_transformer_tensors(self):
        from benchmarks.build_tp2_bundle_assets import auxiliary
        from benchmarks.checkpoint_ranges import assemble
        plan=auxiliary(self.source,'dit');target=self.root/'assembled_aux.safetensors'
        raw=self.source.read_bytes();assemble(plan,target,lambda a,b:raw[a:b])
        reader=CheckpointReader(target)
        self.assertEqual(set(reader.header)-{'__metadata__'},set(self.native)-{n for n in self.native if n.startswith('blocks.')})
        for name in reader.header:
            if name!='__metadata__':self.assertTrue(torch.equal(reader.read_tensor(name),self.native[name]))

    def test_node_routes_offline_bundle_without_full_checkpoint_bake_or_online_adapter(self):
        import custom_nodes.minimax_h3_tp2 as node
        class Service:
            def __init__(self,config):self.config=config;self.closed=False
            def close(self):self.closed=True
        with patch.object(node,'_service',None),patch.object(node,'PhysicalTPService',Service), \
                patch.object(node,'create_model'),patch.object(node,'create_clip'), \
                patch('patches.offline_lora.cached_bake') as full_bake:
            service=node.MiniMaxH3TP2Loader().load('te',str(self.descriptor),
                lora_stack=((str(self.lora),.7),),lora_mode='offline')[2]
            self.assertTrue(service.config.bundle_offline_lora)
            self.assertEqual(service.config.lora_stack,((str(self.lora),.7),))
            full_bake.assert_not_called()


if __name__=='__main__':unittest.main()
