"""CPU regression contracts for the five opt-in TP2 enhancements.

GPU arithmetic is checked separately by test_tp2_p2_epilogue.py and the
tiny-block integration gates. No checkpoint or full H3 generation required.
"""
import ast
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import inspect
from pathlib import Path
import sys
import os
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),os.environ.get('COMFY_ROOT',str(ROOT/'ComfyUI'))]
import torch
from safetensors.torch import save_file
from patches.windowed_tp import CheckpointReader
from patches import h3_operator_policy as policy, tilelang_transport_p2 as transport
from patches.comfy_tp_runtime import RuntimeConfig
from patches.weight_residency import ResidencyWindow


class PositionalIO(unittest.TestCase):
    def setUp(self):
        self.folder=tempfile.TemporaryDirectory();self.addCleanup(self.folder.cleanup)
        self.path=Path(self.folder.name)/'fixture.safetensors'
        self.tensors={'matrix':torch.arange(35,dtype=torch.float32).reshape(7,5),
            'bf16':torch.arange(16,dtype=torch.bfloat16).reshape(4,4),
            'vector':torch.arange(9,dtype=torch.int64),'scalar':torch.tensor(3.),
            'cube':torch.arange(24,dtype=torch.int8).reshape(2,3,4)}
        save_file(self.tensors,str(self.path));self.blob=self.path.read_bytes()
        self.reader=CheckpointReader(self.path)

    def pread(self,fd,size,offset):
        return self.blob[offset:offset+min(size,7)]  # Force partial reads.

    def check_cases(self):
        for name,value in self.tensors.items():
            torch.testing.assert_close(self.reader.read_tensor(name),value,rtol=0,atol=0)
        selected=torch.cat((self.tensors['matrix'][4:6,1:4],self.tensors['matrix'][0:2,1:4]))
        torch.testing.assert_close(self.reader.read_tensor('matrix',rows=[(4,6),(0,2)],columns=(1,4)),selected,rtol=0,atol=0)
        torch.testing.assert_close(self.reader.read_tensor('matrix',rows=[(4,6),(0,2)]),
            torch.cat((self.tensors['matrix'][4:6],self.tensors['matrix'][0:2])),rtol=0,atol=0)
        torch.testing.assert_close(self.reader.read_tensor('vector',rows=[(7,9),(0,2)]),
            torch.tensor([7,8,0,1]),rtol=0,atol=0)

    def test_pread_short_reads_no_seek(self):
        real_open=Path.open
        class Stream:
            def __init__(self,s):self.s=s
            def __enter__(self):return self
            def __exit__(self,*args):self.s.close()
            def fileno(self):return self.s.fileno()
            def seek(self,*args):raise AssertionError('pread path must not seek')
            def readinto(self,*args):raise AssertionError('pread path must not readinto')
        with patch('patches.windowed_tp.os.pread',side_effect=self.pread,create=True), \
                patch.object(Path,'open',lambda path,*a,**kw:Stream(real_open(path,*a,**kw))):
            self.check_cases()
        self.assertGreater(self.reader.range_read_calls,20)
        expected=sum(t.numel()*t.element_size() for t in self.tensors.values())+48+80+32
        self.assertEqual(self.reader.logical_read_bytes,expected)

    def test_unavailable_and_unsupported_fd_fallback(self):
        for pread in (None, lambda *args: (_ for _ in ()).throw(OSError('unsupported fd'))):
            with patch('patches.windowed_tp.os.pread',pread,create=True):self.check_cases()

    def test_fallback_after_partial_pread(self):
        calls=0
        def read(*args):
            nonlocal calls
            calls+=1
            if calls==1:return self.pread(*args)
            raise OSError('fd no longer supports positional reads')
        with patch('patches.windowed_tp.os.pread',read,create=True):
            torch.testing.assert_close(self.reader.read_tensor('matrix'),self.tensors['matrix'],rtol=0,atol=0)

    def test_short_readinto_and_eof(self):
        real_open=Path.open
        class Stream:
            def __init__(self,s):self.s=s
            def __enter__(self):return self
            def __exit__(self,*args):self.s.close()
            def seek(self,*args):return self.s.seek(*args)
            def readinto(self,buffer):return self.s.readinto(buffer[:3])
        with patch('patches.windowed_tp.os.pread',None,create=True), \
                patch.object(Path,'open',lambda path,*a,**kw:Stream(real_open(path,*a,**kw))):
            self.check_cases()
        with patch('patches.windowed_tp.os.pread',return_value=b'',create=True):
            with self.assertRaisesRegex(IOError,'incomplete'):self.reader.read_tensor('matrix')

    def test_concurrent_reads_and_checkpoint_mutation(self):
        with patch('patches.windowed_tp.os.pread',side_effect=self.pread,create=True),ThreadPoolExecutor(4) as pool:
            results=list(pool.map(lambda _:self.reader.read_tensor('matrix',rows=[(2,5)],columns=(1,3)),range(12)))
        for value in results:torch.testing.assert_close(value,self.tensors['matrix'][2:5,1:3],rtol=0,atol=0)
        self.assertEqual(self.reader.logical_read_bytes,12*3*2*4)
        self.path.write_bytes(self.blob+b'x')
        with self.assertRaisesRegex(RuntimeError,'changed'):self.reader.read_tensor('matrix')


class OperatorSelection(unittest.TestCase):
    def test_defaults_and_optional_dispatch(self):
        default=RuntimeConfig('te','dit')
        self.assertTrue(default.adaptive_lookahead)
        self.assertFalse(hasattr(default,'swiglu_reduction'))
        self.assertEqual(default.operators.row_linear_backend,'tilelang_scaled_fp16')
        self.assertEqual(default.operators.fc2_quantizer,'fused_stored')
        previous=replace(default.operators,row_linear_backend='tilelang_packed',fc2_quantizer='owned_fp32')
        x=torch.randn(2,256);w=torch.zeros(128,256,dtype=torch.int8);ws=torch.ones(128)
        module=SimpleNamespace(weight=w,weight_scale=ws,world_size=2,convrot_groupsize=256,
                               operator_policy=previous)
        from patches import tilelang_swiglu_pad as swiglu
        with patch.object(policy,'sm75',return_value=True),patch.object(swiglu,'row_linear',return_value=x) as native, \
                patch.object(transport,'row_linear',return_value=x) as scaled:
            policy.row_linear(module,x,None,None);self.assertEqual(native.call_count,1);scaled.assert_not_called()
            self.assertEqual(native.call_args.kwargs['rotation_mode'],'owned_fp32')
            module.operator_policy=replace(previous,fc2_quantizer='native')
            policy.row_linear(module,x,None,None)
            self.assertEqual(native.call_args.kwargs['rotation_mode'],'native')
            module.operator_policy=replace(previous,row_linear_backend='tilelang_scaled_fp16')
            policy.row_linear(module,x,None,None);self.assertEqual(scaled.call_count,1)
            scaled.assert_called_with(x,w,ws,None,2,256,None)
        with patch.object(policy,'sm75',return_value=True),patch.object(transport,'row_linear_parts',return_value=x) as parts:
            policy.row_linear_parts(module,x,x,None,None);self.assertEqual(parts.call_count,1)
        with self.assertRaises(TypeError):replace(module.operator_policy,swiglu_padding=False)
        # Unsupported devices must preserve the supplied native implementation.
        marker=object()
        with patch.object(policy,'sm75',return_value=False):
            self.assertIs(policy.row_linear(module,x,lambda *a:marker,None),marker)

    def test_codec_collect_and_fused_finish_sequence(self):
        from patches import parallel_int8 as pi,tilelang_row_quant as row
        q=torch.ones(3,256,dtype=torch.int8);w=torch.ones(128,256,dtype=torch.int8)
        scale=torch.ones(3,1);ws=torch.ones(128);partial=torch.ones(3,128,dtype=torch.int32)
        events=[]
        class Comm:
            ReduceOp=SimpleNamespace(SUM='sum')
            def all_reduce(self,x,op):
                self.payload=x.clone();events.append(('sum',x.dtype,op));x.mul_(2)
        comm=Comm()
        def encode(x,scheme,k):
            self.assertIs(x,partial);events.append(('encode',scheme,k));return x.half()
        def finish(x,a,b,scheme,k):
            events.append(('fused_epilogue',scheme,k));return x.float()*a*b
        from contextlib import nullcontext
        with patch.object(pi,'int_mm',return_value=partial),patch.object(transport,'encode',side_effect=encode), \
                patch.object(transport,'finish',side_effect=finish),patch.object(row,'label',side_effect=lambda _:nullcontext()):
            actual=transport._finish_row(q,scale,(1,3,256),w,ws,comm)
        self.assertEqual(actual.shape,(1,3,128));self.assertTrue((actual==2).all())
        self.assertEqual(events,[('encode','fp16',256),('sum',torch.float16,'sum'),('fused_epilogue','fp16',256)])

    def test_loader_roundtrip_and_dependencies(self):
        import custom_nodes.minimax_h3_tp2 as node
        from patches import h3_operator_dependencies as dep
        options=node.MiniMaxH3TP2Loader.INPUT_TYPES()['optional']
        self.assertTrue(options['adaptive_lookahead'][1]['default'])
        self.assertEqual(options['row_linear_backend'][1]['default'],'tilelang_scaled_fp16')
        self.assertEqual(options['fc2_quantizer'][1]['default'],'fused_stored')
        class Service:
            def __init__(self,config):self.config=config;self.closed=False
            def close(self):self.closed=True
        with patch.object(node,'_service',None),patch.object(node,'PhysicalTPService',Service), \
                patch.object(node,'create_model'),patch.object(node,'create_clip'):
            loader=node.MiniMaxH3TP2Loader()
            first=loader.load('te','dit')[2]
            self.assertEqual(first.config.operators.fc2_quantizer,'fused_stored')
            native=loader.load('te','dit',fc2_quantizer='native')[2]
            self.assertTrue(first.closed)
            self.assertEqual(native.config.operators.fc2_quantizer,'native')
            second=loader.load('te','dit',adaptive_lookahead=True,row_linear_backend='tilelang_scaled_fp16')[2]
            self.assertTrue(native.closed)
            self.assertTrue(first.closed);self.assertTrue(second.config.adaptive_lookahead)
            self.assertEqual(second.config.operators.row_linear_backend,'tilelang_scaled_fp16')
            self.assertIs(second,loader.load('te','dit',adaptive_lookahead=True,row_linear_backend='tilelang_scaled_fp16')[2])
        p=policy.H3OperatorPolicy(te_attention='native',row_linear_backend='tilelang_scaled_fp16')
        with patch.object(dep,'version',side_effect=lambda name:{'comfy-kitchen':'0.2.37','tilelang':'0.1.13'}[name]) as version:
            dep.validate_dependencies(p,t4=True)
            self.assertIn(('tilelang',),[c.args for c in version.call_args_list])


class AdaptiveWindow(unittest.TestCase):
    def test_headroom_byte_budget_resident_reuse_and_bounds(self):
        w=ResidencyWindow.__new__(ResidencyWindow)
        w.adaptive=True;w.lookahead=2;w.max_lookahead=2;w.cursor=0;w.device='cuda:0'
        w.uses=('a','a','b','c');w.specs={k:SimpleNamespace(max_bytes=100) for k in 'abc'}
        w.entries={};w.max_weight_bytes=300;w.reserve_bytes=50;w.stats={'lookahead_changes':[]}
        with patch.object(w,'_live_bytes',return_value=0),patch('torch.cuda.mem_get_info',return_value=(249,1000)), \
                patch('torch.cuda.memory_stats',return_value={}):
            w._adapt_gpu_lookahead();self.assertEqual(w.lookahead,1)
        # Reusable reserved memory counts, including precise boundary equality.
        with patch.object(w,'_live_bytes',return_value=0),patch('torch.cuda.mem_get_info',return_value=(150,1000)), \
                patch('torch.cuda.memory_stats',return_value={'reserved_bytes.all.current':150,'active_bytes.all.current':50}):
            w._adapt_gpu_lookahead();self.assertEqual(w.lookahead,2)
            w.max_weight_bytes=199;w._adapt_gpu_lookahead();self.assertEqual(w.lookahead,1)
            w.entries={'a':object()};w.max_weight_bytes=200
            with patch.object(w,'_live_bytes',return_value=100):
                w._adapt_gpu_lookahead();self.assertEqual(w.lookahead,2)
            w.max_lookahead=1;w._adapt_gpu_lookahead();self.assertEqual(w.lookahead,1)
        w.adaptive=False
        with patch('torch.cuda.mem_get_info',side_effect=AssertionError('Disabled path must not query GPU')):
            w._adapt_gpu_lookahead()

    def test_all_runtime_window_constructors_forward_flag(self):
        import patches.comfy_tp_runtime as runtime
        tree=ast.parse(inspect.getsource(runtime))
        constructors=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='ResidencyWindow']
        self.assertEqual(len(constructors),2)
        for call in constructors:
            values={kw.arg:ast.unparse(kw.value) for kw in call.keywords}
            self.assertEqual(values['adaptive'],'self.config.adaptive_lookahead')
            self.assertEqual(values['lookahead'],'self.config.gpu_lookahead')


class RefinerAndLifetime(unittest.TestCase):
    def test_refiner_native_sdpa_and_legacy_alias(self):
        from torch import nn
        class Base(nn.Module):
            def __init__(self):
                super().__init__();self.heads=2;self.head_dim=4
                self.qkv_proj=nn.Linear(8,24,bias=False);self.out_proj=nn.Linear(8,8,bias=False)
                self.q_norm=nn.RMSNorm(4);self.k_norm=nn.RMSNorm(4)
        base=Base();x=torch.randn(9,8)
        q,k,v=base.qkv_proj(x).chunk(3,-1)
        q,k=base.q_norm(q.reshape(9,2,4)),base.k_norm(k.reshape(9,2,4))
        q,k,v=(a.reshape(9,2,4).transpose(0,1).unsqueeze(0) for a in (q,k,v))
        expected=base.out_proj(torch.nn.functional.scaled_dot_product_attention(q,k,v).transpose(1,2).reshape(9,8))
        with patch('comfy_kitchen.int8_attention',side_effect=AssertionError('Refiner cannot use INT8')), \
                patch('torch.nn.functional.scaled_dot_product_attention',wraps=torch.nn.functional.scaled_dot_product_attention) as call:
            actual=policy.NativeRefinerAttention(base)(x)
            self.assertEqual(call.call_count,1)
        self.assertIs(policy.KitchenRefinerAttention,policy.NativeRefinerAttention)
        self.assertTrue(torch.equal(actual,expected))

    def test_te_deletions_preserve_exact_arithmetic_both_mlp_paths(self):
        # Compare the real forward against the same function with only del nodes
        # removed, using deterministic CPU operators. Hooks prove buffers are gone
        # before the next stage rather than merely checking source spelling.
        import textwrap,weakref
        from patches import tp_qwen3vl as te
        from torch import nn
        tree=ast.parse(textwrap.dedent(inspect.getsource(te.TPTransformerBlockINT8.forward)))
        class WithoutDeletes(ast.NodeTransformer):
            def visit_Delete(self,node):return None
        ns=dict(te.__dict__);exec(compile(ast.fix_missing_locations(WithoutDeletes().visit(tree)),'<without_del>','exec'),ns)
        for configured in (False,True):
            refs={};block=SimpleNamespace(world_size=1,num_heads_per_rank=2,num_kv_heads_per_rank=2,head_dim=4,
                operator_policy=SimpleNamespace() if configured else None)
            for name in ('input_layernorm','post_attention_layernorm','q_norm','k_norm'):setattr(block,name,nn.RMSNorm(4 if name in ('q_norm','k_norm') else 8))
            class Projection(nn.Linear):
                def forward(self,x,**kw):return super().forward(x)
                def forward_parts(self,g,u):return self(torch.nn.functional.silu(g)*u)
            for name in ('q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'):setattr(block,name,Projection(8,8,bias=False))
            x=torch.randn(1,5,8)
            def attend(b,q,k,v,mask):return torch.nn.functional.scaled_dot_product_attention(q,k,v).transpose(1,2).reshape(1,5,8)
            with torch.no_grad(),patch.object(policy,'te_attention',new=attend), \
                    patch('comfy.text_encoders.llama.apply_rope',new=lambda q,k,**kw:(q,k)):
                expected=ns['forward'](block,x)
                for name in ('q_proj','k_proj','v_proj'):
                    getattr(block,name).register_forward_hook(lambda m,a,o,n=name:refs.update({n:weakref.ref(o)}))
                def check(*args):
                    self.assertEqual([name for name,ref in refs.items() if ref() is not None],[])
                block.post_attention_layernorm.register_forward_pre_hook(check)
                # rms_norm consumes weights directly, so observe MLP's first projection.
                block.gate_proj.register_forward_pre_hook(check)
                actual=te.TPTransformerBlockINT8.forward(block,x)
            self.assertTrue(torch.equal(actual,expected))


if __name__=='__main__':unittest.main(verbosity=2)
