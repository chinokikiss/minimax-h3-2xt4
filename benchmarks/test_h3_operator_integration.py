"""Operator integration gates only; no H3 generation or performance claims."""
import argparse
from dataclasses import replace
from contextlib import contextmanager
import json
import os
from importlib.metadata import version
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch
import torch
from torch import nn

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),os.environ.get('COMFY_ROOT',str(ROOT/'ComfyUI'))]
from patches import h3_operator_policy as policy


def control_gates():
    from patches.comfy_tp_runtime import RuntimeConfig
    from patches.windowed_tp import tp_specs
    p=policy.H3OperatorPolicy();assert p.te_attention=='tilelang_group32' and p.dit_attention=='kitchen'
    config=RuntimeConfig('te','dit');assert config.operators==p
    assert replace(config,operators=replace(p,dit_attention='tilelang_scalar'))!=config
    for kwargs in ({'dit_attention':'bad'},{'te_attention':'bad'},{'row_linear_backend':'bad'}, {'scalar_score_workspace_mib':-1}):
        try:policy.H3OperatorPolicy(**kwargs)
        except ValueError:pass
        else:raise AssertionError(kwargs)
    layout=SimpleNamespace(seq_len=111,segments=[(0,13,'text'),(13,47,'ref_img'),(47,51,'audio'),(51,111,'video')])
    assert policy.packed_prefix({'minimax_h3_layout':layout},111)==51
    try:policy.packed_prefix({'minimax_h3_layout':layout},110)
    except ValueError:pass
    else:raise AssertionError('Stale layout accepted')
    reader=SimpleNamespace(identity='checkpoint',operator_policy=p)
    with patch('patches.rank_local_int8.rank_block_bytes',return_value=1024):
        a=tp_specs(reader,'dit',0,1)[0].key
        reader.operator_policy=replace(p,dit_attention='tilelang_drop')
        b=tp_specs(reader,'dit',0,1)[0].key
    assert a!=b and a[1:]==b[1:]
    from patches.weight_residency import copy_module
    block=SimpleNamespace(operator_policy=p)
    q=torch.randn(1,4,5,64);k=torch.randn_like(q);v=torch.randn_like(q)
    with patch('comfy_kitchen.int8_attention',return_value=v) as call:
        out=policy.dit_attention(block,q,k,v,{})
        assert out.shape==(1,5,256) and call.call_count==1
    shell=nn.Module();shell.out_proj=nn.Identity();shell.fc2=nn.Identity()
    policy.configure_block(shell,'dit',p)
    copied=copy_module(shell,lambda t:t.clone())
    assert copied.operator_policy==p and copied.fc2.operator_policy==p
    import custom_nodes.minimax_h3_tp2 as node
    MiniMaxH3TP2Loader=node.MiniMaxH3TP2Loader
    inputs=MiniMaxH3TP2Loader.INPUT_TYPES()['optional']
    assert inputs['te_attention'][1]['default']=='tilelang_group32'
    assert inputs['dit_attention'][1]['default']=='kitchen'
    from patches import h3_operator_dependencies as dependencies
    with patch.object(dependencies,'version',side_effect=lambda name:{'comfy-kitchen':'0.2.37','tilelang':'0.1.15'}[name]):
        dependencies.validate_dependencies(p,t4=True)
    with patch.object(dependencies,'version',return_value='0.2.33'):
        try:dependencies.validate_dependencies(p,t4=True)
        except RuntimeError as error:assert '0.2.37' in str(error)
        else:raise AssertionError('Incompatible Kitchen ABI accepted')
    class Service:
        def __init__(self,config):self.config=config;self.closed=False
        def close(self):self.closed=True
    with patch.object(node,'_service',None),patch.object(node,'PhysicalTPService',Service), \
            patch.object(node,'create_model',return_value='MODEL'),patch.object(node,'create_clip',return_value='CLIP'):
        loader=MiniMaxH3TP2Loader()
        _,_,first=loader.load('te','dit',dit_attention='tilelang_scalar',scalar_score_workspace_mib=0)
        _,_,again=loader.load('te','dit',dit_attention='tilelang_scalar',scalar_score_workspace_mib=0)
        assert first is again and first.config.operators.dit_attention=='tilelang_scalar'
        assert first.config.operators.scalar_score_workspace_mib==0
        _,_,second=loader.load('te','dit',dit_attention='kitchen')
        assert first.closed and first is not second
    preprocessing_seam_gates()
    return dict(name='config_layout_cache_dispatch_residency_UI',status='pass')


def preprocessing_seam_gates():
    from patches.windowed_tp import DiTPreprocessing
    class Model(nn.Module):
        hidden_size=8
        def __init__(self):
            super().__init__();self.video_patch_proj=nn.Linear(6,8);self.audio_patch_proj=nn.Linear(4,8)
            self.condition_proj=nn.Linear(12,8);self.token_refiner=nn.Identity()
        def preprocess_text_embeds(self,x):return x
    class OldModel(Model):
        def _embed_and_pack(self,v,a,c,layout,payload,options):
            return torch.cat((c[0],self.video_patch_proj(v),self.audio_patch_proj(a)),0)
    class Comm:
        def __init__(self):self.rank=0;self.values=[];self.cursor=0
        def broadcast(self,x,src):
            if self.rank==0:self.values.append(x.clone())
            else:x.copy_(self.values[self.cursor]);self.cursor+=1
    for old in (False,True):
        cls=OldModel if old else Model
        torch.manual_seed(819);models=[cls(),cls()];models[1].load_state_dict(models[0].state_dict())
        v=torch.randn(5,6);a=torch.randn(3,4);c=torch.randn(1,2,8)
        reference=torch.cat((c[0],models[0].video_patch_proj(v),models[0].audio_patch_proj(a)),0)
        comm=Comm();owners=[DiTPreprocessing(m,r,'cpu',comm,'identity') for r,m in enumerate(models)]
        class Window:
            calls=0
            @contextmanager
            def use(self,key):
                self.calls+=1
                yield nn.ModuleDict({name:owners[0].canonical[name] for name in ('video_patch_proj','audio_patch_proj')})
        owners[0].window=Window()
        for rank in (0,1):
            comm.rank=rank
            if old:y=models[rank]._embed_and_pack(v,a,c,SimpleNamespace(seq_len=10),{}, {})
            else:y=torch.cat((c[0],models[rank].video_patch_proj(v),models[rank].audio_patch_proj(a)),0)
            torch.testing.assert_close(y,reference,rtol=0,atol=0)
        assert owners[0].window.calls==owners[0].project_uses==(1 if old else 2)
        for owner in owners:owner.close()


def selective_b_gates(device):
    from patches.h3_lora import H3LoRAStore,NativeAlignedLoRA
    torch.manual_seed(342)
    factors={'A':torch.randn(128,5376)*.01,'B':torch.randn(28672,128)*.01}
    reads=[]
    class Reader:
        header={'A':{'shape':[128,5376]},'B':{'shape':[28672,128]}}
        def read_tensor(self,key,rows=None,columns=None):
            reads.append((key,rows,columns));value=factors[key]
            if rows is not None:value=torch.cat([value[a:b] for a,b in rows])
            if columns is not None:value=value[:,columns[0]:columns[1]]
            return value
    target='blocks.0.mlp.fc1';reader=Reader()
    store=object.__new__(H3LoRAStore);store.selective_b=True;store.row_partial=False
    store.sources=[(reader,1.,{target:('A','B',.6)}),(reader,1.,{target:('A','B',-.2)})]
    store.check=lambda:None
    x=torch.randn(7,5376,device=device)
    for rank in (0,1):
        rows=store.selective_rows(target,rank)
        adapter=store.adapter(target,rank).to(device)
        assert adapter.selective_b and all(t.up.shape==(14336,128) for t in adapter.terms)
        assert all(row[1]==rows for row in reads if row[0]=='B')
        baseline=NativeAlignedLoRA([(factors['A'],factors['B'],.6),(factors['A'],factors['B'],-.2)],rows).to(device)
        base=torch.randn(7,14336,device=device)
        a=adapter.add_to(x,base.clone());b=baseline.add_to(x,base.clone())
        torch.testing.assert_close(a,b,rtol=1e-6,atol=1e-6)
        expected=4*2*(128*5376+14336*128)
        assert store.prefix_bytes('blocks.0.',rank)==expected
        reads.clear()
    assert store.selective_rows('blocks.0.attn.qkv_proj',0) is None
    assert store.selective_rows('token_refiner.blocks.0.mlp.fc1',0) is None
    reader.header={'A':{'shape':[64,5376]},'B':{'shape':[28672,64]}}
    assert store.selective_rows(target,0) is None
    return dict(name='selective_B_both_owners_stack_scale_order_and_bytes',device=str(device),status='pass')


def audio_gates():
    from patches.parallel_vae import channel_decode
    z=torch.randn(2,32,2,5)
    model=SimpleNamespace(decode=lambda x:x.sum(1))
    expected=model.decode(z)
    class Comm:
        def get_world_size(self):return 2
        def all_gather(self,parts,local):
            for i,p in enumerate(parts):p.copy_(expected[:,i:i+1])
    torch.testing.assert_close(channel_decode(model,z,Comm(),0),expected,rtol=0,atol=0)
    assert channel_decode(model,z,Comm(),1) is None
    return dict(name='persistent_audio_channel_ownership_and_join',status='pass')


def masked_reference(p,mask):
    from patches.tilelang_group32_masked import normalize_mask
    b,h,_,d=p.qi.shape;hk=p.ki.shape[1];n=p.queries;nk=p.keys
    def dq(i,s,length,heads):
        return (i[:,:,:length].float().reshape(b,heads,length,d//32,32)*s[:,:,:length,:,None]).reshape(b,heads,length,d)
    q=dq(p.qi,p.qs,n,h);k=dq(p.ki,p.ks,nk,hk).repeat_interleave(h//hk,1)
    v=p.v[:,:,:nk].repeat_interleave(h//hk,1).float()
    mask=normalize_mask(mask,b,h,n,nk,q.device) if mask is not None else 0.
    logits=q@k.transpose(-1,-2)/d**.5+mask
    # Independent eager streaming reference retains the existing per-leaf FP16 P.
    m=torch.full((b,h,n),-float('inf'),device=q.device);den=torch.zeros_like(m);out=torch.zeros_like(q)
    for start in range(0,nk,64):
        z=logits[...,start:start+64];updated=torch.maximum(m,z.amax(-1))
        safe=torch.where(updated==-float('inf'),torch.zeros_like(updated),updated)
        alpha=(m-safe).exp();prob=(z-safe[...,None]).exp()
        out=out*alpha[...,None]+prob.half().float()@v[:,:,start:start+64]
        den=den*alpha+prob.sum(-1);m=updated
    out=torch.where(den[...,None]==0,torch.zeros_like(out),out/den[...,None]).half().float()
    return out.transpose(1,2).reshape(b,n,h*d)


def masked_gates(local_preview):
    from patches import tilelang_grouped_tp_optimized as preparation
    from patches import tilelang_group32_masked as dense
    device=torch.device('cuda',torch.cuda.current_device());torch.manual_seed(928)
    q=torch.randn(2,4,37,64,device=device,dtype=torch.float16)*.2
    k=torch.randn(2,1,73,64,device=device,dtype=torch.float16)*.2;v=torch.randn_like(k)
    # Eager preparation makes the preview architecture explicit; production uses fused SM75 preparation.
    p=preparation.prepare(q,k,v,fused=False,mode='dense',query_tile=32)
    masks=[None,torch.ones(37,73,device=device,dtype=torch.bool).tril(),torch.randn(2,1,1,73,device=device)*.2]
    masks[1][3,:64]=False # Initial leaf has no mass; later leaf does.
    masks[1][4,:]=False # Entirely masked query must be zero.
    records=[]
    for index,mask in enumerate(masks):
        y=dense.execute_prepared(p,mask,local_preview=local_preview)
        expected=masked_reference(p,mask)
        torch.testing.assert_close(y,expected,rtol=.006,atol=.002)
        assert torch.isfinite(y).all()
        if index==1:assert torch.count_nonzero(y[:,4])==0
        records.append(dict(name='masked_GQA_'+str(index),status='pass',max_abs=float((y-expected).abs().max())))
    return records


def t4_dispatch_gates():
    from patches import tilelang_grouped_tp_kitchen_v2 as dense
    from patches import tilelang_group32_masked as masked
    device=torch.device('cuda',torch.cuda.current_device());torch.manual_seed(812)
    q=torch.randn(1,4,65,128,device=device);k=torch.randn(1,1,65,128,device=device);v=torch.randn_like(k)
    mask=torch.ones(65,65,device=device,dtype=torch.bool).tril()
    block=SimpleNamespace(operator_policy=policy.H3OperatorPolicy())
    y=policy.te_attention(block,q,k,v,mask)
    expected=masked_reference(dense.prepare(q,k,v),mask)
    torch.testing.assert_close(y,expected,rtol=.006,atol=.003)
    q=torch.randn(1,2,321,128,device=device)*.2;k=torch.randn_like(q)*.2;v=torch.randn_like(q)
    options={'minimax_h3_layout':SimpleNamespace(seq_len=321,segments=[(0,7,'text'),(7,321,'video')])}
    # Exercise real dispatch and both scalar workspace branches on the target.
    for mode in ('tilelang_drop','tilelang_scalar','tilelang_dense'):
        block.operator_policy=replace(block.operator_policy,dit_attention=mode)
        actual=policy.dit_attention(block,q,k,v,options)
        assert actual.shape==(1,321,256) and torch.isfinite(actual).all()
        if mode=='tilelang_scalar':
            block.operator_policy=replace(block.operator_policy,scalar_score_workspace_mib=0)
            inline=policy.dit_attention(block,q,k,v,options)
            torch.testing.assert_close(actual,inline,rtol=.001,atol=.0005)
    return dict(name='SM75_fused_TE_and_real_sparse_dispatch',status='pass')


def distributed_row_gates(comm,rank):
    from patches import optimized_tp as native,parallel_int8 as original
    native.enable()
    device=torch.device('cuda',rank);torch.manual_seed(294+rank)
    records=[]
    for width,parts in ((7168,False),(12800,True)):
        gate=torch.randn(17,width,device=device);up=torch.randn_like(gate)
        w=torch.randint(-127,128,(256,width),device=device,dtype=torch.int8)
        ws=torch.rand(256,1,device=device)*.001
        module=SimpleNamespace(weight=w,weight_scale=ws,world_size=2,convrot_groupsize=256,
            operator_policy=policy.H3OperatorPolicy())
        # Independent full gather/Native quantization, then actual INT32 SUM.
        packed=torch.cat((gate,up),-1)
        expected=original.row_linear(packed,w,ws,comm,2,256,'swiglu')
        actual=(policy.row_linear_parts(module,gate,up,native.row_linear,comm) if parts else
                policy.row_linear(module,packed,native.row_linear,comm,'swiglu'))
        assert torch.equal(actual,expected),(width,float((actual-expected).abs().max()))
        records.append(dict(name='actual_NCCL_packed_row_'+str(width),rank=rank,status='bitwise_pass'))
    return records


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--cuda-preview',action='store_true')
    parser.add_argument('--t4',action='store_true');parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--distributed',action='store_true')
    args=parser.parse_args();sys.argv=sys.argv[:1]
    torch.set_grad_enabled(False);torch.backends.cuda.matmul.allow_tf32=False
    rank=int(os.environ.get('LOCAL_RANK','0'))
    if args.cuda_preview or args.t4:torch.cuda.set_device(rank)
    if args.distributed:
        assert args.t4 and os.environ['NCCL_P2P_DISABLE']=='1'
        import torch.distributed as dist
        dist.init_process_group('nccl')
    records=[control_gates(),selective_b_gates('cpu'),audio_gates()]
    if args.cuda_preview or args.t4:
        assert torch.cuda.get_device_capability()==((12,0) if args.cuda_preview else (7,5))
        records.append(selective_b_gates(torch.device('cuda',rank)))
        records+=masked_gates(args.cuda_preview)
        if args.t4:records.append(t4_dispatch_gates())
    if args.distributed:records+=distributed_row_gates(dist,rank)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    output=args.output.with_name(args.output.stem+f'_rank{rank}.json') if args.distributed else args.output
    output.write_text(json.dumps(dict(scope='Operator arithmetic/integration gates; no full-pipeline speed/quality qualification',
        torch=torch.__version__,comfy_kitchen=version('comfy-kitchen'),tilelang=version('tilelang'),hardware=torch.cuda.get_device_name() if args.cuda_preview or args.t4 else 'CPU',records=records),indent=2))
    print('H3_OPERATOR_INTEGRATION_PASS',len(records),flush=True)
    if args.distributed:dist.barrier();dist.destroy_process_group()


if __name__=='__main__':main()
