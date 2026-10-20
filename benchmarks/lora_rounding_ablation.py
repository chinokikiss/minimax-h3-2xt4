"""Opt-in controlled arithmetic ablations; never installed in inference."""
import hashlib
import json
from pathlib import Path
import simulate_weight_residency as setup
import torch
import torch.nn.functional as F
import retest
import te_dit_study as study
from qualify_h3_lora import native_adapt
from logical_dual_gpu import LogicalDualGPU,isolate_cublas_workspaces
from patches.h3_lora import H3LoRAStore,NativeAlignedLoRA
from patches.windowed_tp import CheckpointReader,prepare_tp_block
from patches.optimized_tp import enable
from patches.parallel_int8 import quantize


class Ablation(NativeAlignedLoRA):
    def __init__(self,terms,rows,columns,variant):
        super().__init__(terms,rows,columns);self.variant=variant
    def add_to(self,x,out,comm=None,input_act=None):
        variant=self.variant
        if variant=='native_full':return super().add_to(x,out,comm,input_act)
        value=x.reshape(-1,x.shape[-1]);target=out.reshape(-1,out.shape[-1])
        if input_act=='swiglu':
            gate,up=value.chunk(2,-1);value=F.silu(gate)*up
        partial=self.columns is not None and variant in ('row_partial','old_concat')
        if self.columns is not None and not partial:
            parts=[torch.empty_like(value) for _ in (0,1)];comm.all_gather(parts,value.contiguous());value=torch.cat(parts,-1)
        down=[t.down[:,self.columns[0]:self.columns[1]].contiguous() if partial else t.down for t in self.terms]
        sliced=self.rows is not None and variant in ('column_B_shard','old_concat')
        up=[torch.cat([t.up[a:b] for a,b in self.rows],0) if sliced else t.up for t in self.terms]
        def add(delta):
            if self.rows is None or sliced:target.add_(delta)
            else:
                start=0
                for a,b in self.rows:target[:,start:start+b-a].add_(delta[:,a:b]);start+=b-a
        if variant=='old_concat':
            hidden=F.linear(value,torch.cat(down,0))
            if partial:comm.all_reduce(hidden,op=comm.ReduceOp.SUM)
            add(F.linear(hidden,torch.cat([b*t.scale for b,t in zip(up,self.terms)],1)))
        else:
            combined=F.linear(value,torch.cat(down,0)) if variant=='concat_A' else None
            offset=0
            for a,b,term in zip(down,up,self.terms):
                hidden=F.linear(value,a) if combined is None else combined[:,offset:offset+a.shape[0]].contiguous()
                offset+=a.shape[0]
                if partial:comm.all_reduce(hidden,op=comm.ReduceOp.SUM)
                delta=F.linear(hidden,b*term.scale) if variant=='fold_scale' else F.linear(hidden,b)*term.scale
                add(delta)
        return out


class Store(H3LoRAStore):
    variant='native_full'
    def adapter(self,target,rank=None):
        exact=super().adapter(target,rank)
        if exact is None:return None
        return Ablation([(t.down,t.up,t.scale) for t in exact.terms],exact.rows,exact.columns,self.variant)


def main():
    import comfy_kitchen as ck
    import patches.tp_dit as dit
    ck.backends.cuda._max_dynamic_shared_memory_per_block=lambda _:49152
    torch.set_grad_enabled(False);torch.backends.cuda.matmul.allow_tf32=False;isolate_cublas_workspaces();enable('scalemax')
    root=Path(__file__).resolve().parents[1];base=CheckpointReader(setup.PATHS['dit'])
    adapter='D:/minimaxH3/loras/dmad_minimax_h3_4step_full_critic_comfyui.safetensors'
    store=Store(((adapter,.6),(adapter,-.2)),base);rows=[];original_comm=dit.dist
    try:
        for count in (257,512):
            xlist,alist=study.inputs('dit',base.path,count,1,'cuda');x=xlist[0];args=alist[0]
            captures={}
            def hook(label):
                def capture(module,inputs,out):captures[label]=out.detach().clone()
                return capture
            native=native_adapt(retest.load_native_block('dit',base.path),store,0)
            native.attn.qkv_proj.register_forward_hook(hook('native_qkv'))
            native.attn.out_proj.register_forward_hook(hook('native_out'))
            old_add=native.mlp.fc1_adapter.add_to
            def fc1_add(*a,**kw):
                out=old_add(*a,**kw);captures['native_fc1']=out.detach().clone();return out
            native.mlp.fc1_adapter.add_to=fc1_add
            reference=native(x.clone(),*args).cpu();del native
            for variant in ('native_full','fold_scale','concat_A','row_partial','column_B_shard','old_concat'):
                store.variant=variant;base.lora_store=store
                blocks=[prepare_tp_block(base,'dit',0,r).cuda() for r in (0,1)]
                for r in (0,1):
                    blocks[r].qkv_proj.register_forward_hook(hook(f'qkv_{r}'))
                    blocks[r].out_proj.register_forward_hook(hook(f'out_{r}'))
                    blocks[r].fc1.register_forward_hook(hook(f'fc1_{r}'))
                comm=LogicalDualGPU(180);dit.dist=comm
                outputs=comm.run([lambda r=r:blocks[r](x.clone(),*args) for r in (0,1)])
                stages={}
                for label,components in (('qkv',3),('out',1),('fc1',2)):
                    packed=captures[label+'_0'] if label=='out' else torch.cat([
                        torch.cat([captures[f'{label}_{r}'].chunk(components,-1)[c] for r in (0,1)],-1) for c in range(components)],-1)
                    stages[label]=retest.metrics(packed,captures['native_'+label])
                    if label=='fc1':
                        q,s=quantize(packed,input_act='swiglu');qr,sr=quantize(captures['native_fc1'],input_act='swiglu')
                        stages['fc2_quantizer']=dict(changed_payload=int((q!=qr).sum()),changed_scales=int((s!=sr).sum()),total=q.numel())
                row=dict(rows=count,variant=variant,stages=stages,output=retest.metrics(outputs[0].cpu(),reference))
                rows.append(row);print('ABLATION',count,variant,'block_rel',row['output'].get('rel_l2'),
                    'qkv_exact',stages['qkv']['is_bit_exact'],'qchanges',stages['fc2_quantizer']['changed_payload'],flush=True)
                del blocks,outputs,packed,q,s,qr,sr;torch.cuda.empty_cache()
            captures.clear();del xlist,alist,x,args,reference;torch.cuda.empty_cache()
    finally:dit.dist=original_comm
    receipt=dict(physical_t4=False,scope='Controlled numerical ablations on the local 5060 Ti; no timing claims',cases=rows,
        source_sha256={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in
            ('patches/h3_lora.py','patches/tp_dit.py','patches/optimized_tp.py','benchmarks/lora_rounding_ablation.py')})
    (root/'semantic_results/h3_lora_rounding_ablation.json').write_text(json.dumps(receipt,indent=2))


if __name__=='__main__':main()
