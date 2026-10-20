"""Same trained decoder on logical ranks (qualification) or physical NCCL ranks.

Raw decoding only. Approximate ablations are timed to explain cost, never ranked
as valid speedups. No instrumentation or correctness copies in timing samples.
"""
import argparse
from contextlib import contextmanager
from dataclasses import replace
from datetime import timedelta
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import statistics
import sys
import time
from types import MethodType

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT/'benchmarks', ROOT/'ComfyUI', ROOT/'.retest-deps'):
    sys.path.insert(0, str(path))
import torch
import torch.distributed as dist
from retest import load_video_native
import comfy.ops
import comfy.quant_ops
from comfy.ldm.minimax.vae import create_token_ids
from patches.slab_video_vae import SlabPlan, stitch_slabs
from logical_dual_gpu import LogicalDualGPU, isolate_cublas_workspaces

MODES = ('native', 'independent', 'global_rope', 'shared_kv_global',
         'full_query_control', 'query_block_control', 'aligned')


def quality(a, b):
    # Compare on CPU outside performance windows; avoid full-video FP32 VRAM.
    a = a.detach().cpu().float(); b = b.detach().cpu().float()
    delta = a-b
    return dict(bit_exact=torch.equal(a,b), finite=bool(torch.isfinite(a).all()),
                rel_l2=float(delta.norm()/b.norm().clamp_min(1e-20)),
                max_abs=float(delta.abs().max()))


class DecoderRun:
    def __init__(self, model, z, comm, rank, mode):
        self.model, self.z, self.comm, self.rank, self.mode = model,z,comm,rank,mode
        self.buffers = {}; self.events = []; self.profiling = False
        b,_,t,h,w = z.shape; self.patches = t*h*w; self.length = self.patches+5
        self.plan = SlabPlan.create(w,128)
        cut = ((self.length+255)//256)*128
        self.ranges = [(0,cut),(cut,self.length)]
        grid = torch.arange(self.patches,device=z.device).view(t,h,w)
        suffix = torch.arange(self.patches,self.length,device=z.device)
        self.ids = [torch.cat([grid[...,:self.plan.left_end].flatten(),suffix]),
                    torch.cat([grid[...,self.plan.right_start:].flatten(),suffix])]
        owners=[]
        for r,ids in enumerate(self.ids):
            patch=ids<self.patches
            mask=patch & ((ids%w<w//2) if r==0 else (ids%w>=w//2))
            if r==0: mask|=~patch
            owners.append(torch.where(mask)[0])
        self.owners=owners
        self.order=torch.cat([self.ids[r][owners[r]] for r in range(2)]).argsort()
        self.coords=create_token_ids((t,h,w),z.device,z.dtype).expand(b,-1,-1)
        self.coords=torch.cat([self.coords,torch.zeros((b,5,3),device=z.device,dtype=z.dtype)],1)
        if mode=='global_rope':
            original=model.decoder.pos_embed.forward
            model.decoder.pos_embed.forward=lambda unused:original(self.coords.index_select(1,self.ids[rank]))
        elif mode in ('shared_kv_global','full_query_control','query_block_control','aligned'):
            for block in model.decoder.transformer_blocks:
                block.attn._profile_run=self
                block.attn.forward=MethodType(parallel_attention,block.attn)

    @contextmanager
    def stage(self,name):
        if not self.profiling:
            yield; return
        start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        with torch.profiler.record_function(name):
            start.record(); yield; end.record()
        self.events.append((name,start,end))

    def exchange(self, x, sizes, name, dim):
        """Known sizes, one collective, reused transport buffers, no .item()."""
        # Transpose sequence to dimension zero so a single NCCL allgather works.
        y=x.movedim(dim,0).contiguous(); maximum=max(sizes)
        key=(name,tuple(y.shape[1:]),maximum,y.dtype)
        if key not in self.buffers:
            padded=torch.empty((maximum,*y.shape[1:]),dtype=y.dtype,device=y.device)
            full=torch.empty((2*maximum,*y.shape[1:]),dtype=y.dtype,device=y.device)
            self.buffers[key]=(padded,full)
        padded,full=self.buffers[key]
        padded[:y.shape[0]].copy_(y)
        if y.shape[0]<maximum:padded[y.shape[0]:].zero_()
        self.comm.all_gather_into_tensor(full,padded)
        return torch.cat([full[:sizes[0]],full[maximum:maximum+sizes[1]]],0).movedim(0,dim).contiguous()

    def run(self):
        m=self.model; b,_,t,h,w=self.z.shape
        if self.mode=='native':return m._decode_pixels(self.z) if self.rank==0 else None
        if self.mode in ('independent','global_rope'):
            local=m._decode_pixels(self.plan.slices(self.z)[self.rank].contiguous())
            full=self.exchange(local,[self.plan.left_end*16,(w-self.plan.right_start)*16],'pixels',-1)
            return stitch_slabs(full[...,:self.plan.left_end*16],full[...,self.plan.left_end*16:],self.plan)
        decoder=m.decoder
        if self.mode=='aligned':
            lo,hi=self.ranges[self.rank]; otherlo,otherhi=self.ranges[1]
            # Full Native startup projection once; broadcast only rank one's
            # owned hidden tokens. This transfer is included in decoder timing.
            with self.stage('prefix_projection_and_handoff'):
                remote=torch.empty((b,otherhi-otherlo,2048),dtype=self.z.dtype,device=self.z.device)
                if self.rank==0:
                    h0=decoder.x_embedder(m.post_quant_conv(self.z).flatten(2).transpose(1,2))
                    h0=torch.cat([h0,comfy.ops.cast_to_input(decoder.register_tokens,h0).expand(b,-1,-1),torch.zeros_like(h0[:,:1])],1)
                    remote.copy_(h0[:,otherlo:otherhi]); local=h0[:,lo:hi].contiguous()
                    del h0
                self.comm.broadcast(remote,src=0)
                if self.rank==1:local=remote
                rope=decoder.pos_embed(self.coords[:,lo:hi].contiguous())
        else:
            ids=self.ids[self.rank]
            x=m.post_quant_conv(self.plan.slices(self.z)[self.rank].contiguous())
            local=decoder.x_embedder(x.flatten(2).transpose(1,2))
            local=torch.cat([local,comfy.ops.cast_to_input(decoder.register_tokens,local).expand(b,-1,-1),torch.zeros_like(local[:,:1])],1)
            rope=decoder.pos_embed(self.coords.index_select(1,ids))
        for block in decoder.transformer_blocks:
            with self.stage('transformer_block'):local=block(local,rope)
        with self.stage('output_projection_and_join'):
            local=decoder.proj_out(decoder.norm_out(local))
            if self.mode=='aligned':
                projected=self.exchange(local,[hi-lo for lo,hi in self.ranges],'projected',1)[:,:self.patches]
            else:
                owned=local.index_select(1,self.owners[self.rank])
                projected=self.exchange(owned,[len(o) for o in self.owners],'projected',1).index_select(1,self.order)[:,:self.patches]
            projected=projected.view(b,t,h,w,decoder.out_channels,decoder.patch_size_t,decoder.patch_size,decoder.patch_size)
            return projected.permute(0,4,1,5,2,6,3,7).contiguous().reshape(b,decoder.out_channels,t*4,h*16,w*16)


def parallel_attention(self,x,rotary_pos_emb,pre_norm,residual,residual_scale):
    run=self._profile_run; b,s,_=x.shape
    with run.stage('qkv_projection_rope'):
        with comfy.ops.CastBiasWeightContext(pre_norm,x,offloadable=True) as (weight,bias):
            qkv=comfy.ops.linear_input_act(self.to_qkv,x,'rms_norm',weight,pre_norm.eps)
        q,k,v=qkv.view(b,s,-1,3*self.dim_head).chunk(3,-1)
        q,k=comfy.quant_ops.ck.rms_rope_split_half_(q,k,rotary_pos_emb,self.qk_norm_scale.to(q.device),epsilon=self.norm_q.eps,rot_dim=rotary_pos_emb.shape[-3]*2)
        q,k,v=(a.transpose(1,2) for a in (q,k,v))
    with run.stage('kv_pack_collective_layout'):
        if run.mode=='aligned':
            sizes=[hi-lo for lo,hi in run.ranges]; owned=torch.stack((k,v))
        else:
            sizes=[len(o) for o in run.owners]
            owned=torch.stack((k.index_select(2,run.owners[run.rank]),v.index_select(2,run.owners[run.rank])))
        full=run.exchange(owned,sizes,'kv',3)
        if run.mode!='aligned':full=full.index_select(3,run.order)
        full_k,full_v=full.unbind(0)
    with run.stage('query_control_collectives_and_attention' if run.mode in ('full_query_control','query_block_control') else 'native_int8_attention'):
        if run.mode in ('full_query_control','query_block_control'):
            fq=run.exchange(q.index_select(2,run.owners[run.rank]),sizes,'query',2).index_select(2,run.order)
            if run.mode=='full_query_control':out=comfy.quant_ops.ck.int8_attention(fq,full_k,full_v).index_select(2,run.ids[run.rank])
            else:
                packed=comfy.quant_ops.ck.prequantize_int8_attention(fq,full_k,full_v)
                lo,hi=run.ranges[run.rank]
                local=replace(packed,q=packed.q[:,:,lo:hi].contiguous(),q_scale=packed.q_scale[:,:,lo//4:((hi+127)//128)*32].contiguous())
                partial=comfy.quant_ops.ck.int8_attention_from_prequantized(local)
                out=run.exchange(partial,[h-l for l,h in run.ranges],'attention_output',2).index_select(2,run.ids[run.rank])
        else:out=comfy.quant_ops.ck.int8_attention(q,full_k,full_v)
    with run.stage('attention_output_projection'):
        out=out.transpose(1,2).reshape(b,s,-1)
        return comfy.ops.linear_input_act(self.to_out,torch.nan_to_num(out),None,residual=residual,residual_scale=residual_scale)


def inputs():
    for name in ('768x768_1F','768x1344_2F'):
        path=ROOT/'slab_results/saved_references'/('input_'+name+'.pt')
        cpu=torch.load(path,map_location='cpu',weights_only=True)
        rows=json.loads((path.parent/'manifest.json').read_text())
        row=next(r for r in rows if r['file']==path.name)
        assert hashlib.sha256(cpu.numpy().tobytes()).hexdigest()==row['tensor_sha256']
        yield name,cpu,'saved random benchmark latent'
    # Realistic 512px / 124 decoded frames (~5 seconds at 25fps), random fixture
    # values; geometry alone does not establish generated-video visual quality.
    gen=torch.Generator().manual_seed(1729)
    yield '512x512_124frames',torch.randn((1,24,31,32,32),generator=gen).half(),'synthetic latent, realistic 124-frame geometry'


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2));tmp.replace(path)


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--qualify',action='store_true')
    p.add_argument('--iterations',type=int,default=5);p.add_argument('--warmups',type=int,default=2)
    args=p.parse_args();torch.set_grad_enabled(False);torch.backends.cuda.matmul.allow_tf32=False
    assert hashlib.file_digest(args.checkpoint.open('rb'),'sha256').hexdigest()=='52a2c8c73583c86e4f41cdcce3a6ad0ea562987bc0bf3d60a0cef5f5c8e60c0e'
    if args.qualify:
        isolate_cublas_workspaces();comm=LogicalDualGPU(timeout=120);ranks=[0,1];devices=['cuda:0']*2
    else:
        assert torch.cuda.device_count()==2 and all('T4' in torch.cuda.get_device_name(i) for i in range(2))
        rank=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(rank)
        dist.init_process_group('nccl',timeout=timedelta(seconds=180));comm=dist;ranks=[rank];devices=[f'cuda:{rank}']
        assert dist.get_world_size()==2
    models=[]
    for device in devices:
        with torch.cuda.device(device):model,sd=load_video_native(args.checkpoint,device);del sd;models.append(model)
    originals=[]
    for model in models:
        originals.append((model.decoder.pos_embed.forward,[block.attn.forward for block in model.decoder.transformer_blocks]))
    result=dict(physical=not args.qualify,gpu=torch.cuda.get_device_name(),torch=torch.__version__,
                kitchen=importlib.metadata.version('comfy-kitchen'),cases=[],completed=False,
                nccl_policy={key:os.environ.get(key) for key in ('NCCL_P2P_DISABLE','NCCL_IB_DISABLE','NCCL_DEBUG')},
                boundary='GPU-resident latent to GPU-resident complete raw pixels on rank0, includes prefix/output transfers; model loading excluded',
                limits=['Raw decoder only; public VAE normalization/cropping excluded.','Approximate outputs receive no accepted speedup.','Profiler iteration excluded from latency samples.'],
                source_sha256={str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in (Path(__file__),ROOT/'ComfyUI/comfy/ldm/minimax/vae.py')})
    def save():
        if args.qualify or ranks[0]==0:write(args.output/'results.json',result)
    save()
    for name,cpu,fixture in inputs():
        zs=[cpu.to(d) for d in devices];native=None
        for mode in MODES:
            for model,(position,attention) in zip(models,originals):
                model.decoder.pos_embed.forward=position
                for block,forward in zip(model.decoder.transformer_blocks,attention):block.attn.forward=forward
            runs=[DecoderRun(model,z,comm,r,mode) for model,z,r in zip(models,zs,ranks)]
            @torch.inference_mode()
            def call():
                if args.qualify:return comm.run([run.run for run in runs])[0]
                return runs[0].run()
            with torch.inference_mode():out=call()
            if args.qualify or ranks[0]==0:
                if mode=='native':native=out.cpu()
                check=quality(out,native)
                row=dict(workload=name,shape=list(cpu.shape),fixture=fixture,mode=mode,correctness=check,
                         accepted=check['bit_exact'] and check['finite'],query_ranges=runs[0].ranges,
                         input_sha256=hashlib.sha256(cpu.numpy().tobytes()).hexdigest())
                result['cases'].append(row);save();print('VAE CHECK',name,mode,check,flush=True)
            del out
            if args.qualify:
                # Retain every diagnostic failure. Qualification requires the
                # deployable candidate, not cropped-prefix diagnostic controls.
                continue
            # Diagnostic ablations can be profiled even if incorrect, with their
            # failures retained. Exact paths must pass before performance ranking.
            for _ in range(args.warmups):call();torch.cuda.synchronize()
            torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
            samples=[];gpu_samples=[]
            for _ in range(args.iterations):
                dist.barrier();torch.cuda.synchronize()
                start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                before=time.perf_counter();start.record();out=call();end.record();end.synchronize()
                wall=(time.perf_counter()-before)*1000
                timing=torch.tensor([wall,start.elapsed_time(end)],device=zs[0].device,dtype=torch.float64)
                dist.all_reduce(timing,op=dist.ReduceOp.MAX)
                samples.append(float(timing[0]));gpu_samples.append(float(timing[1]));del out
            memory=dict(peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved(),transport_buffers=sum(t.numel()*t.element_size() for pair in runs[0].buffers.values() for t in pair))
            mem=[None]*2;dist.all_gather_object(mem,memory)
            # One separate traced decode per mode on both ranks.
            runs[0].profiling=True;dist.barrier()
            hooks=[];active={}
            def begin(label):
                def hook(module,args):
                    ctx=runs[0].stage(label);ctx.__enter__();active[id(module)]=ctx
                return hook
            def finish(module,args,out):active.pop(id(module)).__exit__(None,None,None)
            for block in models[0].decoder.transformer_blocks:
                for module,label in ((block.ff,'native_ffn'),) if hasattr(block,'ff') else ((block.ffn,'native_ffn'),):
                    hooks.extend([module.register_forward_pre_hook(begin(label)),module.register_forward_hook(finish)])
                if mode in ('native','independent','global_rope'):
                    hooks.extend([block.attn.register_forward_pre_hook(begin('native_attention_total')),block.attn.register_forward_hook(finish)])
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],record_shapes=True,profile_memory=True) as prof:
                traced=call();torch.cuda.synchronize()
            for hook in hooks:hook.remove()
            del traced;runs[0].profiling=False
            stem=args.output/f'{name}_{mode}_rank{ranks[0]}'
            prof.export_chrome_trace(str(stem.with_suffix('.json.gz')))
            stem.with_suffix('.txt').write_text(prof.key_averages().table(sort_by='self_cuda_time_total',row_limit=60))
            stages={}
            for label,a,b in runs[0].events:stages.setdefault(label,[]).append(a.elapsed_time(b))
            write(stem.with_suffix('.stages.json'),{label:dict(total_ms=sum(vals),calls=len(vals),mean_ms=statistics.mean(vals)) for label,vals in stages.items()})
            if ranks[0]==0:
                row.update(wall_ms_samples=samples,cuda_ms_samples=gpu_samples,median_wall_ms=statistics.median(samples),median_cuda_ms=statistics.median(gpu_samples),rank_memory=mem,
                           profiler_prefix=stem.name,profiled_stage_ms={key:sum(vals) for key,vals in stages.items()})
                save();print('VAE TIMING',name,mode,row['median_wall_ms'],'accepted',row['accepted'],flush=True)
            del runs
        del zs,native
    result['completed']=True
    if args.qualify:result['passed']=all(r['accepted'] for r in result['cases'] if r['mode'] in ('native','aligned'))
    save()
    if not args.qualify:dist.destroy_process_group()


if __name__=='__main__':main()
