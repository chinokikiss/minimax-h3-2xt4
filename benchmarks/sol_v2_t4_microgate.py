"""Minimal physical-T4 gate for Sol v2: no model weights, no TP2, no generation."""
from __future__ import annotations
import argparse,json,math,statistics,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
import torch.nn.functional as F
from patches import sol_attention
from patches import sol_sm75_backend as native

def view(shape,seed):
    b,h,l,d=shape
    g=torch.Generator(device='cpu').manual_seed(seed)
    cpu=torch.randn((b,l,h,d),generator=g,dtype=torch.float16)
    return cpu.cuda().transpose(1,2)

def metrics(out,ref):
    a=out.float();b=ref.float();delta=a-b
    denom=torch.linalg.vector_norm(b).clamp_min(1e-12)
    return dict(
        rel_l2=float(torch.linalg.vector_norm(delta)/denom),
        mae=float(delta.abs().mean()),max_abs=float(delta.abs().max()),
        rms=float(delta.square().mean().sqrt()),
        cosine=float(F.cosine_similarity(a.flatten(),b.flatten(),dim=0)),
        finite=bool(torch.isfinite(a).all()))

def wall_ms(fn,repeats=5,warmup=2):
    for _ in range(warmup):fn()
    torch.cuda.synchronize()
    samples=[]
    for _ in range(repeats):
        begin=time.perf_counter();fn();torch.cuda.synchronize()
        samples.append((time.perf_counter()-begin)*1000)
    return dict(median_ms=statistics.median(samples),min_ms=min(samples),samples_ms=samples)

def peak_temp(fn):
    torch.cuda.synchronize();torch.cuda.empty_cache();torch.cuda.synchronize()
    base=torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    fn();torch.cuda.synchronize()
    peak=torch.cuda.max_memory_allocated()
    return int(max(0,peak-base))

def old_d64(q,k,v,*,protected_tokens=0,protected_start=0,tau=1.0,sync_free_lut=True):
    q128=F.pad(q,(0,64)).mul_(math.sqrt(2.0))
    k128=F.pad(k,(0,64));v128=F.pad(v,(0,64))
    out=sol_attention.attention(q128,k128,v128,protected_tokens=protected_tokens,
        protected_start=protected_start,tau=tau,sync_free_lut=sync_free_lut,
        collect_route_stats=False)
    return out[...,:64].contiguous()

def new_d64(q,k,v,*,protected_tokens=0,protected_start=0,tau=1.0,topology=None,sync_free_lut=True):
    return sol_attention.attention(q,k,v,protected_tokens=protected_tokens,
        protected_start=protected_start,tau=tau,allow_d64=True,topology=topology,
        sync_free_lut=sync_free_lut,collect_route_stats=False)

def route_density(r):
    return float(r['row_count'].sum().item()/(r['row_count'].numel()*r['exact_mask'].shape[-1]))

def reference_native_path(q,k,v,*,protected_tokens=0,protected_start=0,tau=1.0,topology=None,sync_free_lut=True):
    """Pre-refactor operation order using native.prepare -> quantize KC -> native.run."""
    routing=native.prepare(q,k,v,tau=float(tau),sink_tokens=int(protected_tokens),
        sink_start=int(protected_start) if protected_tokens else None,quantize_qk=True,
        topology=topology,sync_free_lut=bool(sync_free_lut),collect_route_stats=False)
    kc,kcs=native.quantize(routing['k_centroid'],64)
    approx=(kc,routing['v_centroid'],kcs,routing['exact_mask'],
            int(routing['centroid_count']),int(routing['centroid_padded']))
    return native.run(routing['q_int8'],routing['k_int8'],v,routing['q_scale'],
        routing['k_scale'],routing['row_count'],routing['lut'],all_int8=True,
        approximation=approx,validated_routing=bool(routing.get('validated_routing',False)))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args();args.output.parent.mkdir(parents=True,exist_ok=True)
    state=dict(status='starting',scope='single-T4 zero-weight Sol v2 micro-gate; no H3 generation claim')
    def save():args.output.write_text(json.dumps(state,indent=2))
    save()
    try:
        assert torch.cuda.is_available() and torch.cuda.get_device_capability(0)==(7,5)
        torch.cuda.set_device(0);torch.set_grad_enabled(False)
        torch.backends.cuda.matmul.allow_tf32=False
        state['hardware']=dict(name=torch.cuda.get_device_name(0),capability=list(torch.cuda.get_device_capability(0)))
        state['native_library']=sol_attention.prepare_native()
        state['abi']=int(native._load_library().star7_sla_sm75_abi_version())
        state['cases']={};save()

        # 1. Native D64 must reproduce the old pad-D128 semantics to quantization tolerance
        # when routing is fully exact. This isolates the D64 port from sparsity.
        q=view((1,4,1029,64),1);k=view((1,4,1029,64),2);v=view((1,4,1029,64),3)
        dense=F.scaled_dot_product_attention(q,k,v)
        old=old_d64(q,k,v,protected_tokens=1029,protected_start=0)
        new=new_d64(q,k,v,protected_tokens=1029,protected_start=0)
        state['cases']['d64_exact_semantics']=dict(
            shape=list(q.shape),stride=list(q.stride()),
            old_vs_dense=metrics(old,dense),new_vs_dense=metrics(new,dense),
            new_vs_old=metrics(new,old))
        del dense,old,new,q,k,v;torch.cuda.empty_cache();save()

        # The fused preprocess must materialize exactly four Q16 scales per Q64
        # block, including 1.0 sentinel scales for empty tail warps.
        q=view((1,4,1029,128),4);k=view((1,4,1029,128),5);v=view((1,4,1029,128),6)
        routing=native.prepare(q,k,v,tau=1.0,quantize_qk=True,
            sync_free_lut=True,collect_route_stats=False)
        blocks=(q.shape[-2]+63)//64
        tail=routing['q_scale'][..., -4:].detach().cpu()
        prepared=sol_attention.prepare_attention(q,k,v,tau=1.0,
            sync_free_lut=True,collect_route_stats=False)
        prepared_out=sol_attention.execute_prepared(prepared)
        reference_out=reference_native_path(q,k,v,tau=1.0,sync_free_lut=True)
        state['cases']['prepared_and_qscale']=dict(
            q_scale_shape=list(routing['q_scale'].shape),expected_last_dim=blocks*4,
            tail_scales=tail.tolist(),
            empty_tail_scales_are_one=bool(torch.equal(tail[...,1:],torch.ones_like(tail[...,1:]))),
            prepared_vs_reference=metrics(prepared_out,reference_out))
        del routing,prepared,prepared_out,reference_out,q,k,v;torch.cuda.empty_cache();save()

        # 2. D64 performance: emulate the old pad/sqrt2/slice path against native D64.
        q=view((1,32,1285,64),11);k=view((1,32,1285,64),12);v=view((1,32,1285,64),13)
        old_fn=lambda:old_d64(q,k,v,protected_tokens=5,protected_start=1280,tau=1.0)
        new_fn=lambda:new_d64(q,k,v,protected_tokens=5,protected_start=1280,tau=1.0)
        old_t=wall_ms(old_fn,7);new_t=wall_ms(new_fn,7)
        state['cases']['d64_perf']=dict(
            shape=list(q.shape),stride=list(q.stride()),old_pad_d128=old_t,native_d64=new_t,
            speedup_old_over_native=old_t['median_ms']/new_t['median_ms'],
            old_peak_temporary=peak_temp(old_fn),native_peak_temporary=peak_temp(new_fn))
        del q,k,v;torch.cuda.empty_cache();save()

        # 3. Same D128 routing, compact-LUT host sync versus sync-free full-stride LUT.
        q=view((1,28,8192,128),21);k=view((1,28,8192,128),22);v=view((1,28,8192,128),23)
        old_fn=lambda:sol_attention.attention(q,k,v,tau=1.0,sync_free_lut=False,collect_route_stats=True)
        new_fn=lambda:sol_attention.attention(q,k,v,tau=1.0,sync_free_lut=True,collect_route_stats=False)
        old_out=old_fn();new_out=new_fn()
        routing=native.prepare(q,k,v,tau=1.0,quantize_qk=True,sync_free_lut=False,collect_route_stats=True)
        old_t=wall_ms(old_fn,5);new_t=wall_ms(new_fn,5)
        state['cases']['d128_sync_free_lut']=dict(
            shape=list(q.shape),stride=list(q.stride()),same_semantics=metrics(new_out,old_out),
            route_density=float(routing['density']),compact_lut=old_t,sync_free_lut=new_t,
            speedup_compact_over_sync_free=old_t['median_ms']/new_t['median_ms'],
            compact_peak_temporary=peak_temp(old_fn),sync_free_peak_temporary=peak_temp(new_fn))
        del q,k,v,old_out,new_out,routing;torch.cuda.empty_cache();save()

        # 4. H3 token-space topology: actual modulo geometry (frame stride 836, video starts at token %64 == 46).
        video_start=302;frame_stride=836;length=video_start+3*frame_stride
        q=torch.zeros((1,2,length,128),device='cuda',dtype=torch.float16)
        k=torch.zeros_like(q);v=torch.ones_like(q)
        old_r=native.prepare(q,k,v,tau=100.0,sink_tokens=video_start,sink_start=0,
            quantize_qk=False,sync_free_lut=True,collect_route_stats=False)
        topo=dict(kind='dit',start_token=video_start,end_token=length,
            frame_stride_tokens=frame_stride,temporal_radius=1,local_radius_tokens=64,
            protect_frame_boundaries=True)
        new_r=native.prepare(q,k,v,tau=100.0,sink_tokens=video_start,sink_start=0,
            quantize_qk=False,topology=topo,sync_free_lut=True,collect_route_stats=False)
        qtok=video_start+frame_stride+200;ktok=video_start+200
        qb,kb=qtok//64,ktok//64
        boundary=(video_start+frame_stride)//64
        far_q=(video_start+2*frame_stride+500)//64
        prefix_blocks=(video_start+63)//64
        state['cases']['dit_topology']=dict(
            length=length,video_start=video_start,frame_stride_tokens=frame_stride,
            cross_frame_probe=dict(q_block=qb,k_block=kb,old=bool(old_r['exact_mask'][0,0,qb,kb]),
                                   new=bool(new_r['exact_mask'][0,0,qb,kb])),
            boundary_key_probe=dict(q_block=far_q,k_block=boundary,
                                    old=bool(old_r['exact_mask'][0,0,far_q,boundary]),
                                    new=bool(new_r['exact_mask'][0,0,far_q,boundary])),
            protected_prefix_bidirectional=bool(new_r['exact_mask'][...,:prefix_blocks].all() and
                                                new_r['exact_mask'][...,:prefix_blocks,:].all()),
            old_density=route_density(old_r),new_density=route_density(new_r))
        assert not state['cases']['dit_topology']['cross_frame_probe']['old']
        assert state['cases']['dit_topology']['cross_frame_probe']['new']
        assert state['cases']['dit_topology']['boundary_key_probe']['new']
        assert state['cases']['dit_topology']['protected_prefix_bidirectional']
        del q,k,v,old_r,new_r;torch.cuda.empty_cache();save()

        # 4b. Query-block subset launch must be exactly the same for computed
        # rows as a full launch from the identical prepared buffers.
        q=view((1,4,1285,64),31);k=view((1,4,1285,64),32);v=view((1,4,1285,64),33)
        topo=dict(kind='video_vae',start_token=0,end_token=1280,
            frame_stride_tokens=256,temporal_radius=1,local_radius_tokens=64,
            protect_frame_boundaries=False)
        prepared=sol_attention.prepare_attention(
            q,k,v,protected_tokens=5,protected_start=1280,tau=1.0,
            allow_d64=True,topology=topo,sync_free_lut=True,
            collect_route_stats=False)
        full=sol_attention.execute_prepared(prepared)
        base,count=8,6
        subset=sol_attention.execute_prepared(
            prepared,q_block_base=base,q_block_count=count)
        lo=base*64;hi=min((base+count)*64,q.shape[-2])
        state['cases']['query_block_subset']=dict(
            block_base=base,block_count=count,token_range=[lo,hi],
            subset_vs_full=metrics(subset[:,:,lo:hi],full[:,:,lo:hi]))
        del q,k,v,prepared,full,subset;torch.cuda.empty_cache();save()

        # 5. VAE topology: 4x256 latent-frame tokens + five global suffix tokens.
        content=1024;length=1029
        q=torch.zeros((1,2,length,64),device='cuda',dtype=torch.float16)
        k=torch.zeros_like(q);v=torch.ones_like(q)
        old_r=native.prepare(q,k,v,tau=100.0,sink_tokens=5,sink_start=content,
            quantize_qk=False,sync_free_lut=True,collect_route_stats=False)
        topo=dict(kind='video_vae',start_token=0,end_token=content,
            frame_stride_tokens=256,temporal_radius=1,local_radius_tokens=64,
            protect_frame_boundaries=False)
        new_r=native.prepare(q,k,v,tau=100.0,sink_tokens=5,sink_start=content,
            quantize_qk=False,topology=topo,sync_free_lut=True,collect_route_stats=False)
        qb=(2*256+128)//64;kb=(1*256+128)//64;global_block=content//64;far_q=64//64
        state['cases']['vae_topology']=dict(
            length=length,frame_stride_tokens=256,
            cross_frame_probe=dict(q_block=qb,k_block=kb,old=bool(old_r['exact_mask'][0,0,qb,kb]),
                                   new=bool(new_r['exact_mask'][0,0,qb,kb])),
            global_suffix_bidirectional=bool(new_r['exact_mask'][...,global_block].all() and
                                             new_r['exact_mask'][...,global_block,:].all()),
            old_density=route_density(old_r),new_density=route_density(new_r))
        assert not state['cases']['vae_topology']['cross_frame_probe']['old']
        assert state['cases']['vae_topology']['cross_frame_probe']['new']
        assert state['cases']['vae_topology']['global_suffix_bidirectional']

        # 6. True local-Q/full-KV query partitioning.
        length=1029;content=1024;cut=((length+255)//256)*128
        q=view((1,8,length,64),31);k=view((1,8,length,64),32);v=view((1,8,length,64),33)
        topo=dict(kind='video_vae',start_token=0,end_token=content,
            frame_stride_tokens=256,temporal_radius=1,local_radius_tokens=64,
            protect_frame_boundaries=False)
        full_fn=lambda:sol_attention.attention(
            q,k,v,protected_tokens=5,protected_start=content,tau=1.0,
            allow_d64=True,topology=topo,sync_free_lut=True,
            collect_route_stats=False)
        full=full_fn()
        parts=[];part_times=[];part_peaks=[]
        for lo,hi in ((0,cut),(cut,length)):
            q_local=q[:,:,lo:hi]
            fn=lambda q_local=q_local,lo=lo:sol_attention.attention_partition(
                q_local,k,v,q_token_offset=lo,protected_tokens=5,
                protected_start=content,tau=1.0,topology=topo,
                sync_free_lut=True,collect_route_stats=False)
            out=fn()
            parts.append(out.contiguous())
            part_times.append(wall_ms(fn,5,2))
            part_peaks.append(peak_temp(fn))
        joined=torch.cat(parts,dim=2)
        full_t=wall_ms(full_fn,5,2)
        full_peak=peak_temp(full_fn)
        state['cases']['query_partition']=dict(
            shape=list(q.shape),cut=cut,
            joined_vs_full=metrics(joined,full),
            full=full_t,rank0=part_times[0],rank1=part_times[1],
            full_peak_temporary=full_peak,
            rank0_peak_temporary=part_peaks[0],
            rank1_peak_temporary=part_peaks[1],
            max_rank_over_full=max(x['median_ms'] for x in part_times)/full_t['median_ms'],
            max_rank_peak_over_full=max(part_peaks)/full_peak)
        del q,k,v,full,joined,parts;torch.cuda.empty_cache();save()

        state['acceptance']=dict(
            physical_sm75=True,abi12=state['abi']==12,
            d64_native_finite=state['cases']['d64_exact_semantics']['new_vs_dense']['finite'],
            d64_port_close_to_old=state['cases']['d64_exact_semantics']['new_vs_old']['rel_l2']<0.02,
            qscale_direct_shape=state['cases']['prepared_and_qscale']['q_scale_shape'][-1]==state['cases']['prepared_and_qscale']['expected_last_dim'],
            qscale_tail_initialized=state['cases']['prepared_and_qscale']['empty_tail_scales_are_one'],
            prepared_semantics=state['cases']['prepared_and_qscale']['prepared_vs_reference']['rel_l2']<1e-5,
            d64_no_perf_regression=state['cases']['d64_perf']['speedup_old_over_native']>=0.95,
            sync_free_semantics=state['cases']['d128_sync_free_lut']['same_semantics']['rel_l2']<1e-5,
            sync_free_no_perf_regression=state['cases']['d128_sync_free_lut']['speedup_compact_over_sync_free']>=0.95,
            query_subset_semantics=state['cases']['query_block_subset']['subset_vs_full']['rel_l2']<1e-5,
            dit_topology_exact=True,vae_topology_exact=True,
            query_partition_semantics=state['cases']['query_partition']['joined_vs_full']['rel_l2']<1e-5)
        state['status']='complete' if all(state['acceptance'].values()) else 'gate_failed'
        save()
        if state['status']!='complete':raise AssertionError(state['acceptance'])
    except BaseException as exc:
        state.update(status='failed',error=type(exc).__name__+': '+str(exc))
        state['traceback']=traceback.format_exc();save();raise

if __name__=='__main__':main()

