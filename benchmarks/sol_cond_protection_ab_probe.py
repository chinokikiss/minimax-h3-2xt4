"""Single-T4 Sol-v2 conditioning-protection A/B on identical QKV.

Runs in one of two isolated source trees: original bidirectional or
benchmark-only key-protected (one-way). Never modifies production files.
"""
from __future__ import annotations
import argparse, hashlib, json, math, statistics, time
from pathlib import Path
import torch

N=65166
VIDEO_START=4974
AUDIO_START=4164
FRAMES=72
FRAME_TOKENS=836
BLOCK=64
EXPECTED={
    0:"4c9116fd7f603539c4e49e7e440c034c893e246f8649371b21376483fafd34e8",
    1:"9138f989a127295555ace8ca3914e007c0f3d6d9cd5f26f296c48789ad80bc91",
}

def digest(path:Path)->str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream,"sha256").hexdigest()

def gpu_time(fn,repeats=5,warmup=2):
    for _ in range(warmup):
        value=fn()
        del value
    torch.cuda.synchronize()
    wall=[]
    events=[]
    for _ in range(repeats):
        a=torch.cuda.Event(enable_timing=True)
        b=torch.cuda.Event(enable_timing=True)
        a.record()
        start=time.perf_counter()
        value=fn()
        b.record()
        b.synchronize()
        wall.append((time.perf_counter()-start)*1000)
        events.append(a.elapsed_time(b))
        del value
    return dict(wall_median_ms=statistics.median(wall),
                gpu_event_median_ms=statistics.median(events),
                wall_samples_ms=wall,gpu_event_samples_ms=events)

def counts(p):
    m=p["exact_mask"]
    if m is None:raise RuntimeError("A/B requires exact_mask")
    rows=m.shape[-2]
    pb=(VIDEO_START+BLOCK-1)//BLOCK
    return dict(total=int(m.sum().item()),all_pairs=int(m.numel()),
        prefix_to_video=int(m[..., :pb,pb:].sum().item()),
        prefix_to_video_possible=int(m[..., :pb,pb:].numel()),
        video_to_prefix=int(m[..., pb:,:pb].sum().item()),
        video_to_prefix_possible=int(m[..., pb:,:pb].numel()),
        prefix_to_prefix=int(m[..., :pb,:pb].sum().item()),
        prefix_blocks=pb,all_blocks=rows)

def input_synthetic(shape,seed):
    b,h,n,d=shape
    values=[]
    for i in range(3):
        g=torch.Generator(device="cpu").manual_seed(seed+i)
        t=torch.randn((b,n,h,d),dtype=torch.float16,generator=g)
        values.append(t.cuda().transpose(1,2))
    return values

def sampled_dense(q,k,v):
    # 85 actual queries, all 65k K/V; no dense NxN materialization.
    indices=sorted(set([0,1,1264,2527,2528,3343,3344,4163,
        AUDIO_START,4550,VIDEO_START-1,VIDEO_START,N-1]+[
            VIDEO_START+i*FRAME_TOKENS+FRAME_TOKENS//2 for i in range(FRAMES)]))
    idx=torch.tensor(indices,device="cuda")
    qi=q[:,:,idx].float()
    ki=k.float()
    vi=v.float()
    logits=torch.matmul(qi,ki.transpose(-1,-2))*(q.shape[-1]**-0.5)
    dense=torch.matmul(torch.softmax(logits,dim=-1),vi)
    return indices,dense

def one_case(name,q,k,v,sol,outdir,*,store=False):
    topology=dict(kind="dit",start_token=VIDEO_START,end_token=N,
        frame_stride_tokens=FRAME_TOKENS,temporal_radius=1,
        local_radius_tokens=64,protect_frame_boundaries=True)
    args=dict(protected_tokens=VIDEO_START,protected_start=0,
        tau=1.0,topology=topology,sync_free_lut=True,collect_route_stats=False)
    prepare=lambda:sol.prepare_attention(q,k,v,**args)
    prepared=prepare()
    routed=counts(prepared)
    execute=lambda:sol.execute_prepared(prepared)
    result=execute()
    output=dict(shape=list(q.shape),routed=routed,finite=bool(torch.isfinite(result).all()))
    if not output["finite"]:
        raise ValueError(name+" generated NaN/Inf")
    output["prepare"]=gpu_time(prepare)
    output["execute"]=gpu_time(execute)
    output["complete"]=gpu_time(lambda:sol.attention(q,k,v,**args))
    if store:
        indices,dense=sampled_dense(q,k,v)
        torch.save(dict(output=result.detach().cpu(),mask=prepared["exact_mask"].detach().cpu(),
            dense_indices=indices,dense=dense.detach().cpu()),
            outdir/(name+"_comparison.pt"))
        output["saved_output_and_dense"]=True
        output["sampled_dense_queries"]=len(indices)
    del result,prepared
    torch.cuda.synchronize()
    return output

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--variant",choices=("bidirectional","key_only"),required=True)
    ap.add_argument("--out",type=Path,required=True)
    ap.add_argument("--capture-root",type=Path)
    ap.add_argument("--repeat-id",type=int,required=True)
    args=ap.parse_args()
    args.out.mkdir(parents=True,exist_ok=False)
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32=False
    assert torch.cuda.is_available() and torch.cuda.get_device_capability(0)==(7,5)
    torch.cuda.set_device(0)
    from patches import sol_attention as sol
    from patches import sol_sm75_backend as native
    library=sol.prepare_native()
    assert int(native._load_library().star7_sla_sm75_abi_version())==12
    record=dict(status="starting",variant=args.variant,repeat_id=args.repeat_id,
        gpu=torch.cuda.get_device_name(0),torch=torch.__version__,library=library,
        capture_source=None,cases={})
    def save():(args.out/"result.json").write_text(json.dumps(record,indent=2))
    save()
    try:
        if args.capture_root:
            for rank in (0,1):
                file=next(iter(args.capture_root.rglob(f"rank{rank}_real_qkv.pt")),None)
                if file is None:raise FileNotFoundError(f"rank{rank}_real_qkv.pt missing")
                assert file.stat().st_size==66732515
                assert digest(file)==EXPECTED[rank],("Real QKV SHA mismatch",rank)
                sample=torch.load(file,map_location="cpu",weights_only=True)
                for key in ("q","k","v","sol_v2"):
                    assert sample[key].shape==(1,1,N,128) and sample[key].dtype==torch.float16
                q,k,v=[sample[k].cuda() for k in ("q","k","v")]
                case=f"rank{rank}_real"
                record["cases"][case]=one_case(case,q,k,v,sol,args.out,store=True)
                # Preserve original-source output drift as a separate comparison.
                baseline=sample["sol_v2"].float()
                new=torch.load(args.out/(case+"_comparison.pt"),map_location="cpu",weights_only=True)["output"].float()
                record["cases"][case]["vs_historical_capture_rel_l2"]=float(
                    torch.linalg.vector_norm(new-baseline)/torch.linalg.vector_norm(baseline).clamp_min(1.e-12))
                del sample,q,k,v,baseline,new
                torch.cuda.empty_cache()
                save()
            record["capture_source"]="two_verified_real_H3_rank_heads"
        else:
            record["capture_source"]="missing_previous_Kaggle_kernel_mount"
        q,k,v=input_synthetic((1,4,N,128),20261040)
        record["cases"]["synthetic_4head_full_length"]=one_case(
            "synthetic_4head_full_length",q,k,v,sol,args.out,store=False)
        del q,k,v
        record["status"]="complete"
        save()
        print("SOL_COND_AB_VARIANT_COMPLETE",args.variant,args.repeat_id,flush=True)
    except BaseException as exc:
        record.update(status="failed",error=repr(exc))
        save()
        raise

if __name__=="__main__":
    main()
