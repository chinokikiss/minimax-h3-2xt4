"""Deterministic physical-SM75 A/B probe for the Sol routing workspace pool.

Run as separate processes with H3_SOL_WORKSPACE_POOL=0/1.  The probe reports
both steady per-call temporary peak and the pool's persistent allocated bytes;
this prevents retained workspace from making peak deltas look artificially low.
"""
from __future__ import annotations
import argparse,hashlib,json,os,statistics
from pathlib import Path

import torch
from patches import sol_attention
from patches import sol_sm75_backend as native


def inputs(shape,seed):
    b,h,length,dim=shape
    out=[]
    for i in range(3):
        g=torch.Generator(device="cpu").manual_seed(seed+i)
        x=torch.randn((b,length,h,dim),dtype=torch.float16,generator=g)
        out.append(x.cuda().transpose(1,2))
    return out


def gpu_ms(fn,repeats=5,warmup=2):
    for _ in range(warmup):
        y=fn();del y
    torch.cuda.synchronize()
    values=[]
    for _ in range(repeats):
        a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True)
        a.record();y=fn();b.record();b.synchronize()
        values.append(float(a.elapsed_time(b)));del y
    return dict(median_ms=statistics.median(values),samples_ms=values)


def memory_profile(fn,pool_enabled):
    if pool_enabled:
        native.clear_workspace_pool()
    torch.cuda.empty_cache();torch.cuda.synchronize()
    initial=int(torch.cuda.memory_allocated())
    torch.cuda.reset_peak_memory_stats()
    y=fn();torch.cuda.synchronize();del y;torch.cuda.synchronize()
    cold_peak=int(torch.cuda.max_memory_allocated()-initial)
    after_first=int(torch.cuda.memory_allocated())
    resident=max(0,after_first-initial)

    # Measure a reuse call from the retained steady-state baseline, but also
    # report total bytes above the original pre-pool baseline.
    torch.cuda.reset_peak_memory_stats()
    steady_base=int(torch.cuda.memory_allocated())
    y=fn();torch.cuda.synchronize();del y;torch.cuda.synchronize()
    steady_peak_abs=int(torch.cuda.max_memory_allocated())
    return dict(
        initial_allocated=initial,
        cold_peak_over_initial=cold_peak,
        resident_after_first=resident,
        steady_base_allocated=steady_base,
        steady_peak_increment=max(0,steady_peak_abs-steady_base),
        steady_peak_over_initial=max(0,steady_peak_abs-initial),
        pool_snapshot=native.workspace_pool_snapshot() if pool_enabled else [],
    )


def valid_lut_sha256(lut, counts):
    packed=lut.detach().cpu().reshape(-1,lut.shape[-1])
    sizes=counts.reshape(-1)
    digest=hashlib.sha256()
    for row in range(sizes.numel()):
        digest.update(packed[row,:int(sizes[row])].numpy().tobytes())
    return digest.hexdigest()


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--out",type=Path,required=True)
    args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=False)
    torch.set_grad_enabled(False);torch.cuda.set_device(0)
    assert torch.cuda.is_available() and torch.cuda.get_device_capability(0)==(7,5)
    pool=os.environ.get("H3_SOL_WORKSPACE_POOL","0")=="1"
    sol_attention.prepare_native()
    payload=dict(variant="pool" if pool else "baseline",abi=12,
                 gpu=torch.cuda.get_device_name(0),cases={})
    for name,shape,seed in [
        ("d64_vae",(1,8,1029,64),20261101),
        ("real_shape_d128",(1,4,65166,128),20261110),
    ]:
        q,k,v=inputs(shape,seed)
        if name=="d64_vae":
            topo=dict(kind="video_vae",start_token=0,end_token=1024,
                      frame_stride_tokens=256,temporal_radius=1,
                      local_radius_tokens=64,protect_frame_boundaries=False)
            kwargs=dict(protected_tokens=5,protected_start=1024,allow_d64=True,
                        tau=1.,topology=topo,sync_free_lut=True,
                        collect_route_stats=False)
        else:
            topo=dict(kind="dit",start_token=4974,end_token=65166,
                      frame_stride_tokens=836,temporal_radius=1,
                      local_radius_tokens=64,protect_frame_boundaries=True)
            kwargs=dict(protected_tokens=4974,protected_start=0,allow_d64=False,
                        tau=1.,topology=topo,sync_free_lut=True,
                        collect_route_stats=False)
        fn=lambda:sol_attention.attention(q,k,v,**kwargs)

        # Materialize one deterministic output/route witness.  prepare is kept
        # alive until execute finishes; releasing it afterwards returns a pool
        # lease only on the same CUDA stream.
        prepared=sol_attention.prepare_attention(q,k,v,**kwargs)
        exact=prepared["exact_mask"].detach().cpu().clone()
        count=prepared["row_count"].detach().cpu().clone()
        lut_sha256=valid_lut_sha256(prepared["lut"],count)
        output=sol_attention.execute_prepared(prepared).detach().cpu().clone()
        torch.cuda.synchronize();del prepared
        timing=gpu_ms(fn)
        memory=memory_profile(fn,pool)
        torch.save(dict(output=output,exact_mask=exact,row_count=count),
                   args.out/(name+".pt"))
        payload["cases"][name]=dict(
            shape=list(shape),complete=timing,memory=memory,lut_sha256=lut_sha256,
            route_density=float(count.sum().item()/(count.numel()*exact.shape[-1])))
        del q,k,v,output,exact,count
        if pool:native.clear_workspace_pool()
        torch.cuda.empty_cache()
    (args.out/"result.json").write_text(json.dumps(payload,indent=2))
    print(json.dumps(payload,indent=2),flush=True)


if __name__=="__main__":
    main()
