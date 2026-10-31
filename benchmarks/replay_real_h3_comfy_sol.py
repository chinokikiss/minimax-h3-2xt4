"""Replay physically captured real MiniMax-H3 QKV on ComfyUI built-in Sol.

Do not synthesize QKV. Files are emitted by real_h3_cross_backend_capture.py
on physical T4 with exact TP rank-local post-RMS/post-RoPE D128 tensors.
The T4's restored Sol-v2 output is the reference; CK executes on SM80+.

Numerical semantics only: cross-device GPU timings are NOT compared.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import torch
import torch.nn.functional as F
import comfy_kitchen as ck

BLOCK=64

def sha_bytes(raw):
    return hashlib.sha256(raw).hexdigest()

def tensor_sha(x):
    return sha_bytes(x.detach().contiguous().cpu().numpy().tobytes())

def metrics(a,b):
    a=a.float();b=b.float();diff=a-b
    na=torch.linalg.vector_norm(a);nb=torch.linalg.vector_norm(b)
    return dict(
        rel_l2=float(torch.linalg.vector_norm(diff)/nb.clamp_min(1e-12)),
        mae=float(diff.abs().mean()),
        max_abs=float(diff.abs().max()),
        rms=float(diff.square().mean().sqrt()),
        cosine=float(F.cosine_similarity(a.flatten(),b.flatten(),dim=0)),
        norm_ratio=float(na/nb.clamp_min(1e-12)),
        finite=bool(torch.isfinite(a).all()),
    )

def intervals(t,ref,layout):
    rows={}
    for a,b,kind in layout["segments"]:
        name=kind+"_"+str(a)
        rows[name]=metrics(t[:,:,a:b],ref[:,:,a:b])
    audio=next((a,b) for a,b,k in layout["segments"] if k=="audio")
    video=next((a,b) for a,b,k in layout["segments"] if k=="video")
    rows["all"]=metrics(t,ref)
    rows["prefix"]=metrics(t[:,:,:video[0]],ref[:,:,:video[0]])
    rows["generated_audio"]=metrics(t[:,:,audio[0]:audio[1]],ref[:,:,audio[0]:audio[1]])
    rows["video"]=metrics(t[:,:,video[0]:video[1]],ref[:,:,video[0]:video[1]])
    frames=int(layout["signature"][1])
    frame_tokens=(video[1]-video[0])//frames
    assert frames==72 and frame_tokens==836 and (video[1]-video[0])%frames==0
    per_frame=[metrics(t[:,:,video[0]+i*frame_tokens:video[0]+(i+1)*frame_tokens],
                       ref[:,:,video[0]+i*frame_tokens:video[0]+(i+1)*frame_tokens])["rel_l2"]
               for i in range(frames)]
    rows["video_frames"]=dict(count=frames,mean=float(statistics.mean(per_frame)),
        median=float(statistics.median(per_frame)),max=float(max(per_frame)),
        min=float(min(per_frame)),per_frame_rel_l2=per_frame)
    return rows

def sampled_fp32_dense(q,k,v,layout):
    """True dense attention on ~85 selected real queries, never the 65k×65k matrix."""
    audio_start=next(a for a,b,kind in layout['segments'] if kind=='audio')
    video_start,video_end=next((a,b) for a,b,kind in layout['segments'] if kind=='video')
    frame_tokens=(video_end-video_start)//72
    indices=sorted(set([0,1,1264,2527,2528,3343,3344,4163,
                       audio_start,4550,video_start-1,video_start,video_end-1]+
                       [video_start+i*frame_tokens+frame_tokens//2 for i in range(72)]))
    idx=torch.tensor(indices,device=q.device,dtype=torch.int64)
    qb=q.transpose(1,2)[:,:,idx].float()
    kb=k.transpose(1,2).float()
    vb=v.transpose(1,2).float()
    logits=torch.matmul(qb,kb.transpose(-2,-1))*(q.shape[-1]**-0.5)
    dense=torch.matmul(torch.softmax(logits,dim=-1),vb)
    del qb,kb,vb,logits
    return idx,dense

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--capture-dir",required=True,type=Path)
    parser.add_argument("--output",required=True,type=Path)
    parser.add_argument("--ranks",default="0,1")
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    if not torch.cuda.is_available():raise RuntimeError("Requires a CUDA GPU supporting ComfyUI built-in Sol")
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32=False  # Explicit FP32 sampled dense baseline.
    device=torch.device("cuda:0")
    if not ck.sol_attn_is_available(device):
        raise RuntimeError("ComfyUI comfy_kitchen.sol_attn not available on this GPU; requires SM80+")
    capability=list(torch.cuda.get_device_capability(device))
    assert capability>=[8,0],capability
    state=dict(status="running",scope="same serialized real QKV, cross-device Sol numerical semantics only",
        source="real MiniMax-H3 first attention post-RMS/post-RoPE, T4 Sol-v2, then local ComfyUI Sol",
        replay_env=dict(torch=torch.__version__,gpu=torch.cuda.get_device_name(device),
            capability=capability,kitchen=getattr(ck,"__version__",None)),ranks={})
    def save():
        (args.output/"result.json").write_text(json.dumps(state,indent=2))
    save()
    try:
        for rank in [int(n) for n in args.ranks.split(",")]:
            root=args.capture_dir
            meta=json.loads((root/f"rank{rank}.json").read_text())
            payload=root/meta["payload_file"]
            assert payload.stat().st_size==meta["payload_bytes"]
            assert sha_bytes(payload.read_bytes())==meta["payload_sha256"]
            sample=torch.load(payload,map_location="cpu",weights_only=True)
            for name,t in sample.items():
                assert tuple(t.shape)==(1,1,65166,128)
                assert t.dtype==torch.float16
                assert tensor_sha(t)==meta["tensor_sha256"][name],(rank,name)
            layout=meta["layout"]
            assert layout["sequence_tokens"]==65166
            assert meta["sol"]["tau"]==1.0
            q,k,v=[sample[name].to(device,non_blocking=False).transpose(1,2).contiguous()
                   for name in ("q","k","v")]
            ref=sample["sol_v2"].to(device)
            vid_start=next(a for a,b,kind in layout["segments"] if kind=="video")
            audio_start=next(a for a,b,kind in layout["segments"] if kind=="audio")
            sink=[0,math.ceil(vid_start/BLOCK)]
            comfy_sink_q=[audio_start//BLOCK, sink[1]]
            if meta["sol"]["protected_tokens"]!=vid_start:
                raise RuntimeError("T4 Sol prefix and video boundary differ")
            dense_indices,dense_out=sampled_fp32_dense(q,k,v,layout)
            dense_reference=dict(query_indices=dense_indices.cpu().tolist(),
                rows=len(dense_indices),numerical_reference="FP32 QK softmax PV; sampled real queries only",
                sol_v2_vs_dense=metrics(ref[:,:,dense_indices],dense_out))
            variants=[
                ("matched_dense_prefix_tau1_extra0",1.0,[0,sink[1]],0),
                ("comfy_h3_default_sinks_tau1_extra0",1.0,comfy_sink_q,0),
                ("comfy_h3_default_sinks_tau1_extra256",1.0,comfy_sink_q,256),
                ("comfy_ui_default_tau1p3_extra256",1.3,comfy_sink_q,256),
            ]
            cases={}
            for name,tau,sink_q,extra in variants:
                try:
                    y=ck.sol_attn(q,k,v,tau=tau,scale=None,
                        sink_blocks=sink,sink_q=sink_q,
                        topk_ratio=0.0,token_aug=extra,
                        tail=True)
                    torch.cuda.synchronize()
                    assert y.shape==q.shape and y.dtype==q.dtype
                    got=y.transpose(1,2)
                    cases[name]=dict(status="complete",tau=tau,
                        sink_blocks=sink,sink_q=sink_q,
                        token_aug=extra,topk_ratio=0.,tail=True,
                        metrics=intervals(got,ref,layout),
                        sampled_queries_vs_fp32_dense=metrics(got[:,:,dense_indices],dense_out),
                        output_sha256=tensor_sha(y))
                    print("CK_SOL_REPLAY_CASE",rank,name,
                          cases[name]["metrics"]["all"]["rel_l2"],flush=True)
                except Exception as exc:
                    cases[name]=dict(status="failed",error=type(exc).__name__+": "+str(exc))
                    save()
                    raise
            state["ranks"][str(rank)]=dict(status="complete",input_meta=meta,
                dense_sample_reference=dense_reference,
                cases=cases,source_tensor_sha256=meta["tensor_sha256"])
            save()
            del q,k,v,ref,sample
        state["status"]="complete"
        save()
    except BaseException as exc:
        state["status"]="failed"
        state["error"]=type(exc).__name__+": "+str(exc)
        save()
        raise

if __name__=="__main__":
    main()
