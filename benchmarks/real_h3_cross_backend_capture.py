"""Capture a real post-RMS/post-RoPE MiniMax-H3 QKV head for cross-GPU Sol semantics.

T4 runs frozen pre-math-fix Sol v2. The exact serialized FP16 BHND Q/K/V
are replayable by ComfyUI's sm80+ comfy_kitchen.sol_attn on another GPU.
"""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import torch
import real_h3_attention_microgate as original
from patches import h3_operator_policy as policy
from patches import sol_attention as sol

def hash_tensor(x):
    h=hashlib.sha256()
    h.update(x.detach().contiguous().cpu().numpy().tobytes())
    return h.hexdigest()

def capture(owner,q,k,v,options):
    outdir=Path(os.environ["H3_REAL_ATTN_CAPTURE_DIR"])
    outdir.mkdir(parents=True,exist_ok=True)
    rank=int(owner.rank)
    layout=options.get("minimax_h3_layout")
    if layout is None or int(layout.seq_len)!=int(q.shape[-2]):
        raise RuntimeError("Real H3 PackedLayout missing or mismatch")
    # Deliberate 1 rank-local head: real (not generated) source and full 65k sequence.
    # Copy to BHND contiguous without changing any value.
    qs,ks,vs=[x[:,0:1].contiguous() for x in (q,k,v)]
    if qs.shape!=(1,1,65166,128) or ks.shape!=qs.shape or vs.shape!=qs.shape:
        raise RuntimeError("Expected 65166-token post-RoPE, D128 real H3 QKV")
    if any(x.dtype!=torch.float16 or not torch.isfinite(x).all().item() for x in (qs,ks,vs)):
        raise RuntimeError("QKV must be finite FP16")
    prefix,topo=policy.sol_v2_routing(options,qs.shape[-2])
    # Explicitly use the restored old Sol source (no mathematical fix).
    output=sol.attention(qs,ks,vs,protected_tokens=prefix,protected_start=0,
                         tau=1.,topology=topo,
                         sync_free_lut=True,collect_route_stats=False)
    torch.cuda.synchronize()
    cpu=dict(q=qs.cpu(),k=ks.cpu(),v=vs.cpu(),sol_v2=output.cpu())
    saved=outdir/f"rank{rank}_real_qkv.pt"
    torch.save(cpu,saved)
    digest=hashlib.sha256(saved.read_bytes()).hexdigest()
    row=dict(status="complete",rank=rank,nfe=owner.context.get("nfe"),
        gpu=torch.cuda.get_device_name(q.device),capability=list(torch.cuda.get_device_capability(q.device)),
        source="real MiniMax-H3 post-RMS/post-RoPE first denoiser, first DiT attention",
        qkv_dtype="float16",heads=1,head_index=0,sequence_tokens=qs.shape[-2],
        dims=list(qs.shape),tensor_sha256={name:hash_tensor(value) for name,value in cpu.items()},
        payload_file=saved.name,payload_bytes=saved.stat().st_size,payload_sha256=digest,
        layout=dict(segments=[list(x) for x in layout.segments],
            signature=list(layout.signature),sequence_tokens=int(layout.seq_len)),
        sol=dict(tau=1.,protected_tokens=prefix,topology=topo,
            sync_free_lut=True,collect_route_stats=False,
            workspace_pool=os.environ.get("H3_SOL_WORKSPACE_POOL","0"),
            bitmap_lut=os.environ.get("H3_SOL_LUT_BITMAP","0")),
        comparison_scope="cross-device same serialized QKV; numerical semantics only, not speed or model-quality equivalence")
    (outdir/f"rank{rank}.json").write_text(json.dumps(row,indent=2))
    del cpu,output,qs,ks,vs
    print("REAL_H3_SOL_CROSS_CAPTURE_COMPLETE",rank, saved.stat().st_size,flush=True)

original.microgate=capture

if __name__=="__main__":
    original.main()
