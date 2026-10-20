"""Trained legacy Video VAE, two logical CUDA ranks; no TP2 speed claim."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),os.environ.get('COMFY_ROOT',str(ROOT/'ComfyUI')),str(ROOT/'benchmarks')]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();sys.argv=sys.argv[:1]
    import torch
    import comfy_kitchen as ck
    from patches.legacy_runtime import validate_imported_source,validate_legacy_attention_api
    from patches.parallel_vae import checkpoint_plan,load_decode_weights,AlignedVideoDecode,query_partitioning_status
    from logical_dual_gpu import LogicalDualGPU,isolate_cublas_workspaces
    source=validate_imported_source();api=validate_legacy_attention_api()
    torch.set_grad_enabled(False);torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
    isolate_cublas_workspaces()
    checkpoint=Path('D:/minimaxH3/minimax_h3_video_vae_int8_convrot.safetensors')
    reader,names,_=checkpoint_plan(checkpoint,'video')
    models=[load_decode_weights(reader,names,'video','cuda:0') for _ in range(2)]
    statuses=[query_partitioning_status(m) for m in models]
    assert all(s['available'] and s['block_count']==36 for s in statuses)
    assert all(torch.equal(b.attn.qk_norm_scale,torch.ones_like(b.attn.qk_norm_scale))
        for m in models for b in m.decoder.transformer_blocks)
    z=torch.randn((1,24,1,16,16),generator=torch.Generator().manual_seed(871)).half().cuda()
    comm=LogicalDualGPU(timeout=120)
    calls={'native':0,'parallel':0};stage='native';kitchen=ck.int8_attention
    def count(*a,**kw):calls[stage]+=1;return kitchen(*a,**kw)
    with torch.inference_mode(),patch.object(ck,'int8_attention',new=count):
        expected=models[0]._decode_pixels(z)
        stage='parallel'
        runners=[AlignedVideoDecode(m,z.clone(),comm,r) for r,m in enumerate(models)]
        outputs=comm.run([runner.run for runner in runners])
        assert outputs[1] is None and torch.equal(outputs[0],expected)
        assert torch.isfinite(expected).all()
    assert calls==dict(native=36,parallel=72),calls
    assert all(not hasattr(b.attn,'_query_run') for m in models for b in m.decoder.transformer_blocks)
    assert not comm.steps and all(not r.buffers for r in runners)
    report=dict(scope='Trained 36-block Video VAE only. Two logical ranks on one GPU; simulated transport, no NCCL/T4 performance qualification.',
        comfy_root=source,attention_api=api,torch=torch.__version__,hardware=torch.cuda.get_device_name(),
        capability=torch.cuda.get_device_capability(),checkpoint=str(checkpoint),compatibility=statuses,
        query_tokens=261,partitions=runners[0].ranges,kitchen_calls=calls,bit_exact=True,
        shape=list(expected.shape),temporary_hooks_restored=True,collective_buffers_released=True,
        output_sha256=hashlib.sha256(expected.cpu().numpy().tobytes()).hexdigest())
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('LEGACY_VIDEO_QUERY_PASS',calls,flush=True)


if __name__=='__main__':main()
