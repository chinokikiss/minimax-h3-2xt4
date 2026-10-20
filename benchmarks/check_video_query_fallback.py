"""Check actual Video VAE fallback, commands and backend on the current source.

261 query tokens can be partitioned; this is a VAE-only local check.
"""
import argparse
import hashlib
import inspect
import json
from pathlib import Path
import sys
import os
import threading
from types import SimpleNamespace
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),os.environ.get('COMFY_ROOT',str(ROOT/'ComfyUI'))]
import torch


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,
        default=Path('D:/attention_eval/video_query_fallback_20261019/check.json'))
    args=parser.parse_args();sys.argv=sys.argv[:1]
    import comfy.sd,comfy.utils,comfy.ops,comfy_kitchen as ck
    import comfy.ldm.minimax.vae as native
    from patches.staged_vae import StagedVAE
    from patches.comfy_sampler import ParallelDecodeHandoff
    from patches.host_weight_cache import HostBudget
    from patches.parallel_vae import aligned_ranges
    torch.set_grad_enabled(False);torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
    commands=[]
    def call(command,*a,**kw):
        commands.append(command)
        if command=='vae_prepare':return dict(accepted=False,reason='Injected worker memory admission denial for fallback qualification')
        if command not in ('decode_start','decode_complete'):
            raise AssertionError('Fallback entered parallel dispatch: '+command)
    service=SimpleNamespace(lock=threading.RLock(),closed=False,call=call,
        register_staged_vae=lambda v:None,budget=HostBudget(16*2**30,reserve_bytes=0))
    checkpoint='D:/minimaxH3/minimax_h3_video_vae_int8_convrot.safetensors'
    reference=comfy.sd.VAE(sd=comfy.utils.load_torch_file(checkpoint),device=torch.device('cuda:0'),dtype=torch.float16)
    staged=StagedVAE(checkpoint,service);wrapper=ParallelDecodeHandoff(staged,service,checkpoint,'auto')
    z=torch.randn((1,24,1,16,16),generator=torch.Generator().manual_seed(871)).half()
    ranges=aligned_ranges(16*16+5);assert ranges is not None
    counters=dict(sdpa=0,kitchen=0);sdpa=comfy.ops.scaled_dot_product_attention;kitchen=ck.int8_attention
    def count_sdpa(*a,**kw):counters['sdpa']+=1;return sdpa(*a,**kw)
    def count_kitchen(*a,**kw):counters['kitchen']+=1;return kitchen(*a,**kw)
    rows=[]
    for method in ('decode','decode_tiled'):
        expected=getattr(reference,method)(z.clone())
        begin=len(commands);old_counters=dict(counters)
        with patch.object(comfy.ops,'scaled_dot_product_attention',new=count_sdpa),patch.object(ck,'int8_attention',new=count_kitchen):
            actual=getattr(wrapper,method)(z.clone())
        assert wrapper.last_strategy=='native'
        expected_commands=['vae_prepare','decode_start','decode_complete'] if wrapper.last_query_partitioning['available'] else ['decode_start','decode_complete']
        assert commands[begin:]==expected_commands
        assert torch.equal(actual,expected),(method,float((actual-expected).abs().max()))
        assert '_decode_pixels' not in staged.first_stage_model.__dict__
        rows.append(dict(method=method,bit_exact=True,shape=list(actual.shape),commands=commands[begin:],
            attention_calls={k:counters[k]-old_counters[k] for k in counters}))
        print('VIDEO_QUERY_FALLBACK_PASS',method,rows[-1]['attention_calls'],flush=True)
    status=wrapper.last_query_partitioning
    assert not status['unquantized_blocks'] and status['block_count']==36
    files=[Path(inspect.getfile(native)),ROOT/'patches/parallel_vae.py',ROOT/'patches/comfy_sampler.py']
    report=dict(scope='Current source, trained Video VAE-only decode on one local GPU. No full H3 or physical TP2 performance.',
        hardware=torch.cuda.get_device_name(),capability=torch.cuda.get_device_capability(),
        query_tokens=261,otherwise_valid_partition=ranges,checkpoint=checkpoint,
        compatibility=status,fallback_reason=wrapper.last_fallback,records=rows,
        source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2))
    staged.release();assert service.budget.snapshot()['reserved_bytes']==0


if __name__=='__main__':main()
