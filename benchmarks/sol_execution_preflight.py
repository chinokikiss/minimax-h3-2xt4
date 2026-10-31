"""Physical-T4 execution-semantic preflight for the production Sol hybrid path.

This is deliberately not a generation-quality or performance benchmark.  It
executes two complete trained DiT calls in one persistent TP2 session (Kitchen
then Sol) and one small real Video VAE decode through the production temporal
handoff.  It fails closed on dispatch, layout, AOT, stride, and finite-output
contracts.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'ComfyUI'),str(ROOT/'benchmarks')]

import torch
import ref2va_e2e_run as observation
from benchmarks.dit_pair_contract import fixture,identity,digest,TOKENS,SEGMENTS
from benchmarks.e2e_attention_validation import denoiser_attention_counts
from patches import comfy_tp_runtime as rt


def tensor_contract(t):
    return dict(shape=list(t.shape),stride=list(t.stride()),dtype=str(t.dtype),
        contiguous=bool(t.is_contiguous()),device=str(t.device))


class PreflightRank(observation.ObservedRank):
    def _install_observers(self):
        super()._install_observers()
        self.preflight_qkv=[]

        import patches.h3_operator_policy as policy
        original_prepare=policy.prepare_dit_attention
        def prepare(block,q,k,v,options=None):
            result=original_prepare(block,q,k,v,options)
            if result is not None:
                self._record_qkv('kitchen',q,k,v,options or {})
            return result
        policy.prepare_dit_attention=prepare

        import patches.sol_attention as sol
        original_sol=sol.attention
        def sol_attention(q,k,v,*args,**kwargs):
            self._record_qkv('sol',q,k,v,{},
                protected_tokens=kwargs.get('protected_tokens'),
                protected_start=kwargs.get('protected_start'),
                allow_d64=kwargs.get('allow_d64',False),
                tau=kwargs.get('tau',1.0))
            return original_sol(q,k,v,*args,**kwargs)
        sol.attention=sol_attention

    def _record_qkv(self,backend,q,k,v,options,**extra):
        command=self.context.get('command')
        nfe=self.context.get('nfe',0)
        key=(backend,command,nfe,tuple(q.shape),tuple(q.stride()))
        if any(row['_key']==key for row in self.preflight_qkv):
            return
        prefix=None
        if command=='denoise' and backend=='sol':
            import patches.h3_operator_policy as policy
            prefix=policy.protected_prefix(options,q.shape[-2]) if options.get('minimax_h3_layout') is not None else extra.get('protected_tokens')
        self.preflight_qkv.append(dict(_key=key,backend=backend,command=command,nfe=nfe,
            q=tensor_contract(q),k=tensor_contract(k),v=tensor_contract(v),
            protected_prefix=prefix,**extra))

    def dispatch(self,command,args=(),kwargs=None):
        result=super().dispatch(command,args,kwargs)
        if command=='stats':
            result['preflight_qkv']=[{k:v for k,v in row.items() if k!='_key'} for row in self.preflight_qkv]
        return result


def worker(*args):
    rt.RankRuntime=PreflightRank
    rt.ResidencyWindow=observation.ObservedWindow
    observation.BASE_WORKER(*args)


class Service(observation.Service):
    pass


def assert_transpose_view(row):
    for name in ('q','k','v'):
        t=row[name]
        assert t['shape']==[1,28,TOKENS,128],(name,t)
        assert t['dtype']=='torch.float16',(name,t)
        assert t['stride'][-1]==1,(name,t)
        assert not t['contiguous'],('Expected production transpose-view QKV',name,t)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--video',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();sys.argv=sys.argv[:1]
    args.output.mkdir(parents=True,exist_ok=False)
    state=dict(status='starting',scope=(
        'Physical T4 execution-semantic preflight only: two complete trained DiT calls '
        '(NFE1 Kitchen, NFE2 Sol) plus one small real Video VAE Kitchen temporal decode (no Sol); '
        'no generation-quality or speed claim.'))
    def save():(args.output/'result.json').write_text(json.dumps(state,indent=2,default=str))
    save()
    service=None
    staged=None
    try:
        assert torch.cuda.device_count()==2
        devices=[dict(name=torch.cuda.get_device_name(i),capability=list(torch.cuda.get_device_capability(i)),
            uuid=str(torch.cuda.get_device_properties(i).uuid)) for i in range(2)]
        assert len({x['uuid'] for x in devices})==2
        assert all(x['capability']==[7,5] and 'T4' in x['name'] for x in devices)
        torch.set_grad_enabled(False);torch.set_num_threads(2)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False

        from safetensors.torch import save_file
        from patches.h3_operator_policy import H3OperatorPolicy
        from patches.h3_tensor_flow import TensorFlowPolicy
        from patches.attention_schedule import parse_attention_schedule

        empty=args.output/'unused_te.safetensors';save_file({},str(empty))
        config=rt.RuntimeConfig(str(empty),str(args.checkpoint),
            lora_stack=(),bundle_offline_lora=False,
            cross_request_te=False,cross_model_cpu=False,idle_residency=False,
            operators=H3OperatorPolicy(te_attention='kitchen_causal',dit_attention='kitchen'),
            dit_attention_schedule=parse_attention_schedule('2:sol@1.0'),
            sol_tau=1.0,video_attention='kitchen',tensor_flow=TensorFlowPolicy('fp16'))
        state.update(hardware=devices,runtime=asdict(config));save()

        rt._worker=worker
        start=time.monotonic();service=Service(config)
        state['service_start_seconds']=time.monotonic()-start

        # Explicitly include the production AOT seam before the first denoiser.
        state['dit_aot_parent_reply']=service.call('prepare_dit',TOKENS)
        inputs=fixture();state['input_identity']=identity(inputs)
        state['expected_protected_prefix']=SEGMENTS[-2][0]
        service.call('begin_sample')
        inputs['context']=service.call('refine',inputs['context'])
        before=[digest(t) for t in inputs['x']]+[digest(inputs['context'])]
        state['denoise']=[]
        for nfe in (1,2):
            begin=time.monotonic()
            output=service.call('denoise',**inputs)
            assert [digest(t) for t in inputs['x']]+[digest(inputs['context'])]==before
            summaries=[dict(shape=list(t.shape),finite=bool(torch.isfinite(t).all()),sha256=digest(t)) for t in output]
            assert all(x['finite'] for x in summaries)
            state['denoise'].append(dict(nfe=nfe,seconds=time.monotonic()-begin,outputs=summaries));save()
            del output
        service.call('end_sample')

        # Exercise the public production handoff, not a direct VAE operator call.
        from patches.staged_vae import StagedVAE
        from patches.comfy_sampler import ParallelDecodeHandoff
        staged=StagedVAE(str(args.video),service)
        decoder=ParallelDecodeHandoff(staged,service,str(args.video),'temporal')
        gen=torch.Generator().manual_seed(20261030)
        # Match the real completed E2E boundary: sampled Video latents arrive
        # at the public VAE as FP32; Native decode normalizes/casts each temporal
        # clip before the worker's strict FP16 raw-decoder boundary.
        latent=torch.randn((1,24,12,16,16),generator=gen,dtype=torch.float32)
        begin=time.monotonic();pixels=decoder.decode(latent).cpu()
        state['video_decode']=dict(seconds=time.monotonic()-begin,
            strategy=decoder.last_strategy,fallback=decoder.last_fallback,
            input=tensor_contract(latent),output=tensor_contract(pixels),
            finite=bool(torch.isfinite(pixels).all()))
        assert decoder.last_strategy=='native_temporal_pairs'
        assert decoder.last_fallback is None and state['video_decode']['finite']

        ranks=service.call('stats');state['ranks']=ranks
        expected_prefix=SEGMENTS[-2][0]
        for rank in ranks:
            bench=rank['benchmark']
            kitchen=denoiser_attention_counts(bench,'kitchen/int8_attention')
            sol=denoiser_attention_counts(bench,'DiT/sol_all_int8')
            assert kitchen['total']==50 and kitchen['per_step'][1]==50 and kitchen['per_step'][2]==0,kitchen
            assert sol['total']==50 and sol['per_step'][1]==0 and sol['per_step'][2]==50,sol
            assert kitchen['outside_command']==sol['outside_command']==0

            rows=rank['preflight_qkv']
            krow=next(x for x in rows if x['backend']=='kitchen' and x['command']=='denoise' and x['nfe']==1)
            srow=next(x for x in rows if x['backend']=='sol' and x['command']=='denoise' and x['nfe']==2)
            assert_transpose_view(krow);assert_transpose_view(srow)
            assert srow['protected_prefix']==expected_prefix,(srow,expected_prefix)
            assert rank['operators']['video_attention']=='kitchen'
            aot=rank.get('aot_preparation',{})
            assert aot.get('dit',{}).get('sol_precompiled'),aot
            rank['preflight_acceptance']=dict(kitchen=kitchen,sol=sol,
                raw_dit_kitchen=krow,raw_dit_sol=srow,video_backend='kitchen')

        state.update(status='complete',acceptance=dict(
            mixed_nfe_dispatch=True,dit_transpose_view=True,condition_prefix=True,
            dit_sol_aot=True,video_temporal_handoff=True,video_kitchen=True,
            finite_outputs=True))
    except BaseException as error:
        state.update(status='failed',error=type(error).__name__+': '+str(error))
        (args.output/'exception.txt').write_text(traceback.format_exc())
        if service and not service.closed:
            try:state['failure_ranks']=service.call('stats')
            except Exception as capture:state['failure_capture_error']=repr(capture)
        raise
    finally:
        if staged is not None:
            try:staged.release()
            except Exception:pass
        if service:service.close()
        save()


if __name__=='__main__':main()
