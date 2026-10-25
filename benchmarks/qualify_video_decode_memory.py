"""Trained Video-only local qualification: two logical ranks, never a T4 claim."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import sys
import time
import weakref
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'ComfyUI'),str(ROOT/'benchmarks')]
import torch
from logical_dual_gpu import isolate_cublas_workspaces
from qualify_parallel_vae_pipeline import LogicalVAEService
from patches.comfy_sampler import ParallelDecodeHandoff
from patches.staged_vae import StagedVAE
from patches.comfy_tp_runtime import RuntimeConfig


SOURCE_FILES=('benchmarks/qualify_video_decode_memory.py','patches/video_decode_planner.py',
    'patches/comfy_sampler.py','patches/comfy_tp_runtime.py','patches/staged_vae.py',
    'patches/parallel_vae.py','patches/h3_tensor_flow.py')


def source_hashes():
    return {name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in SOURCE_FILES}


class VideoService(LogicalVAEService):
    def __init__(self):
        super().__init__()
        self.config=RuntimeConfig('unused','unused',reserve_bytes=9*2**30,cross_request_te=False)
        for rank in self.ranks:
            rank.config=self.config
            rank.readers={'dit':SimpleNamespace()}
            rank.idle_window=None
            rank.vae_numeric_flags=None
        self.staged_vaes=weakref.WeakSet()
        self.parent=self.trained=None
        self.parent_checks=[]

    def register_staged_vae(self,vae):self.staged_vaes.add(vae)

    def call(self,command,*args,**kwargs):
        if command in ('vae_prepare','video_pixels'):
            planner=self.parent.first_stage_model
            row=dict(command=command,planner_parameter_bytes=sum(p.numel()*p.element_size() for p in planner.parameters()),
                trained_parent_gpu_bytes=sum(p.numel()*p.element_size() for p in self.trained.parameters() if p.is_cuda))
            assert row['planner_parameter_bytes']==row['trained_parent_gpu_bytes']==0,row
            self.parent_checks.append(row)
        return super().call(command,*args,**kwargs)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--video',type=Path,default=Path('D:/minimaxH3/minimax_h3_video_vae_int8_convrot.safetensors'))
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--decode-inputs',type=Path,help='Replay saved sampler video latents; audio is not decoded')
    parser.add_argument('--long-only',action='store_true',help='Run only the saved-latent comparison')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    import comfy.ldm.minimax.vae
    import comfy.model_management as management
    import comfy_kitchen as ck
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=False
    # Benchmark-only stream-private scratch, as in the prior logical-rank gate.
    ck.backends.cuda._max_dynamic_shared_memory_per_block=lambda unused:49152
    isolate_cublas_workspaces()
    service=VideoService();vae=StagedVAE(args.video,service)
    service.parent=vae.vae
    result=dict(status='running',physical=False,hardware=torch.cuda.get_device_name(),torch=torch.__version__,
        scope='Trained Video VAE public decode; two logical ranks on one local GPU; no TE/DiT or T4 performance qualification',
        cases=[],source_sha256=source_hashes())
    def save():
        (args.output/'result.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    try:
        wrapper=ParallelDecodeHandoff(vae,service,args.video)
        cases=[] if args.long_only else [('single_frame_261',(1,24,1,16,16),False),
                ('two_item_batch',(2,24,1,16,16),False),
                ('temporal_1797',(1,24,7,16,16),False),
                ('native_repeated_tiles',(1,24,31,32,32),False),
                ('public_decode_tiled',(1,24,1,16,16),True)]
        saved=None
        if args.decode_inputs:
            saved=torch.load(args.decode_inputs,map_location='cpu',weights_only=True)['video']
            result['decode_inputs_sha256']=hashlib.sha256(args.decode_inputs.read_bytes()).hexdigest()
            cases.append(('saved_long_sequence',tuple(saved.shape),False))
        if args.long_only and saved is None:raise ValueError('--long-only requires --decode-inputs')
        for name,shape,tiled in cases:
            z=saved if name=='saved_long_sequence' else torch.randn(shape,generator=torch.Generator().manual_seed(20261023)).half()
            method=vae.decode_tiled if tiled else vae.decode
            reference=method(z).cpu();torch.cuda.synchronize()
            service.trained=vae.first_stage_model;before=len(service.parent_checks)
            start=time.monotonic()
            actual=(wrapper.decode_tiled(samples=z) if tiled else wrapper.decode(samples_in=z)).cpu()
            torch.cuda.synchronize()
            max_abs=0.;finite=True
            for actual_batch,reference_batch in zip(actual,reference):
                for a,b in zip(actual_batch.split(8),reference_batch.split(8)):
                    max_abs=max(max_abs,float((a-b).abs().max()));finite &= bool(torch.isfinite(a).all())
            row=dict(name=name,shape=list(shape),output_shape=list(actual.shape),
                bit_exact=torch.equal(actual,reference),finite=finite,
                strategy=wrapper.last_strategy,fallback=wrapper.last_fallback,
                parent_checks=len(service.parent_checks)-before,elapsed_seconds=time.monotonic()-start,
                max_abs=max_abs)
            result['cases'].append(row);save();print(json.dumps(row),flush=True)
            assert row['bit_exact'] and row['finite'] and row['strategy']=='native_aligned_query'
            assert all(rank.vae_model is None for rank in service.ranks)
            assert vae.first_stage_model is service.trained
            del actual,reference,z
        # Actual parent/worker ownership through denial, error and recovery.
        z=torch.randn((1,24,1,16,16),generator=torch.Generator().manual_seed(98)).half()
        reference=vae.decode(z).cpu();service.trained=vae.first_stage_model
        from patches.parallel_vae import checkpoint_plan
        _,_,bound=checkpoint_plan(args.video,'video')
        old_limit=service.budget.limit_bytes;service.budget.limit_bytes=bound
        try:actual=wrapper.decode(z).cpu()
        finally:service.budget.limit_bytes=old_limit
        assert wrapper.last_strategy=='native' and torch.equal(actual,reference)
        result['cases'].append(dict(name='host_refusal_native_fallback',bit_exact=True,reason=wrapper.last_fallback));save()
        service.fail_decode=True
        try:
            wrapper.decode(z)
            raise AssertionError('Injected decode error did not propagate')
        except RuntimeError as error:assert 'injected decode failure' in str(error)
        finally:service.fail_decode=False
        assert vae.first_stage_model is service.trained and all(r.vae_model is None for r in service.ranks)
        assert torch.equal(wrapper.decode(z).cpu(),reference)
        result['cases'].append(dict(name='decode_failure_restoration_and_recovery',bit_exact=True))
        result['parent_checks']=service.parent_checks
        result['admissions']=[[e for e in r.events if e.get('phase')=='vae_prepare'] for r in service.ranks]
        assert result['source_sha256']==source_hashes(),'Source changed during qualification'
        result['status']='passed'
    except BaseException as error:
        result.update(status='failed',error=repr(error));raise
    finally:
        try:
            vae.release();service.close();management.unload_all_models();gc.collect();torch.cuda.empty_cache()
        finally:
            result['host_budget']=service.budget.snapshot()
            save()
        assert result['host_budget']['reserved_bytes']==0


if __name__=='__main__':main()
