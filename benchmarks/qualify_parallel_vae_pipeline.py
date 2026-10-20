"""Trained public ComfyUI VAE outputs with actual parallel worker dispatch.

Two logical CUDA ranks on one device: correctness/lifetimes, not performance.
Does not rerun TE/DiT; their cache lifecycle is represented by empty bounded
caches so the test focuses on the new VAE dispatch and Native public planning.
"""
import gc
import hashlib
import json
from pathlib import Path
import sys
import threading

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT/'ComfyUI', ROOT/'.retest-deps', ROOT/'benchmarks'):
    sys.path.insert(0, str(path))

import torch
from logical_dual_gpu import LogicalDualGPU, isolate_cublas_workspaces
from patches.comfy_tp_runtime import RankRuntime, RuntimeConfig, packet
from patches.comfy_sampler import ParallelDecodeHandoff
from patches.host_weight_cache import HostBudget, HostWeightCache, GiB
from patches.parallel_vae import aligned_ranges


class LogicalVAEService:
    def __init__(self):
        self.lock, self.closed = threading.RLock(), False
        self.comm = LogicalDualGPU(timeout=180)
        self.budget = HostBudget(16*GiB, reserve_bytes=GiB)
        self.ranks = []
        for rank in range(2):
            runtime = RankRuntime.__new__(RankRuntime)
            runtime.config = RuntimeConfig('unused', 'unused', cross_request_te=False)
            runtime.rank, runtime.device, runtime.comm = rank, torch.device('cuda:0'), self.comm
            runtime.cache = HostWeightCache(self.budget, 256*2**20)
            runtime.startup_cache = HostWeightCache(self.budget, 256*2**20)
            runtime.model = runtime.window = runtime.preprocessing = runtime.vae_model = None
            runtime.kind = runtime.vae_kind = None
            runtime.aux_reservation = 0
            runtime.sample_active = runtime.completed_decode = runtime.decode_lookahead_started = False
            runtime.events, runtime.calls = [], 0
            self.ranks.append(runtime)
        self.commands = []
        self.fail_decode = False

    def call(self, command, *args, **kwargs):
        with self.lock:
            self.commands.append(command)
            if self.fail_decode and command == 'video_pixels':
                raise RuntimeError('injected decode failure before collectives')
            args, kwargs = packet(args, 'cpu'), packet(kwargs, 'cpu')
            def run(rank):
                out = self.ranks[rank].dispatch(command, args, kwargs)
                return packet(out, 'cpu')
            return self.comm.run([lambda r=r:run(r) for r in range(2)])[0]

    def close(self):
        self.call('close')
        self.closed = True


def metric(a, b):
    a, b = a.float().cpu(), b.float().cpu()
    delta = a-b
    return dict(bit_exact=torch.equal(a,b), finite=bool(torch.isfinite(a).all()),
                rel_l2=float(delta.norm()/b.norm().clamp_min(1e-20)), max_abs=float(delta.abs().max()))


def main():
    import argparse
    import comfy.sd
    import comfy.utils
    # Resolve Native attention availability before the logical-rank ConvRot
    # scratch-limit substitution, as in the accepted raw query qualification.
    import comfy.ldm.minimax.vae
    p = argparse.ArgumentParser()
    p.add_argument('--video',type=Path,default=Path('D:/minimaxH3/minimax_h3_video_vae_int8_convrot.safetensors'))
    p.add_argument('--audio',type=Path,default=Path('D:/minimaxH3/minimax_h3_audio_vae_fp32_clean.safetensors'))
    p.add_argument('--output',type=Path,default=ROOT/'semantic_results/parallel_vae_pipeline.json')
    args=p.parse_args()
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=False
    import comfy_kitchen as ck
    ck.backends.cuda._max_dynamic_shared_memory_per_block=lambda unused:49152
    isolate_cublas_workspaces()
    service=LogicalVAEService()
    result=dict(passed=False,physical=False,gpu=torch.cuda.get_device_name(),torch=torch.__version__,
        scope='Full trained Native public ComfyUI VAE decode; two logical CUDA ranks; no latency or T4 claim',
        audio_contract='Channel splitting matches sequential mono; small Native rounding differences are accepted explicitly, not bit-exact',
        cases=[])
    def save():
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(result,indent=2),encoding='utf-8')
    def record(name, check, **extra):
        row=dict(name=name,**check,**extra)
        result['cases'].append(row);save();print(json.dumps(row),flush=True)
    try:
        workflow=json.loads((ROOT/'workflows/video_minimax_h3_r2v_tp2_int8.json').read_text(encoding='utf-8'))
        by_id={node['id']:node for node in workflow['nodes']}
        loaders=[node for node in by_id.values() if node['type']=='MiniMaxH3ParallelVAELoader']
        assert len(loaders)==2 and all(node['widgets_values'][1]=='auto' for node in loaders)
        for identity,source,slot,dest,dest_slot,kind in workflow['links']:
            assert by_id[dest]['inputs'][dest_slot]['link']==identity
            assert identity in by_id[source]['outputs'][slot]['links']
        for node in loaders:
            edge=next(link for link in workflow['links'] if link[0]==node['inputs'][0]['link'])
            assert by_id[edge[1]]['type']=='MiniMaxH3TP2Loader' and edge[-1]=='MINIMAX_TP2_SESSION'
        record('default_workflow_dispatch',dict(passed=True),parallel_vae_loaders=2)
        video=comfy.sd.VAE(sd=comfy.utils.load_torch_file(str(args.video)),device=torch.device('cuda:0'),
                           dtype=torch.float16)
        wrapper=ParallelDecodeHandoff(video,service,args.video)
        # Single frame, full temporal clip, Native repeated chunks and spatial
        # tile batching (512px width); preserve exact Native output dimensions.
        for name,shape in [('short_query_native_control',(1,24,1,4,4)),
                           ('single_frame',(1,24,1,16,16)),
                           ('two_item_batch',(2,24,1,16,16)),
                           ('temporal_chunk',(1,24,7,16,16)),
                           ('public_512px_31_tokens',(1,24,31,32,32))]:
            z=torch.randn(shape,generator=torch.Generator().manual_seed(732)).half()
            reference=video.decode(z).cpu()
            actual=wrapper.decode(z).cpu()
            check=metric(actual,reference)
            record(name,check,output_shape=list(actual.shape),strategy=wrapper.last_strategy,
                   fallback=wrapper.last_fallback,video_commands=service.commands.count('video_pixels'))
            assert check['bit_exact'] and check['finite'] and wrapper.last_strategy=='native_aligned_query'
            assert '_decode_pixels' not in video.first_stage_model.__dict__
            assert all(r.vae_model is None for r in service.ranks)
            del reference,actual,z
        z=torch.randn((1,24,1,16,16),generator=torch.Generator().manual_seed(5)).half()
        ref=video.decode(z).cpu()
        native=ParallelDecodeHandoff(video,service,args.video,'native')
        count=len(service.commands);out=native.decode(z).cpu()
        assert 'vae_prepare' not in service.commands[count:]
        record('explicit_native',metric(out,ref),strategy=native.last_strategy)
        ref_tiled=video.decode_tiled(z).cpu();out_tiled=wrapper.decode_tiled(z).cpu()
        check=metric(out_tiled,ref_tiled)
        assert check['bit_exact'] and wrapper.last_strategy=='native_aligned_query'
        record('public_decode_tiled',check,strategy=wrapper.last_strategy)
        keyword=wrapper.decode(samples_in=z).cpu();check=metric(keyword,ref)
        assert check['bit_exact']
        record('decode_keyword_argument',check)
        keyword=wrapper.decode_tiled(samples=z).cpu();check=metric(keyword,ref_tiled)
        assert check['bit_exact']
        record('decode_tiled_keyword_argument',check)
        del keyword
        del ref_tiled,out_tiled
        # Coordinated denial must fall back on BOTH ranks, not enter a gather.
        from patches.parallel_vae import checkpoint_plan
        _, _, bound = checkpoint_plan(args.video,'video')
        limit=service.budget.limit_bytes;service.budget.limit_bytes=bound
        count=len(service.commands);out=wrapper.decode(z).cpu();service.budget.limit_bytes=limit
        assert wrapper.last_strategy=='native' and 'video_pixels' not in service.commands[count:]
        record('memory_denial_native_fallback',metric(out,ref),reason=wrapper.last_fallback)
        service.fail_decode=True
        try:
            wrapper.decode(z)
            raise AssertionError('Injected failure did not propagate')
        except RuntimeError as exc:
            assert 'injected decode failure' in str(exc)
        finally:
            service.fail_decode=False
        assert '_decode_pixels' not in video.first_stage_model.__dict__
        assert all(r.vae_model is None for r in service.ranks)
        record('failure_restores_hook_and_releases_replicas',dict(passed=True))
        recovered=wrapper.decode(z).cpu();check=metric(recovered,ref)
        assert check['bit_exact']
        record('decode_after_failure',check)
        del recovered
        del wrapper,native,video,out,ref,z
        import comfy.model_management
        comfy.model_management.unload_all_models();gc.collect();torch.cuda.empty_cache()
        audio=comfy.sd.VAE(sd=comfy.utils.load_torch_file(str(args.audio)),device=torch.device('cuda:0'),
                           dtype=torch.float32)
        wrapper=ParallelDecodeHandoff(audio,service,args.audio)
        for seconds in (5,10,20):
            z=torch.load(ROOT/f'semantic_results/audio_saved_references/input_{seconds}s.pt',weights_only=True)
            ref=audio.decode(z).cpu()
            actual=wrapper.decode(z).cpu()
            with torch.cuda.device(0):
                mono=torch.cat([audio.first_stage_model.decode(z[:,:,i:i+1].cuda()).cpu() for i in (0,1)],1).movedim(1,-1)
            check=metric(actual,ref);control=metric(actual,mono)
            assert control['bit_exact'] and check['finite'] and check['rel_l2']<3e-5
            assert wrapper.last_strategy=='channel_parallel'
            assert 'decode' not in audio.first_stage_model.__dict__
            assert all(r.vae_model is None for r in service.ranks)
            record(f'audio_{seconds}s',check,vs_sequential_mono=control,strategy=wrapper.last_strategy,
                   original_strict_gate_pass=check['rel_l2']<=1e-5)
            del actual,ref,mono,z
        native=ParallelDecodeHandoff(audio,service,args.audio,'native')
        z=torch.zeros((1,32,2,200));ref=audio.decode(z).cpu();out=native.decode(z).cpu()
        record('audio_strict_native_override',metric(out,ref),strategy=native.last_strategy)
        assert aligned_ranges(5) is None
        result['source_sha256']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [Path(__file__),ROOT/'patches/parallel_vae.py',ROOT/'patches/comfy_sampler.py',
                      ROOT/'patches/comfy_tp_runtime.py',ROOT/'benchmarks/logical_dual_gpu.py',
                      ROOT/'custom_nodes/minimax_h3_tp2/__init__.py',ROOT/'build_comfy_tp2_workflow.py',
                      ROOT/'workflows/video_minimax_h3_r2v_tp2_int8.json']}
        result['passed']=True
    finally:
        service.close()
        result['host_budget']=service.budget.snapshot()
        assert result['host_budget']['reserved_bytes']==0
        save()


if __name__=='__main__':
    main()
