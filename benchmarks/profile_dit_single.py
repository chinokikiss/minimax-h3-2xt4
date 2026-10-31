"""Prepare real inputs separately, then profile exactly one Native DiT NFE."""
import argparse
from contextlib import contextmanager
from dataclasses import asdict
import gc
import gzip
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'ComfyUI'),str(ROOT/'benchmarks')]
import torch
import ref2va_e2e_run as observation
from run_h3_l1_current_e2e import CurrentRank,CurrentService,summaries
from patches import comfy_tp_runtime as rt
from patches.h3_operator_policy import H3OperatorPolicy
from patches.h3_tensor_flow import TensorFlowPolicy
from dit_performance_observer import Phases


def runtime(config,te=None):
    return rt.RuntimeConfig(te or config['paths']['te'],config['paths']['dit'],layers=50,
        weight_bytes=6*2**30,pinned_bytes=6*2**30,cache_bytes=3*2**30,host_bytes=16*2**30,
        reserve_bytes=9*2**30,timeout_seconds=1500,prefetch_lifecycle=True,memory_pressure_aware=True,
        cross_request_te=False,cross_model_cpu=config.get('cross_model_cpu',True),idle_residency=False,
        lora_stack=tuple(tuple(x) for x in config.get('lora_stack',())),bundle_offline_lora=config.get('bundle_offline_lora',False),
        operators=H3OperatorPolicy(**config.get('operators',{})),tensor_flow=TensorFlowPolicy('fp16'))


def validate_attention_backend(mode, counts):
    """Require 50 actual backend executions; reject silent fallback/mixed paths."""
    kitchen=counts.get('kitchen/int8_attention',0)
    sol=counts.get('DiT/sol_all_int8',0)
    if mode=='kitchen':assert kitchen==50 and sol==0,counts
    elif mode=='sol':assert sol==50 and kitchen==0,counts
    else:raise ValueError('Single-call comparison supports Kitchen or Sol')


class ProfileWindow(observation.ObservedWindow):
    @contextmanager
    def use(self,key):
        owner=observation.CURRENT
        phases=getattr(owner,'phases',None)
        if phases is None or not phases.enabled:
            with super().use(key) as module:yield module
            return
        previous=phases.block;phases.block=str(key[1:])
        probe=getattr(owner,'cpu_probe',None)
        if probe:probe.mark_progress('weight_acquire',key[1:])
        try:
            with phases.span('weight_acquire_and_block'):
                with super().use(key) as module:
                    if probe:probe.mark_progress('block_compute',key[1:])
                    with phases.span('block_compute'):yield module
            if probe:probe.mark_progress('block_enqueued',key[1:])
            if key[1]=='dit' and (key[2]+1)%5==0:
                owner.checkpoint_trace(f'dit_block_{key[2]}')
        finally:phases.block=previous


class ProfileRank(CurrentRank):
    def __init__(self,*args):
        super().__init__(*args)
        self.phases=Phases(self.rank);self.phases.install()
        self.warmup=None;self.backend_start={}
        self.cuda_checkpoints=None
        if os.environ.get('H3_TRACE_CONTROL')=='1':
            from worker_cuda_checkpoint import CUDAFlushCheckpoint
            self.cuda_checkpoints=CUDAFlushCheckpoint(self.rank,os.environ['H3_WORKER_CPU_OUTPUT'],device=self.device)
        self.cpu_probe=None
        if os.environ.get('H3_WORKER_CPU_OUTPUT'):
            from worker_cpu_probe import WorkerCPUProbe
            self.cpu_probe=WorkerCPUProbe(self.rank,os.environ['H3_WORKER_CPU_OUTPUT'])
            self.phases.cpu_probe=self.cpu_probe

    def checkpoint_trace(self,label,*,resume=True):
        if self.cuda_checkpoints is not None:
            self.cuda_checkpoints.flush(label,resume=resume)
            if self.cpu_probe and self.cpu_probe.enabled:self.cpu_probe.checkpoint()

    def dispatch(self,command,args=(),kwargs=None):
        if command=='profile_operator_warmup':
            # One representative trained block compiles all production kernel
            # geometries; it is not another 50-layer DiT call or trajectory.
            self._phase('dit');self.context['command']='operator_warmup'
            key=self.keys[0];self.window.extend_uses([key]);start=time.monotonic()
            from comfy.ldm.minimax.model import PackedLayout
            layout=PackedLayout(1182,107,32,56,603,refs=[dict(kind='image',latent_t=1,latent_h=32,latent_w=56,ref_audio_t=0)]*2)
            with self.window.use(key) as block:
                x=torch.zeros((51220,5376),device=self.device,dtype=torch.float16)
                emb=torch.zeros((1,8),device=self.device,dtype=torch.float32)
                rope=torch.eye(2,device=self.device).expand(1,51220,1,48,2,2).contiguous()
                output=block(x,emb,[(0,51220,0)],rope,{'minimax_h3_layout':layout})
                torch.cuda.synchronize(self.device);del output,x,emb,rope
            self._drop_model()
            self.warmup=dict(seconds=time.monotonic()-start,scope='One trained block, exact operator shapes; zero full DiT warmup calls')
            return self.warmup
        if command=='denoise':
            self.backend_start=dict(self.backends);self.phases.enabled=True
            if self.cpu_probe:self.cpu_probe.start()
            try:return super().dispatch(command,args,kwargs)
            finally:
                failed=sys.exc_info()[0] is not None
                self.phases.enabled=False
                try:
                    if self.cpu_probe:self.cpu_probe.finish()
                finally:
                    try:self.checkpoint_trace('denoise_exit')
                    except BaseException:
                        if not failed:raise
        if command=='profile_stop_trace':
            self.checkpoint_trace('worker_close',resume=False)
            value=dict(rank=self.rank,pid=os.getpid(),flushed=True)
            if os.environ.get('H3_WORKER_CPU_OUTPUT'):
                (Path(os.environ['H3_WORKER_CPU_OUTPUT'])/f'rank{self.rank}_flush.json').write_text(json.dumps(value))
            return value
        if command=='profile_observer_overhead':
            value=self.observer_overhead()
            (Path(os.environ['H3_WORKER_CPU_OUTPUT'])/f'rank{self.rank}_observer_overhead.json').write_text(json.dumps(value,indent=2))
            return value
        result=super().dispatch(command,args,kwargs)
        if command=='stats':
            result['performance_phases']=self.phases.report()
            result['operator_warmup']=self.warmup
            result['cuda_checkpoints']=self.cuda_checkpoints.report() if self.cuda_checkpoints else None
            result['denoise_backend_counts']={k:v-self.backend_start.get(k,0) for k,v in self.backends.items()}
        return result

    def observer_overhead(self):
        """Operator-only replay after sampling; never a second DiT call."""
        if not self.cpu_probe or self.cpu_probe.segments is None:return dict(blocker='No actual modulation segments observed')
        records=self.cpu_probe.segments
        if any(not isinstance(x['row'],int) for x in records):return dict(blocker='Tensor-index modulation requires a separate exact replay')
        segments=[(x['start'],x['end'],x['row']) for x in records]
        self._drop_model()
        # Nothing after this operator-only replay needs weights. Drain CPU
        # preparation so deterministic wall-clock profiles have no disk work.
        self.cache.close();self.startup_cache.close();torch.cuda.empty_cache()
        from patches import h3_tensor_flow as flow
        originals={(module,name):fn for module,name,fn in self.phases.originals}
        norm_fn=originals[(flow,'modulated_norm')];residual_fn=originals[(flow,'gated_add')]
        x=torch.full((51220,5376),.125,device=self.device,dtype=torch.float16)
        branch=torch.full_like(x,.03125)
        count=max(row for _,_,row in segments)+1
        shift=torch.zeros((count,5376),device=self.device);scale=torch.zeros_like(shift);gate=torch.full_like(shift,.125)
        norm=torch.nn.RMSNorm(5376,eps=1e-6/4096,device=self.device,dtype=torch.float32)
        rows=[];reference=None
        @contextmanager
        def scope(mode,label):
            if mode in ('full','sampled_full','profiled_full'):
                self.phases.enabled=True
                try:
                    with self.phases.span('observer_replay/'+label):yield
                finally:self.phases.enabled=False
            elif mode=='nvtx':
                torch.cuda.nvtx.range_push('observer_replay/'+label)
                try:yield
                finally:torch.cuda.nvtx.range_pop()
            elif mode=='events':
                a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True);a.record()
                try:yield
                finally:b.record()
            else:yield
        # Warm once, then counterbalance the modes; all allocate the same
        # tensors and synchronize outside their host-enqueue measurement.
        warm=norm_fn(norm,x,shift,scale,segments);del warm;torch.cuda.synchronize()
        for repetition in range(3):
            modes=('off','nvtx','events','full','sampled_full','profiled_full')
            for mode in (modes if repetition%2==0 else tuple(reversed(modes))):
                probe=None
                if mode in ('sampled_full','profiled_full'):
                    from worker_cpu_probe import WorkerCPUProbe
                    probe=WorkerCPUProbe(self.rank,Path(os.environ['H3_WORKER_CPU_OUTPUT'])/f'{mode}_{repetition}',
                        deterministic=mode=='profiled_full')
                    if mode=='sampled_full':probe.counts.update({'norm/modulate':1000,'residual/gated_add':1000})
                    probe.start()
                x.fill_(.125);torch.cuda.synchronize()
                a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True);a.record()
                start,cpu=time.monotonic_ns(),time.thread_time_ns()
                with scope(mode,'norm'):
                    h=probe.invoke('norm/modulate',norm_fn,(norm,x,shift,scale,segments),{}) if probe else norm_fn(norm,x,shift,scale,segments)
                with scope(mode,'residual'):
                    if probe:probe.invoke('residual/gated_add',residual_fn,(x,gate,branch,segments),{})
                    else:residual_fn(x,gate,branch,segments)
                host=(time.monotonic_ns()-start)/1e9;thread=(time.thread_time_ns()-cpu)/1e9
                b.record();b.synchronize()
                if probe:probe.finish()
                if reference is None:reference=(h.clone(),x.clone())
                exact=bool(torch.equal(h,reference[0]) and torch.equal(x,reference[1]))
                if not exact:raise AssertionError('Observer replay changed the operator output')
                rows.append(dict(mode=mode,repetition=repetition,host_seconds=host,thread_cpu_seconds=thread,
                    cuda_stream_ms=a.elapsed_time(b),byte_exact=exact))
                del h
        return dict(rows=rows,segment_count=len(segments),scope='Synthetic resident tensors at exact shape and observed segmentation; operator-only overhead, no extra DiT call',
            sampling_threads_disabled_in_modes=['off','nvtx','events','full'])


def worker(*args):
    rt.RankRuntime=ProfileRank;rt.ResidencyWindow=ProfileWindow
    observation.BASE_WORKER(*args)


class ProfileService(CurrentService):
    """Flush in each live CUDA worker, then allow normal process teardown."""
    def close(self):
        if self.closed:return
        try:
            self.call('profile_stop_trace')
            self.call('close')
            for process in self.processes:process.join(timeout=15)
        finally:self.abort()


def prepare(config,path,out):
    from patches.comfy_sampler import create_clip
    from patches.staged_vae import StagedVAE
    from comfy_extras.nodes_minimax_h3 import MiniMaxH3ReferenceToVideo
    from comfy.nested_tensor import NestedTensor
    from PIL import Image
    import numpy as np
    # Conditioning needs no DiT merge or speculative DiT preparation.
    prep_config=config|dict(lora_stack=(),bundle_offline_lora=False,cross_model_cpu=False)
    service=CurrentService(runtime(prep_config));vae=None
    try:
        clip=create_clip(service);vae=StagedVAE(config['paths']['video'],service)
        images={f'ref_image_{i+1}':torch.from_numpy(np.asarray(Image.open(p).convert('RGB'),dtype=np.float32)/255)[None]
            for i,p in enumerate(config['images'])}
        started=time.monotonic()
        positive,latent=tuple(MiniMaxH3ReferenceToVideo.execute(clip,config['prompt'],896,512,362,
            ref_image_size='match',vae=vae,ref_images=images))[:2]
        latent=latent['samples'];generator=torch.Generator().manual_seed(config['seed'])
        noise=NestedTensor([torch.randn(t.shape,generator=generator) for t in latent.unbind()])
        value=rt.packet(dict(positive=positive,latent=latent,noise=noise),'cpu')
        torch.save(value,path)
        receipt=dict(status='real_inputs_ready',seconds=time.monotonic()-started,
            input_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            inputs={k:observation.digest(t) for k,t in [('conditioning',positive[0][0])]+
                [(f'noise_{i}',t) for i,t in enumerate(noise.unbind())]},
            privacy='Temporary inputs stay in job scratch; no activation payload in returned outputs')
        (out/'preparation.json').write_text(json.dumps(receipt,indent=2));return receipt
    finally:
        if vae:vae.release()
        service.close()


def profile(config,path,out):
    from safetensors.torch import save_file
    from patches.comfy_sampler import create_model
    from comfy_extras.nodes_minimax_h3 import MiniMaxH3SigmaShift
    import comfy.samplers
    # This private scratch was produced in the same bounded job and hash-bound.
    receipt=json.loads((out/'preparation.json').read_text())
    assert hashlib.sha256(path.read_bytes()).hexdigest()==receipt['input_sha256']
    value=torch.load(path,map_location='cpu',weights_only=False)
    empty=path.with_name('unused_te.safetensors');save_file({},str(empty))
    cfg=runtime(config,str(empty));rt._worker=worker
    state=dict(status='starting',scope='Exactly one denoiser call; no full sampler trajectory or VAE decode',
        runtime=asdict(cfg),inputs=receipt['inputs'],single_call=True,operator_policy=asdict(cfg.operators),
        instrumented=True,steady_state_claim=False)
    def save():(out/'result.json').write_text(json.dumps(state,indent=2,default=str))
    save();service=None;monitor=observation.Monitor()
    try:
        service=ProfileService(cfg)
        state['operator_warmup']=service.call('profile_operator_warmup');save()
        model=MiniMaxH3SigmaShift.execute(create_model(service),12.,config['audio_shift'])[0]
        guider=comfy.samplers.CFGGuider(model);guider.set_conds(value['positive'],value['positive']);guider.set_cfg(1.)
        sigmas=comfy.samplers.calculate_sigmas(model.model.model_sampling,'simple',4)[:2]
        state['sigma_interval']=sigmas.tolist();save()
        start=time.monotonic_ns()
        result=guider.sample(value['noise'],value['latent'],comfy.samplers.sampler_object('lcm'),sigmas,
            disable_pbar=False,seed=config['seed'])
        state['sample_wrapper_seconds']=(time.monotonic_ns()-start)/1e9
        state['output']=summaries(result);del result
        state['ranks']=service.call('stats');state['service_calls']=service.spans
        for rank in state['ranks']:
            assert rank['calls']==1 and len([c for c in rank['benchmark']['commands'] if c['command']=='denoise'])==1
            validate_attention_backend(cfg.operators.dit_attention,rank['denoise_backend_counts'])
            assert all(l['sequence_tokens']==51220 for l in rank['benchmark']['layouts'])
        assert all(x['finite'] for x in state['output'])
        if os.environ.get('H3_WORKER_CPU_OUTPUT'):
            service.call('profile_observer_overhead')
            state['observer_overhead']=[json.loads((Path(os.environ['H3_WORKER_CPU_OUTPUT'])/
                f'rank{rank}_observer_overhead.json').read_text()) for rank in (0,1)]
            assert all('blocker' not in r and all(x['byte_exact'] for x in r['rows']) for r in state['observer_overhead'])
        state.update(status='complete',quality_acceptance='Not a generation-quality experiment; one call only')
    except BaseException as error:
        state.update(status='failed',error=repr(error));(out/'exception.txt').write_text(traceback.format_exc());raise
    finally:
        if service:service.close()
        if service:state['worker_exit']=[dict(pid=p.pid,exit_code=p.exitcode,alive=p.is_alive()) for p in service.processes]
        monitor.close()
        with gzip.open(out/'nvml_samples.json.gz','wt') as stream:json.dump(monitor.rows,stream)
        save()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--inputs',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--prepare-only',action='store_true');args=parser.parse_args();sys.argv=sys.argv[:1]
    args.output.mkdir(parents=True,exist_ok=True)
    config=json.loads(args.config.read_text());torch.set_grad_enabled(False);torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    assert torch.cuda.device_count()==2 and all(torch.cuda.get_device_capability(i)==(7,5) for i in (0,1))
    if args.prepare_only:prepare(config,args.inputs,args.output)
    else:profile(config,args.inputs,args.output)
    gc.collect()


if __name__=='__main__':main()
