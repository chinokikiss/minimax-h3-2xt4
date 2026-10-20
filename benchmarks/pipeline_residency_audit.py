"""Full trained Native sampler pipeline and residency/transfer audit.

The benchmark instruments the production runtime; it does not change model
operators. Synchronous ablation retains the same CPU/GPU cache capacities.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass,replace
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
for folder in (ROOT, ROOT/'ComfyUI', ROOT/'.retest-deps', ROOT/'benchmarks'):
    sys.path.insert(0,str(folder))

import psutil
import torch
from torch import nn
import retest
from patches import comfy_tp_runtime as runtime
from patches.comfy_sampler import create_clip, create_model, DecodeHandoff
from patches.host_weight_cache import GiB
from patches.weight_residency import ResidencyWindow
from patches.windowed_tp import CheckpointReader
from patches.optimized_tp import enable
from full_models import StreamedBlock

NATIVE_WORKER = runtime._worker
NATIVE_RANK = runtime.RankRuntime
NATIVE_READ = CheckpointReader.read
READ_LOCK = threading.Lock()


def audit_read(reader,prefix,exclude=()):
    data=NATIVE_READ(reader,prefix,exclude)
    if CURRENT_AUDIT is not None:
        size=sum(t.numel()*t.element_size() for t in data.values())
        with READ_LOCK:
            CURRENT_AUDIT.read_bytes+=size
            CURRENT_AUDIT.read_calls+=1
    return data


@dataclass(frozen=True)
class AuditConfig(runtime.RuntimeConfig):
    strategy: str = 'adaptive'
    audit_dir: str = ''
    logical: bool = False


class AuditWindow(ResidencyWindow):
    """Ablation and instrumentation only; production leases remain unchanged."""
    def __init__(self,*args,**kwargs):
        self.audit = CURRENT_AUDIT
        if self.audit.config.logical:
            kwargs.update(max_lookahead=2)
        elif args[0] and args[0][0].key[1]=='dit':
            # S~10k FP32 TP scratch is larger than the default weight guard.
            # Reserve activation space even while the previous step is idle;
            # reusable allocator storage is not all available for weights.
            kwargs.update(reserve_bytes=int(4.5*GiB))
        if self.audit.config.strategy == 'sync':
            kwargs.update(adaptive=False,lookahead=1,max_lookahead=1,cpu_lookahead=1,
                          copy_stream=torch.cuda.current_stream(self.audit.device))
        super().__init__(*args,**kwargs)
        self.stats['audit_h2d_waits'] = []
        self.wait_events = []

    def _prepare(self,spec):
        with torch.profiler.record_function('audit/prepare/'+str(spec.key[1:])):
            return super()._prepare(spec)

    def _stage(self,key,host,size):
        with torch.profiler.record_function('audit/H2D/'+str(key[1:])):
            return super()._stage(key,host,size)

    @contextmanager
    def use(self,key):
        if self.audit.config.strategy == 'sync':
            # Fully expose preparation/copy instead of letting CPU work run
            # while the previous layer's queued CUDA kernels are executing.
            torch.cuda.current_stream(self.device).synchronize()
        with torch.profiler.record_function('audit/lease/'+str(key[1:])):
            before = torch.cuda.Event(enable_timing=True)
            after = torch.cuda.Event(enable_timing=True)
            before.record(torch.cuda.current_stream(self.device))
            with super().use(key) as module:
                after.record(torch.cuda.current_stream(self.device))
                self.wait_events.append((str(key),before,after))
                yield module

    def close(self):
        result = super().close()
        if self.wait_events:
            result['audit_h2d_waits'] += [dict(key=k,cuda_ms=a.elapsed_time(b))
                for k,a,b in self.wait_events]
            self.wait_events.clear()
        return result


CURRENT_AUDIT = None


class AuditRank(NATIVE_RANK):
    def __init__(self,*args,**kwargs):
        global CURRENT_AUDIT
        if args[0].logical:
            config=replace(args[0],weight_bytes=int((3.5 if args[1]==0 else 1.5)*GiB))
            args=(config,*args[1:])
        super().__init__(*args,**kwargs)
        CURRENT_AUDIT = self
        self.audit_calls=[]
        self.profiler = None
        self.read_bytes=self.read_calls=0

    def dispatch(self,command,args=(),kwargs=None):
        if command == 'profile_start':
            self.profiler=torch.profiler.profile(activities=[
                torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],
                record_shapes=False,profile_memory=False,with_stack=False)
            self.profiler.__enter__()
            return None
        if command == 'profile_stop':
            self.profiler.__exit__(None,None,None)
            path=Path(self.config.audit_dir)/f'{self.config.strategy}_rank{self.rank}_trace.json'
            self.profiler.export_chrome_trace(str(path))
            self.profiler=None
            return str(path)
        if command == 'audit_reset':
            for d in [self.device]:
                torch.cuda.reset_peak_memory_stats(d)
            return None
        if command == 'stats':
            base=super().dispatch('stats')
            base.update(audit_calls=self.audit_calls,
                checkpoint_read_bytes=sum(r.logical_read_bytes for r in self.readers.values()),
                checkpoint_read_calls=sum(r.range_read_calls for r in self.readers.values()),
                pinned_allocator=torch.cuda.memory.host_memory_stats(),
                gpu=dict(allocated_peak=torch.cuda.max_memory_allocated(self.device),
                         reserved_peak=torch.cuda.max_memory_reserved(self.device),
                         driver_free=torch.cuda.mem_get_info(self.device)[0]))
            return base
        before=time.perf_counter()
        out=super().dispatch(command,args,kwargs)
        self.audit_calls.append(dict(command=command,dispatch_enqueue_ms=(time.perf_counter()-before)*1000,
            calls=self.calls,rank=self.rank,
            allocated=torch.cuda.memory_allocated(self.device),
            reserved=torch.cuda.memory_reserved(self.device),
            driver_free=torch.cuda.mem_get_info(self.device)[0]))
        return out


def audit_worker(*args):
    runtime.RankRuntime=AuditRank
    runtime.ResidencyWindow=AuditWindow
    CheckpointReader.read=audit_read
    NATIVE_WORKER(*args)


class HostMonitor:
    def __init__(self):
        self.stop=threading.Event()
        self.rows=[]
        self.thread=threading.Thread(target=self.run,daemon=True)
        self.thread.start()

    def run(self):
        parent=psutil.Process()
        try:
            import pynvml
            pynvml.nvmlInit()
            handles=[pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(pynvml.nvmlDeviceGetCount())]
        except (ImportError,RuntimeError):
            handles=[]
        while not self.stop.is_set():
            processes=[parent]+parent.children(recursive=True)
            rss=0
            for process in processes:
                try:rss+=process.memory_info().rss
                except (psutil.NoSuchProcess,psutil.AccessDenied):pass
            vm=psutil.virtual_memory()
            self.rows.append(dict(time=time.time(),process_tree_rss=rss,
                                 system_used=vm.total-vm.available,available=vm.available,
                                 gpu_used=[pynvml.nvmlDeviceGetMemoryInfo(h).used for h in handles]))
            self.stop.wait(.1)

    def close(self):
        self.stop.set();self.thread.join()
        return dict(peak_process_tree_rss=max((r['process_tree_rss'] for r in self.rows),default=0),
                    peak_system_used=max((r['system_used'] for r in self.rows),default=0),
                    peak_driver_gpu_used=[max(r['gpu_used'][i] for r in self.rows)
                        for i in range(len(self.rows[0]['gpu_used']))] if self.rows else [],
                    min_available=min((r['available'] for r in self.rows),default=0),samples=self.rows)


def cold_page_cache(paths):
    """Best effort only: do not claim a physical storage-cold measurement."""
    if not hasattr(os,'posix_fadvise'):
        return 'unsupported; runtime caches still cold'
    for path in paths.values():
        with open(path,'rb') as stream:
            os.posix_fadvise(stream.fileno(),0,0,os.POSIX_FADV_DONTNEED)
    return 'POSIX_FADV_DONTNEED requested; physical storage coldness unverified'


def strict_metrics(actual,expected):
    result=retest.metrics(actual,expected)
    result['passed']=bool(result.get('valid') and result['rel_l2']<=1e-5
        and result['cosine']>=.999999 and result['max_abs']<=.005)
    return result


def load_vae(path,kind):
    import comfy.sd
    with torch.inference_mode(False),torch.no_grad():
        return comfy.sd.VAE(sd=CheckpointReader(path).read(''),device=torch.device('cuda:0'),
                            dtype=torch.float16 if kind=='video' else torch.float32)


def run_sample(model,positive,latent,steps,profile_service=None):
    import comfy.samplers
    from comfy.nested_tensor import NestedTensor
    gen=torch.Generator().manual_seed(1729)
    noise=NestedTensor([torch.randn(t.shape,generator=gen) for t in latent.unbind()])
    guider=comfy.samplers.CFGGuider(model)
    guider.set_conds(positive,positive);guider.set_cfg(1.)
    sigmas=comfy.samplers.calculate_sigmas(model.model.model_sampling,'simple',steps)
    with torch.inference_mode():
        return guider.sample(noise,latent,comfy.samplers.sampler_object('res_multistep'),
            sigmas,disable_pbar=True,seed=1729).cpu()


def conditioning(clip,vae,width,height):
    # Real pixels -> Native vision/DeepStack and Native reference VAE encoder.
    from comfy_extras.nodes_minimax_h3 import MiniMaxH3ReferenceToVideo
    grid=torch.linspace(0,1,width*height).reshape(1,height,width,1)
    image=torch.cat([grid,1-grid,grid*.5+.1],dim=-1)
    output=MiniMaxH3ReferenceToVideo.execute(clip,
        'A bird walks slowly through a quiet garden. Natural movement, soft daylight.',
        width,height,124,vae=vae,ref_images={'ref_image_1':image})
    return output[0],output[1]['samples']


def sync():
    # Physical command acknowledgements already fence both rank GPUs.
    # The ComfyUI parent owns its own VAE/input/output work on GPU zero.
    torch.cuda.synchronize(0)


def request_pipeline(service,clip,model,vaes,args,reference=None,save_reference=False):
    import comfy.model_management
    timings={};request_start=time.perf_counter()
    def timed(name,fn):
        sync();start=time.perf_counter()
        with torch.profiler.record_function('pipeline/'+name):
            out=fn()
        sync();timings[name]=(time.perf_counter()-start)*1000
        print('AUDIT STAGE',args.strategy,name,round(timings[name],1),flush=True)
        return out
    if not vaes:
        vaes.update(timed('vae_construction',lambda:{k:load_vae(args.paths[k],k) for k in ('video','audio')}))
    positive,latent=timed('reference_encode_and_TE',lambda:conditioning(clip,vaes['video'],args.width,args.height))
    sampled=timed('sampler',lambda:run_sample(model,positive,latent,args.steps))
    if service is None:
        # Native reference is streamed; release canonical preprocessing before VAE.
        denoiser=model.model.diffusion_model
        if hasattr(denoiser,'release'):denoiser.release()
        else:denoiser.cpu()
        comfy.model_management.unload_all_models()
    video,audio=sampled.unbind()
    decoder=DecodeHandoff(vaes['video'],service) if service else vaes['video']
    pixels=timed('video_decode',lambda:decoder.decode(video).cpu())
    waveform=timed('audio_decode',lambda:vaes['audio'].decode(audio).cpu())
    request_ms=(time.perf_counter()-request_start)*1000
    outputs=dict(conditioning=positive[0][0].cpu(),video=video.cpu(),audio=audio.cpu(),pixels=pixels,waveform=waveform)
    checks={}
    if reference:
        for name,actual in outputs.items():
            expected=torch.load(reference/f'{name}.pt',map_location='cpu',weights_only=True)
            checks[name]=strict_metrics(actual,expected)
            del expected
        if not all(c['passed'] for c in checks.values()):
            raise AssertionError('Native full-pipeline correctness failed: '+json.dumps(checks))
    if save_reference:
        for name,value in outputs.items():torch.save(value,args.output/'native_reference'/f'{name}.pt')
    return dict(stage_ms=timings,end_to_end_ms=request_ms,correctness=checks,
        video_shape=list(pixels.shape),audio_shape=list(waveform.shape),
        video_latent_shape=list(video.shape),audio_latent_shape=list(audio.shape),
        conditioning_shape=list(outputs['conditioning'].shape)),positive,latent


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--strategy',choices=('native','sync','adaptive'),required=True)
    parser.add_argument('--layers',type=int,default=50)
    parser.add_argument('--width',type=int,default=512)
    parser.add_argument('--height',type=int,default=512)
    parser.add_argument('--steps',type=int,default=4)
    parser.add_argument('--physical',action='store_true')
    parser.add_argument('--checkpoints',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--profile',action='store_true')
    parser.add_argument('--cache-gib',type=float,default=3.)
    parser.add_argument('--qualification',action='store_true',help='Short geometry/depth: correctness only')
    parser.add_argument('--native-resident',action='store_true',help='Qualify Native placement on logical devices')
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'native_reference').mkdir(exist_ok=True)
    args.paths=json.loads(args.checkpoints.read_text()) if args.checkpoints else {
        'te':'D:/minimaxH3/qwen3vl_32b_minimax_h3_int8_convrot.safetensors',
        'dit':'D:/minimaxH3/MiniMax_H3_Ref2VA_pruned_int8_convrot.safetensors',
        'video':'D:/minimaxH3/minimax_h3_video_vae_int8_convrot.safetensors',
        'audio':'D:/minimaxH3/minimax_h3_audio_vae_fp32_clean.safetensors'}
    if args.qualification:
        # The very same code path, shortened output duration only for local
        # sampler plumbing. Main physical requests always keep 124 frames.
        from comfy_extras import nodes_minimax_h3 as nodes_h3
        original=nodes_h3.MiniMaxH3ReferenceToVideo.execute.__func__
        nodes_h3.MiniMaxH3ReferenceToVideo.execute=classmethod(
            lambda cls,clip,prompt,width,height,length,**kw:original(cls,clip,prompt,width,height,5,**kw))
    torch.set_grad_enabled(False);torch.backends.cuda.matmul.allow_tf32=False
    enable('scalemax')
    if not args.physical:
        import comfy_kitchen as ck
        from logical_dual_gpu import isolate_cublas_workspaces
        ck.backends.cuda._max_dynamic_shared_memory_per_block=lambda d:49152
        isolate_cublas_workspaces()
    result=dict(strategy=args.strategy,passed=False,physical=args.physical,layers=args.layers,
        width=args.width,height=args.height,steps=args.steps,requests=[],
        cache_coldness=cold_page_cache(args.paths),
        source_sha256={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [*list((ROOT/'patches').glob('*.py')),Path(__file__)]})
    def save():retest.write_json(args.output/f'{args.strategy}_result.json',result)
    save();monitor=HostMonitor();service=None;vaes={};start=time.perf_counter()
    try:
        if args.strategy == 'native':
            from comfy.text_encoders.minimax import MiniMaxH3Tokenizer
            class NativeClip:
                def __init__(self):
                    self.tokenizer=MiniMaxH3Tokenizer();self.model=None
                def tokenize(self,*a,**kw):return self.tokenizer.tokenize_with_weights(*a,**kw)
                def encode_from_tokens_scheduled(self,tokens):
                    self.model=runtime.native_te(CheckpointReader(args.paths['te']),args.layers).cuda()
                    self.model.qwen3vl_32b.transformer.model.layers=nn.ModuleList([
                        StreamedBlock('te',args.paths['te'],i,'cuda:0') for i in range(args.layers)])
                    self.model.set_clip_options({'execution_device':torch.device('cuda:0')})
                    with torch.inference_mode():out=runtime.packet(self.model.encode_token_weights(tokens),'cpu')
                    del self.model;self.model=None;retest.cleanup()
                    return [[out[0],out[2]]]
            class Placeholder:
                closed=False
                lock=threading.RLock()
            clip=NativeClip();model=create_model(Placeholder())
            model.remove_wrappers_with_key('outer_sample','minimax_tp2_session')
            # Build after TE in a lazy proxy to avoid auxiliary VRAM contention.
            class LazyNative(nn.Module):
                dtype=torch.float32;patch_size=(1,2,2);hidden_size=5376
                def __init__(self):super().__init__();self.inner=None
                def load(self):
                    if self.inner is None:
                        self.inner=runtime.native_dit(CheckpointReader(args.paths['dit'])).cuda()
                        self.inner.blocks=nn.ModuleList([
                            StreamedBlock('dit',args.paths['dit'],i,'cuda:0') for i in range(args.layers)])
                    return self.inner
                def preprocess_text_embeds(self,*a,**kw):return self.load().preprocess_text_embeds(*a,**kw)
                def forward(self,*a,**kw):return self.load()(*a,**kw)
            if args.physical or args.native_resident:
                from native_resident_reference import ResidentNativeDiT
                devices=('cuda:0','cuda:1') if args.physical else ('cuda:0','cuda:0')
                model.model.diffusion_model=ResidentNativeDiT(CheckpointReader(args.paths['dit']),args.layers,devices)
            else:model.model.diffusion_model=LazyNative()
            record,_,_=request_pipeline(None,clip,model,vaes,args,save_reference=True)
            record['end_to_end_ms']=(time.perf_counter()-start)*1000
            result['requests'].append(record);result['passed']=True
        else:
            config=AuditConfig(args.paths['te'],args.paths['dit'],layers=args.layers,
                weight_bytes=int(9*GiB) if args.physical else 4*GiB,
                te_weight_bytes=int(11.7*GiB) if args.physical else 2*GiB,
                cache_bytes=int(args.cache_gib*GiB),host_bytes=(24 if args.physical else 16)*GiB,
                reserve_bytes=int(1.5*GiB),logical=not args.physical,
                strategy=args.strategy,audit_dir=str(args.output.resolve()),timeout_seconds=300)
            if args.physical:
                runtime._worker=audit_worker
                service=runtime.PhysicalTPService(config)
            else:
                from qualify_comfy_sampler import LogicalService
                runtime.RankRuntime=AuditRank;runtime.ResidencyWindow=AuditWindow
                import qualify_comfy_sampler as qualify
                qualify.RankRuntime=AuditRank
                CheckpointReader.read=audit_read
                service=LogicalService(config)
            worker_startup_ms=(time.perf_counter()-start)*1000
            clip=create_clip(service);model=create_model(service)
            for temperature in ('cold','warm'):
                service.call('audit_reset')
                record,positive,latent=request_pipeline(service,clip,model,vaes,args,
                    reference=args.output/'native_reference')
                # Validation/file I/O is outside reported timing boundaries.
                if temperature=='cold':
                    record['worker_startup_ms']=worker_startup_ms
                    record['startup_inclusive_ms']=worker_startup_ms+record['end_to_end_ms']
                record['temperature']=temperature
                record['runtime']=service.call('stats')
                result['requests'].append(record);save()
            if args.profile:
                # Additional real sampler evaluation, excluded from both
                # unprofiled requests; conditioning is already available.
                service.call('profile_start')
                try:
                    run_sample(model,positive,latent,1)
                finally:
                    result['traces']=service.call('profile_stop')
            result['passed']=True
    finally:
        if service:
            service.close();result['host_budget_after_close']=service.budget.snapshot()
        result['host_monitor']=monitor.close()
        result['parent_gpu']=[dict(device=i,allocated_peak=torch.cuda.max_memory_allocated(i),
            reserved_peak=torch.cuda.max_memory_reserved(i)) for i in range(torch.cuda.device_count())]
        save()
    if not result['passed']:raise AssertionError('Audit failed')
    print('AUDIT COMPLETE',args.strategy,len(result['requests']),flush=True)


if __name__=='__main__':main()
