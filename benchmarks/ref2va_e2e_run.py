"""Actual REF2VA requests, bounded benchmark-only observers; no operator changes."""
from __future__ import annotations
import argparse
from collections import Counter
from contextlib import contextmanager
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import threading
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'ComfyUI'), str(ROOT/'benchmarks')]
import torch
from patches import comfy_tp_runtime as rt
from patches.weight_residency import ResidencyWindow
from patches.windowed_tp import CheckpointReader
from patches.host_weight_cache import GiB

CURRENT = None
BASE_RANK = rt.RankRuntime
BASE_WORKER = rt._worker


def event():
    e = torch.cuda.Event(enable_timing=True)
    e.record()
    return e


class ObservedWindow(ResidencyWindow):
    def _stage(self,key,host,size):
        audit=getattr(CURRENT,'memory_audit',None)
        if audit:audit.mark('weight_H2D_submit_begin')
        result=super()._stage(key,host,size)
        if audit:audit.mark('weight_H2D_submit_end',static=True)
        return result

    def _evict(self,key):
        result=super()._evict(key)
        audit=getattr(CURRENT,'memory_audit',None)
        if audit:audit.mark('weight_retire_after_consumer_event',static=True)
        return result

    def _prepare(self, spec, **kwargs):
        owner = CURRENT
        start, cpu = time.monotonic_ns(), time.thread_time_ns()
        context = dict(owner.context)
        try:
            return super()._prepare(spec, **kwargs)
        finally:
            owner.cpu.append(dict(kind='prepare', key=str(spec.key[1:]), **context,
                start_ns=start, end_ns=time.monotonic_ns(), thread_cpu_ns=time.thread_time_ns()-cpu))

    @contextmanager
    def use(self, key):
        owner = CURRENT
        start = time.monotonic_ns()
        a = event()
        with super().use(key) as module:
            audit=getattr(owner,'memory_audit',None)
            if audit:
                audit.block=key[2] if len(key)>2 and key[1]=='dit' else None
                audit.mark('weight_lease_acquired',static=True)
            b = event()
            owner.cpu.append(dict(kind='consumer_wait', key=str(key[1:]), **owner.context,
                start_ns=start, end_ns=time.monotonic_ns()))
            try:
                yield module
            finally:
                c = event()
                owner.gpu.append((dict(kind='block_wait', key=str(key[1:]), **owner.context), a, b))
                owner.gpu.append((dict(kind='block_execution', key=str(key[1:]), **owner.context), b, c))


class ObservedRank(BASE_RANK):
    def __init__(self, *args):
        global CURRENT
        super().__init__(*args)
        CURRENT = self
        self.context = dict(command='startup', nfe=0)
        self.cpu, self.gpu, self.commands, self.layouts = [], [], [], []
        self.backends = Counter()
        self.peak_allocated = self.peak_reserved = 0
        self._install_observers()
        self.memory_audit=None
        if os.environ.get('MINIMAX_H3_GPU_MEMORY_PROFILE')=='1':
            from gpu_memory_overhead import MemoryAudit
            self.memory_audit=MemoryAudit(self)
        self.memory_command_counts=Counter()

    def _install_observers(self):
        owner = self
        # Layouts can be created in the parent and copied into worker packets,
        # bypassing __init__. Observe the actual attention consumer instead.
        import patches.h3_operator_policy as policy
        original_attention = policy.dit_attention
        def dit_attention(block,q,k,v,options):
            owner.record_layout(q,options)
            return original_attention(block,q,k,v,options)
        policy.dit_attention=dit_attention
        original_prepare=policy.prepare_dit_attention
        def prepare_attention(block,q,k,v,options=None):
            prepared=original_prepare(block,q,k,v,options)
            if prepared is not None:owner.record_layout(q,options or {})
            return prepared
        policy.prepare_dit_attention=prepare_attention
        # Observe actual executed entrypoints, not the configured policy string.
        import patches.kitchen_te_attention as kitchen_te
        import patches.sol_attention as sol_attention
        import comfy_kitchen as ck
        import patches.h3_lora as lora
        for module, name, label in ((kitchen_te,'attention','TE/kitchen_causal'),
                (ck,'int8_attention','kitchen/int8_attention'),
                (policy,'execute_dit_attention','kitchen/int8_attention'),
                (lora.NativeAlignedLoRA,'add_to','LoRA/column_or_native_bypass'),
                (lora.RowPartialLoRA,'add_to','LoRA/row_partial_bypass')):
            fn = getattr(module, name)
            def wrap(*a, _fn=fn, _label=label, **kw):
                owner.backends[_label] += 1
                begin = event()
                torch.cuda.nvtx.range_push(_label)
                try:
                    return _fn(*a, **kw)
                finally:
                    torch.cuda.nvtx.range_pop()
                    owner.gpu.append((dict(kind=_label, **owner.context), begin, event()))
            setattr(module, name, wrap)
        sol_fn=sol_attention.attention
        def observe_sol(*a, **kw):
            command=owner.context.get('command')
            label='DiT/sol_all_int8' if command=='denoise' else 'Sol/all_int8'
            owner.backends[label]+=1
            begin=event();torch.cuda.nvtx.range_push(label)
            try:
                return sol_fn(*a,**kw)
            finally:
                torch.cuda.nvtx.range_pop()
                owner.gpu.append((dict(kind=label,**owner.context),begin,event()))
        sol_attention.attention=observe_sol
        read = CheckpointReader.read
        def observed_read(reader, prefix, exclude=()):
            start, cpu = time.monotonic_ns(), time.thread_time_ns()
            try:
                return read(reader, prefix, exclude)
            finally:
                owner.cpu.append(dict(kind='checkpoint_read', prefix=prefix,
                    source=reader.path.name, **owner.context, start_ns=start,
                    end_ns=time.monotonic_ns(), thread_cpu_ns=time.thread_time_ns()-cpu))
        CheckpointReader.read = observed_read

    def record_layout(self,q,options):
        layout=options.get('minimax_h3_layout')
        if layout is None or layout.seq_len!=q.shape[-2]:
            raise AssertionError('Actual DiT attention requires its matching Native packed layout')
        record=dict(sequence_tokens=layout.seq_len,segments=layout.segments,query_shape=list(q.shape))
        if record not in self.layouts:
            self.layouts.append(record)
            print('REF2VA_LAYOUT',self.rank,record,flush=True)

    def dispatch(self, command, args=(), kwargs=None):
        kwargs = dict(kwargs or {})
        # Every RPC owns its events, including Temporal Video VAE. Leaving the
        # previous denoise context active misattributes decoder attention.
        self.context['command'] = command
        if self.memory_audit and command in ('begin_sample','end_sample','vae_prepare','vae_release',
                'decode_start','decode_complete','close'):
            self.context['command']=command
            self.memory_audit.begin(command)
        if command == 'idle_stats' and kwargs.pop('benchmark_snapshot', False):
            return dict(rank=self.rank,allocated=torch.cuda.memory_allocated(self.device),
                reserved=torch.cuda.memory_reserved(self.device),
                peak_allocated=max(self.peak_allocated,torch.cuda.max_memory_allocated(self.device)),
                peak_reserved=max(self.peak_reserved,torch.cuda.max_memory_reserved(self.device)))
        if command == 'idle_stats' and kwargs.pop('benchmark_ready', False):
            w = self.idle_window
            pending = bool(w and w.pending)
            next_spec = next((s for k,s in w.specs.items() if k not in w.entries), None) if w else None
            # Idle is done only after an admitted prefix reaches its byte cap,
            # all layers are resident, or a reported bounded stall is reached.
            full = bool(w and (next_spec is None or w._live_bytes()+next_spec.max_bytes>w.max_weight_bytes))
            ready = full and not pending
            if ready:
                w.copy_stream.synchronize()
                w._reap()
            result = super().dispatch(command, args, kwargs)
            result.update(benchmark_ready=ready, pending=pending, prefix_layers=len(w.entries) if w else 0)
            return result
        if command in ('te','refine','denoise','video_pixels','video_temporal_pair','audio_channels'):
            if command == 'denoise':
                self.context['nfe'] += 1
            self.context['command'] = command
            self.memory_command_counts[command]+=1
            if self.memory_audit and (command!='video_pixels' or self.memory_command_counts[command]<=2
                    or self.memory_command_counts[command]%32==0):
                self.memory_audit.begin(command)
            torch.cuda.reset_peak_memory_stats(self.device)
            start, begin = time.monotonic_ns(), event()
            torch.cuda.nvtx.range_push(f'{command}/nfe{self.context["nfe"]}/rank{self.rank}')
            try:
                result = super().dispatch(command, args, kwargs)
            finally:
                end = event()
                torch.cuda.nvtx.range_pop()
                allocated = torch.cuda.max_memory_allocated(self.device)
                reserved = torch.cuda.max_memory_reserved(self.device)
                self.peak_allocated = max(self.peak_allocated, allocated)
                self.peak_reserved = max(self.peak_reserved, reserved)
                row = dict(**self.context, start_ns=start, enqueue_end_ns=time.monotonic_ns(),
                    peak_allocated=allocated, peak_reserved=reserved)
                self.commands.append(row)
                self.gpu.append((dict(kind='command_gpu_span', **self.context), begin, end))
                print('REF2VA_RANK_COMMAND', self.rank, row, flush=True)
            return result
        if command == 'stats':
            torch.cuda.synchronize(self.device)
            result = super().dispatch(command, args, kwargs)
            result['benchmark'] = dict(rank=self.rank, cpu=self.cpu, commands=self.commands,
                gpu=[meta | dict(cuda_ms=a.elapsed_time(b)) for meta,a,b in self.gpu],
                layouts=self.layouts, executed_backends=dict(self.backends),
                peak_allocated=max(self.peak_allocated,torch.cuda.max_memory_allocated(self.device)),
                peak_reserved=max(self.peak_reserved,torch.cuda.max_memory_reserved(self.device)))
            return result
        return super().dispatch(command, args, kwargs)


class MemoryConnection:
    def __init__(self,connection):self.connection=connection
    def __getattr__(self,name):return getattr(self.connection,name)
    def send(self,message):
        audit=getattr(CURRENT,'memory_audit',None)
        if audit and audit.active:
            try:audit.flush()
            except BaseException as error:
                path=audit.directory/'capture_error.txt'
                path.write_text(type(error).__name__+'\n'+traceback.format_exc())
                raise
        return self.connection.send(message)


def worker(*args):
    rt.RankRuntime = ObservedRank
    rt.ResidencyWindow = ObservedWindow
    if os.environ.get('MINIMAX_H3_GPU_MEMORY_PROFILE')=='1':
        args=(*args[:-1],MemoryConnection(args[-1]))
    BASE_WORKER(*args)


class Service(rt.PhysicalTPService):
    def __init__(self, config):
        self.spans = []
        super().__init__(config)

    def call(self, command, *args, **kwargs):
        start = time.monotonic_ns()
        result = super().call(command, *args, **kwargs)
        self.spans.append(dict(command=command,start_ns=start,end_ns=time.monotonic_ns()))
        return result

    def wait_idle_ready(self, timeout=240):
        start = time.monotonic()
        while time.monotonic()-start < timeout:
            stats = self.call('idle_stats', benchmark_ready=True)
            if any(s['errors'] for s in stats):
                raise RuntimeError('Idle worker reported an error')
            if all(s['benchmark_ready'] for s in stats):
                self.pause_idle(keep=True)
                return dict(seconds=time.monotonic()-start,ranks=stats)
            time.sleep(.5)
        raise TimeoutError('Bounded idle prefix did not become ready; do not accept S2')


class Monitor:
    def __init__(self):
        self.rows = []
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.run,daemon=True)
        self.thread.start()

    def run(self):
        import psutil
        import pynvml as nv
        from patches.host_memory_pressure import CgroupMemory
        nv.nvmlInit()
        handles = [nv.nvmlDeviceGetHandleByIndex(i) for i in range(2)]
        parent, memory = psutil.Process(), CgroupMemory()
        while not self.stop.is_set():
            rss = {}
            for p in [parent,*parent.children(recursive=True)]:
                try:
                    rss[str(p.pid)] = p.memory_info().rss
                except psutil.Error:
                    pass
            per_device=[]
            for i,h in enumerate(handles):
                try:processes=[dict(pid=p.pid,used=p.usedGpuMemory) for p in nv.nvmlDeviceGetComputeRunningProcesses(h)]
                except nv.NVMLError:processes=None
                per_device.append(dict(device=i,used=nv.nvmlDeviceGetMemoryInfo(h).used,
                    utilization=nv.nvmlDeviceGetUtilizationRates(h).gpu,processes=processes))
            self.rows.append(dict(time_ns=time.monotonic_ns(),wall_time_ns=time.time_ns(),rss=rss,
                cgroup=memory.snapshot(),devices=per_device))
            # A 4-hour run retains at most 28,800 compact samples.
            if len(self.rows)>28800:
                self.rows = self.rows[::2]
            self.stop.wait(.5)

    def close(self):
        self.stop.set()
        self.thread.join(timeout=5)


def digest(tensor):
    import numpy as np
    t = tensor.detach().cpu().contiguous()
    raw = t.view(torch.uint8).numpy().reshape(-1)
    h = hashlib.sha256()
    for chunk in range(0,len(raw),8*2**20):
        h.update(memoryview(raw[chunk:chunk+8*2**20]))
    return dict(shape=list(t.shape),dtype=str(t.dtype),sha256=h.hexdigest(),
                finite=bool(np.isfinite(t.numpy()).all()))


def export_media(pixels, waveform, output):
    import av
    import numpy as np
    import wave
    from fractions import Fraction
    # Native H3 public decode returns B,T,H,W,C; image-oriented callers may
    # already have T,H,W,C. Never flatten multiple requests into one video.
    if pixels.ndim==5:
        if pixels.shape[0]!=1:raise ValueError('MP4 export requires exactly one video request')
        pixels=pixels[0]
    if pixels.ndim!=4 or pixels.shape[-1] not in (3,4):
        raise ValueError('Expected T,H,W,RGB or single-request B,T,H,W,RGB video')
    a = waveform[0] if waveform.ndim==3 else waveform
    if a.shape[0]>a.shape[-1]:
        a = a.T
    a = a.float().clamp(-1,1).contiguous()
    channels, samples = a.shape
    with wave.open(str(output/'audio.wav'),'wb') as stream:
        stream.setnchannels(channels);stream.setsampwidth(2);stream.setframerate(32000)
        for offset in range(0,samples,32000):
            stream.writeframes((a[:,offset:offset+32000].T.numpy()*32767).astype('<i2').tobytes())
    layout = 'stereo' if channels==2 else 'mono'
    with av.open(str(output/'video.mp4'),'w') as container:
        v = container.add_stream('libx264',rate=24)
        v.width, v.height = pixels.shape[-2], pixels.shape[-3]
        v.pix_fmt='yuv420p';v.options={'crf':'18','preset':'fast'}
        audio = container.add_stream('aac',rate=32000);audio.layout=layout
        # Interleave by timestamp without retaining another video-sized tensor.
        offset = 0
        for i, frame in enumerate(pixels):
            rgb = (frame[...,:3].float().clamp(0,1).numpy()*255).round().astype(np.uint8)
            vf = av.VideoFrame.from_ndarray(rgb,format='rgb24');vf.pts=i;vf.time_base=Fraction(1,24)
            for packet in v.encode(vf):container.mux(packet)
            limit = min(samples,round((i+1)*32000/24))
            while offset+1024<=limit:
                af=av.AudioFrame.from_ndarray(a[:,offset:offset+1024].numpy(),format='fltp',layout=layout)
                af.sample_rate=32000;af.pts=offset;af.time_base=Fraction(1,32000)
                for packet in audio.encode(af):container.mux(packet)
                offset+=1024
        if offset<samples:
            af=av.AudioFrame.from_ndarray(a[:,offset:].numpy(),format='fltp',layout=layout)
            af.sample_rate=32000;af.pts=offset;af.time_base=Fraction(1,32000)
            for packet in audio.encode(af):container.mux(packet)
        for packet in v.encode():container.mux(packet)
        for packet in audio.encode():container.mux(packet)
    return dict(video_bytes=(output/'video.mp4').stat().st_size,audio_bytes=(output/'audio.wav').stat().st_size)


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True)
    args=p.parse_args();config=json.loads(args.config.read_text())
    out=Path(config['output']);out.mkdir(parents=True,exist_ok=True)
    state=dict(status='starting',config=config,stages={},snapshots={})
    def save():(out/'result.json').write_text(json.dumps(state,indent=2,default=str))
    save()
    assert torch.cuda.device_count()==2
    assert all(torch.cuda.get_device_capability(i)==(7,5) and 'T4' in torch.cuda.get_device_name(i) for i in (0,1))
    torch.set_grad_enabled(False);torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    from patches.h3_operator_policy import H3OperatorPolicy
    from patches.comfy_sampler import create_clip,create_model,ParallelDecodeHandoff
    from pipeline_residency_audit import load_vae
    from comfy_extras.nodes_minimax_h3 import MiniMaxH3ReferenceToVideo,MiniMaxH3SigmaShift
    import comfy.samplers
    import comfy.model_management
    from comfy.nested_tensor import NestedTensor
    from PIL import Image
    import numpy as np
    operators=H3OperatorPolicy(**config['operators'])
    rt._worker=worker
    runtime=rt.RuntimeConfig(te_path=config['paths']['te'],dit_path=config['paths']['dit'],
        layers=50,weight_bytes=6*GiB,pinned_bytes=6*GiB,cache_bytes=3*GiB,host_bytes=16*GiB,
        reserve_bytes=int(config['reserve_gib']*GiB),timeout_seconds=1800,
        prefetch_lifecycle=True,memory_pressure_aware=True,cross_request_te=False,
        cross_model_cpu=True,idle_residency=config['idle'],idle_weight_bytes=4*GiB,
        lora_stack=tuple((config['paths']['lora'],1.) for _ in [0])
            if config['lora'] and not config.get('offline_lora') else (),
        lora_row_partial_reduce=config['lora'] and not config.get('offline_lora'),operators=operators)
    service=None;monitor=Monitor();vaes={};parent_memory=None
    if config.get('memory_profile'):
        from types import SimpleNamespace
        from gpu_memory_overhead import MemoryAudit
        parent_runtime=SimpleNamespace(rank=0,device=torch.device('cuda:0'),window=None,
            model=None,vae_model=None,context={'nfe':0},extra_models=[])
        parent_memory=MemoryAudit(parent_runtime,role='parent_gpu0',install_wrappers=False)
    def snap(label):
        state['snapshots'][label]=dict(time_ns=time.monotonic_ns(),
            parent_allocated=torch.cuda.memory_allocated(0),parent_reserved=torch.cuda.memory_reserved(0),
            ranks=service.call('idle_stats',benchmark_snapshot=True) if service else [])
        save()
    def stage(label,fn):
        state['status']=label;save()
        if parent_memory:parent_memory.begin(label)
        start=time.monotonic_ns();a=event()
        try:
            result=fn()
            b=event();b.synchronize()
            state['stages'][label]=dict(start_ns=start,end_ns=time.monotonic_ns(),parent_cuda_span_ms=a.elapsed_time(b))
            if parent_memory and label=='vae_construction':
                parent_runtime.extra_models=[('vae_'+k+'_weights',v.first_stage_model) for k,v in result.items()]
        finally:
            if parent_memory and parent_memory.active:parent_memory.flush()
        save();print('REF2VA_STAGE',config['name'],label,state['stages'][label],flush=True)
        return result
    try:
        setup=time.monotonic()
        service=Service(runtime)
        clip=create_clip(service);model=create_model(service)
        model=MiniMaxH3SigmaShift.execute(model,12.,config['audio_shift'])[0]
        images={f'ref_image_{i+1}':torch.from_numpy(np.asarray(Image.open(path).convert('RGB'),dtype=np.float32)/255)[None]
                for i,path in enumerate(config['images'])}
        state['setup_seconds']=time.monotonic()-setup
        snap('initial_idle')
        if config['idle']:
            start=time.monotonic_ns()
            if not service.request_idle():raise RuntimeError('Idle prefix not admitted')
            state['preload']=service.wait_idle_ready()
            state['preload']['start_ns']=start;state['preload']['end_ns']=time.monotonic_ns()
            snap('post_preload')
        # Foreground trigger includes VAE construction, reference encoding, TE,
        # sampling, both decoders and MP4/WAV encoding. Downloads/service setup
        # and S2's explicit idle window are separately reported.
        trigger=time.monotonic_ns();state['trigger_ns']=trigger
        vaes=stage('vae_construction',lambda:{k:load_vae(config['paths'][k],k) for k in ('video','audio')})
        positive,latent=stage('reference_and_TE',lambda:tuple(MiniMaxH3ReferenceToVideo.execute(
            clip,config['prompt'],864,480,config['frames'],ref_image_size='match',
            vae=vaes['video'],ref_images=images))[:2])
        latent=latent['samples']
        snap('post_TE')
        generator=torch.Generator().manual_seed(config['seed'])
        noise=NestedTensor([torch.randn(t.shape,generator=generator) for t in latent.unbind()])
        guider=comfy.samplers.CFGGuider(model);guider.set_conds(positive,positive);guider.set_cfg(1.)
        sigmas=comfy.samplers.calculate_sigmas(model.model.model_sampling,'simple',config['steps'])
        sampled=stage('sampler',lambda:guider.sample(noise,latent,
            comfy.samplers.sampler_object(config['sampler']),sigmas,disable_pbar=False,seed=config['seed']).cpu())
        snap('post_denoise')
        video,audio=sampled.unbind()
        decoders={k:ParallelDecodeHandoff(vaes[k],service,config['paths'][k],'auto') for k in ('video','audio')}
        pixels=stage('video_decode',lambda:decoders['video'].decode(video).cpu())
        waveform=stage('audio_decode',lambda:decoders['audio'].decode(audio).cpu())
        state['decoded_shapes']=dict(pixels=list(pixels.shape),waveform=list(waveform.shape))
        save()
        state['media']=stage('file_export',lambda:export_media(pixels,waveform,out))
        ready=time.monotonic_ns();state['output_ready_ns']=ready
        state['foreground_seconds']=(ready-trigger)/1e9
        state['preload_plus_foreground_seconds']=state['foreground_seconds']+state.get('preload',{}).get('seconds',0)
        state['outputs']={k:digest(t) for k,t in dict(conditioning=positive[0][0],video=video,audio=audio,pixels=pixels,waveform=waveform).items()}
        if not all(t['finite'] for t in state['outputs'].values()):
            raise AssertionError('Nonfinite output tensors; generation is not an accepted benchmark')
        state['decode']={k:dict(strategy=d.last_strategy,fallback=d.last_fallback) for k,d in decoders.items()}
        state['ranks']=service.call('stats');state['service_calls']=service.spans
        for rank in state['ranks']:
            bench=rank['benchmark']
            if not bench['layouts']:raise AssertionError('No actual packed DiT layout was observed')
            te_backend={'kitchen_causal':'TE/kitchen_causal'}.get(operators.te_attention)
            if te_backend and not bench['executed_backends'].get(te_backend):raise AssertionError('Configured TE kernel not executed')
            dit_backend={'kitchen':'kitchen/int8_attention','sol':'DiT/sol_all_int8'}.get(operators.dit_attention)
            if dit_backend and not bench['executed_backends'].get(dit_backend):raise AssertionError('Configured DiT attention kernel not executed')
            if config['lora']:
                denoise=[c for c in bench['commands'] if c['command']=='denoise']
                if max(c['peak_allocated'] for c in denoise)>=14.5*GiB:
                    raise AssertionError('L1 exceeded the allocated-memory acceptance bound')
        state['status']='complete'
        state['correctness_scope']='S1/S2 compared by tensor hashes outside request timing; no new dense Native reference generation or L1 dense-equivalence claim'
    except BaseException as error:
        state.update(status='failed',error_type=type(error).__name__)
        (out/'exception.txt').write_text(traceback.format_exc())
        # Preserve the compact decode inputs only on failure, allowing an
        # export/decode recovery without repeating TE or the 20-step sampler.
        if 'video' in locals() and 'audio' in locals():
            torch.save(dict(video=video.cpu(),audio=audio.cpu()),out/'decode_recovery.pt')
            state['decode_recovery_bytes']=(out/'decode_recovery.pt').stat().st_size
        if service and not service.closed:
            try:
                state['ranks']=service.call('stats');state['service_calls']=service.spans
            except Exception as capture_error:
                state['failure_metrics_error']=type(capture_error).__name__
        raise
    finally:
        if service:service.close()
        vaes.clear();gc.collect();comfy.model_management.unload_all_models();torch.cuda.empty_cache()
        state['cleanup_parent']=dict(allocated=torch.cuda.memory_allocated(0),reserved=torch.cuda.memory_reserved(0))
        monitor.close()
        import gzip
        with gzip.open(out/'memory_samples.json.gz','wt') as stream:json.dump(monitor.rows,stream)
        save()


if __name__=='__main__':main()
