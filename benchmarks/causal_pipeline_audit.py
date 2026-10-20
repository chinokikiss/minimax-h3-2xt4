"""Actual multi-request ComfyUI inference, with process-local diagnostic hooks.

No extra denoiser evaluations, no stage fences added by the diagnostic runner,
and no checking/export/stats calls between continuous requests. Timing and
trace passes run in separate fresh processes with identical policy/input.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time

import pipeline_residency_audit as audit
from patches import comfy_tp_runtime as runtime
from patches.comfy_sampler import ParallelDecodeHandoff, create_clip, create_model
from patches.weight_residency import ResidencyWindow
import causal_trace as tracing
import torch


class MemoryMonitor:
    def __init__(self):
        self.rows=[];self.unsupported=set();self.finished=threading.Event()
        self.thread=threading.Thread(target=self.run,daemon=True);self.thread.start()

    def run(self):
        import psutil
        try:
            import pynvml as nv
            nv.nvmlInit();handles=[nv.nvmlDeviceGetHandleByIndex(i) for i in range(nv.nvmlDeviceGetCount())]
        except Exception as error:
            self.unsupported.add('NVML/'+type(error).__name__);handles=[]
        parent=psutil.Process()
        while not self.finished.is_set():
            vm=psutil.virtual_memory();rss=0;devices=[]
            for process in [parent,*parent.children(recursive=True)]:
                try:rss+=process.memory_info().rss
                except (psutil.NoSuchProcess,psutil.AccessDenied):pass
            for i,handle in enumerate(handles):
                values=dict(device=i,used=nv.nvmlDeviceGetMemoryInfo(handle).used)
                for name,fn in (
                    ('util_percent',lambda:nv.nvmlDeviceGetUtilizationRates(handle).gpu),
                    ('rx_bytes_s',lambda:nv.nvmlDeviceGetPcieThroughput(handle,nv.NVML_PCIE_UTIL_RX_BYTES)*1024),
                    ('tx_bytes_s',lambda:nv.nvmlDeviceGetPcieThroughput(handle,nv.NVML_PCIE_UTIL_TX_BYTES)*1024)):
                    try:values[name]=fn()
                    except nv.NVMLError as error:
                        values[name]=None;self.unsupported.add(name+'/'+type(error).__name__)
                devices.append(values)
            self.rows.append(dict(time_ns=time.time_ns(),monotonic_ns=time.monotonic_ns(),
                process_tree_rss=rss,available=vm.available,system_used=vm.total-vm.available,devices=devices))
            self.finished.wait(.05)

    def close(self):
        self.finished.set();self.thread.join()
        return dict(samples=self.rows,unsupported=sorted(self.unsupported),
            semantics='NVML PCIe KB/s x1024 sampled at 50ms; approximate bus traffic, not exact weight payload or SM occupancy')


@dataclass(frozen=True)
class DiagnosticConfig(audit.AuditConfig):
    pass


class PolicyWindow(audit.AuditWindow):
    # Retain the prior audit's exact admission/ablation settings. Remove its
    # duplicate diagnostic events; new observers wrap the production methods.
    def _prepare(self, spec):
        return ResidencyWindow._prepare(self, spec)

    def _stage(self, key, host, size):
        return ResidencyWindow._stage(self, key, host, size)

    @contextmanager
    def use(self, key):
        if self.audit.config.strategy == 'sync':
            torch.cuda.current_stream(self.device).synchronize()
        with ResidencyWindow.use(self, key) as module:
            yield module

    def close(self):
        return ResidencyWindow.close(self)


class DiagnosticRank(audit.AuditRank):
    def dispatch(self, command, args=(), kwargs=None):
        kwargs = dict(kwargs or {})
        context = kwargs.pop('_diagnostic_context', None)
        if context:
            tracing.ACTIVE.context = context
        recorder = tracing.ACTIVE
        if command == 'trace_start':
            recorder.start(*args)
            torch.cuda.reset_peak_memory_stats(self.device)
            return None
        if command == 'trace_stop':
            recorder.stop()
            return None
        with recorder.span('rank_dispatch/'+command):
            result = runtime_rank_dispatch(self, command, args, kwargs)
        if command == 'stats':
            result['gpu'] = dict(allocated_peak=torch.cuda.max_memory_allocated(self.device),
                reserved_peak=torch.cuda.max_memory_reserved(self.device),
                driver_free=torch.cuda.mem_get_info(self.device)[0])
        return result


runtime_rank_dispatch = audit.NATIVE_RANK.dispatch


class ObservedConnection:
    """Observe the original worker's actual receive/acknowledgement boundary."""
    def __init__(self, connection, recorder):
        self.connection, self.recorder = connection, recorder

    def recv(self):
        with self.recorder.span('worker_receive'):
            message = self.connection.recv()
        if len(message) == 4:
            sequence, command, _, kwargs = message
            if kwargs.get('_diagnostic_context'):
                self.recorder.context = kwargs['_diagnostic_context']
            self.recorder.emit('command_received', sequence=sequence, operation=command)
        return message

    def send(self, message):
        with self.recorder.span('worker_ack_send', sequence=message[0]):
            return self.connection.send(message)

    def close(self):
        return self.connection.close()


def diagnostic_worker(rank, config, budget, port, connection):
    torch.cuda.set_device(rank)
    recorder = tracing.Recorder(config.audit_dir,rank,f'cuda:{rank}')
    tracing.install(recorder)
    runtime.RankRuntime = DiagnosticRank
    runtime.ResidencyWindow = PolicyWindow
    try:
        audit.NATIVE_WORKER(rank,config,budget,port,ObservedConnection(connection,recorder))
    finally:
        tracing.restore()


class DiagnosticService(runtime.PhysicalTPService):
    def __init__(self,config,recorder):
        self.recorder=recorder
        super().__init__(config)

    def call(self,command,*args,**kwargs):
        context = dict(self.recorder.context)
        if command == 'denoise':
            self.recorder.context['nfe'] += 1
            context = dict(self.recorder.context)
        if command == 'te':
            context['stage']='TE'
        elif command in ('refine','denoise'):
            context['stage']='DiT'
        context['command']=self.sequence+1
        if command not in ('trace_start','trace_stop'):
            kwargs['_diagnostic_context']=context
        with self.recorder.span('service_call/'+command, **context):
            return super().call(command,*args,**kwargs)


def request(service,clip,model,vaes,args,recorder,index):
    recorder.context=dict(request=index,stage='request',command=0,nfe=0)
    start=time.monotonic_ns()
    stages={}
    def stage(name,fn):
        recorder.context['stage']=name
        begin=time.monotonic_ns()
        with recorder.span('stage/'+name):
            out=fn()
        finish=time.monotonic_ns()
        stages[name]=dict(start_ns=begin,end_ns=finish,host_wall_ms=(finish-begin)/1e6)
        if getattr(args,'stage_observer',None):args.stage_observer(name)
        return out
    with recorder.span('request'):
        if not vaes:
            vaes.update(stage('vae_construction',lambda:{k:audit.load_vae(args.paths[k],k)
                                                       for k in ('video','audio')}))
            # Reference encoding is a real call nested inside the original
            # Ref2VA node; observe it without rebuilding conditioning math.
            original_encode=vaes['video'].encode
            def reference_encode(*a,**kw):
                with recorder.span('reference_encode'):
                    return original_encode(*a,**kw)
            vaes['video'].encode=reference_encode
        positive,latent=stage('reference_and_TE',lambda:audit.conditioning(
            clip,vaes['video'],args.width,args.height))
        sampled=stage('sampler',lambda:audit.run_sample(model,positive,latent,args.steps))
        video,audio=sampled.unbind()
        decoders={kind:ParallelDecodeHandoff(vaes[kind],service,args.paths[kind],args.vae_mode)
                  for kind in ('video','audio')}
        pixels=stage('video_decode',lambda:decoders['video'].decode(video).cpu())
        waveform=stage('audio_decode',lambda:decoders['audio'].decode(audio).cpu())
        # CPU-delivered output is the production request boundary. No new
        # cross-rank fences: worker acknowledgements already guarantee them.
        outputs=dict(conditioning=positive[0][0].cpu(),video=video.cpu(),audio=audio.cpu(),
                     pixels=pixels,waveform=waveform)
    end=time.monotonic_ns()
    return dict(request=index,start_ns=start,end_ns=end,latency_ms=(end-start)/1e6,
        stages=stages,nfe=recorder.context['nfe'],
        decode={k:dict(strategy=d.last_strategy,fallback=d.last_fallback)
                for k,d in decoders.items()},
        shapes={k:list(v.shape) for k,v in outputs.items()}),outputs


def validate(outputs,reference,output,request_id,vae_mode):
    checks={}
    for name,actual in outputs.items():
        expected=torch.load(reference/f'{name}.pt',map_location='cpu',weights_only=True)
        check=audit.strict_metrics(actual,expected)
        if name == 'waveform' and vae_mode=='auto':
            # Previously accepted CP rounding contract, explicitly separate
            # from Native bit-exactness. Never present this as a strict pass.
            check['native_strict_passed']=check['passed']
            check['contract']='user-accepted CP mono rounding; rel_l2<=2e-5, max_abs<=5e-3, cosine>=0.999999'
            check['passed']=bool(check.get('valid') and check['rel_l2']<=2e-5
                and check['max_abs']<=.005 and check['cosine']>=.999999)
        else:
            check['passed']=bool(check['passed'] and check.get('is_bit_exact',False))
        checks[name]=check
        del expected
    (output/f'correctness_{request_id}.json').write_text(json.dumps(checks,indent=2))
    if not all(c['passed'] for c in checks.values()):
        raise AssertionError('Native comparison failed: '+str(request_id))
    return checks


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--case',choices=('direct_sync','direct_p1','direct_p1_p2','offline_p1_p2'),required=True)
    parser.add_argument('--capture',choices=('timing','kineto','nsys'),default='timing')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--checkpoints',type=Path,required=True)
    parser.add_argument('--reference',type=Path,required=True)
    parser.add_argument('--rank-checkpoint-dir')
    parser.add_argument('--requests',type=int,default=2)
    parser.add_argument('--warmup',type=int,default=0)
    parser.add_argument('--steps',type=int,default=4)
    parser.add_argument('--width',type=int,default=512)
    parser.add_argument('--height',type=int,default=512)
    parser.add_argument('--layers',type=int,default=50)
    parser.add_argument('--vae-mode',choices=('auto','native'),default='auto')
    args=parser.parse_args()
    if args.requests<1 or args.requests>4 or args.warmup<0:
        parser.error('Bound output retention to 1..4 measured requests')
    if args.layers!=50:
        parser.error('Physical diagnostic requires both full 50-layer models')
    if args.case=='offline_p1_p2' and not args.rank_checkpoint_dir:
        parser.error('Offline case requires a verified rank-local checkpoint')
    if torch.cuda.device_count()!=2 or any('T4' not in torch.cuda.get_device_name(i) for i in range(2)):
        raise RuntimeError('Performance evidence requires physical 2xT4')
    args.output.mkdir(parents=True,exist_ok=True)
    args.paths=json.loads(args.checkpoints.read_text())
    torch.set_grad_enabled(False);torch.backends.cuda.matmul.allow_tf32=False
    audit.enable('scalemax')
    nsys=shutil.which('nsys')
    result=dict(passed=False,case=args.case,capture=args.capture,physical=True,
        clock_anchor=dict(wall_ns=time.time_ns(),monotonic_ns=time.monotonic_ns()),
        workload=dict(width=args.width,height=args.height,frames=124,steps=args.steps,layers=args.layers,
            prompt='A bird walks slowly through a quiet garden. Natural movement, soft daylight.',
            reference='deterministic gradient image; actual packed native Ref2VA sequence',seed=1729),
        tools=dict(nsys=nsys,torch=str(torch.__version__),cuda=torch.version.cuda),
        requests=[],warmup=[],correctness=[],
        latency_semantics='output-ready CPU wall time; profiler overhead retained in diagnostic latency',
        source_sha256={p.relative_to(audit.ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [*list((audit.ROOT/'patches').glob('*.py')),Path(__file__),Path(tracing.__file__)]})
    save=lambda:(args.output/'result.json').write_text(json.dumps(result,indent=2))
    save()
    recorder=tracing.Recorder(args.output,'parent','cuda:0')
    tracing.install(recorder)
    runtime._worker=diagnostic_worker
    adaptive=args.case!='direct_sync'
    config=DiagnosticConfig(args.paths['te'],args.paths['dit'],layers=50,
        weight_bytes=9*audit.GiB,te_weight_bytes=int(11.7*audit.GiB),
        cache_bytes=3*audit.GiB,host_bytes=24*audit.GiB,reserve_bytes=int(1.5*audit.GiB),
        strategy='adaptive' if adaptive else 'sync',audit_dir=str(args.output.resolve()),
        timeout_seconds=300,cross_model_cpu=adaptive,
        cross_request_te=args.case in ('direct_p1_p2','offline_p1_p2'),
        rank_checkpoint_dir=args.rank_checkpoint_dir if args.case=='offline_p1_p2' else None)
    monitor=MemoryMonitor();service=None;vaes={};retained=[]
    try:
        service=DiagnosticService(config,recorder)
        clip=create_clip(service);model=create_model(service)
        # Capture the warmup request too, with explicit warmup IDs. It follows
        # the same real request path and is excluded from latency comparisons.
        # Starting here preserves the producer side of the first handoff.
        if args.capture!='timing':
            service.call('trace_start','batch',args.capture=='kineto')
            recorder.start('batch',args.capture=='kineto')
        for index in range(args.warmup):
            record,outputs=request(service,clip,model,vaes,args,recorder,f'warmup{index}')
            result['warmup'].append(record)
            del outputs
        torch.cuda.reset_peak_memory_stats(0)
        previous=None
        with recorder.span('continuous_batch'):
            for index in range(args.requests):
                record,outputs=request(service,clip,model,vaes,args,recorder,index)
                record['handoff_gap_ms']=None if previous is None else (record['start_ns']-previous)/1e6
                if previous is not None:
                    recorder.emit('request_handoff',previous,record['start_ns'],from_request=index-1,to_request=index)
                result['requests'].append(record);retained.append(outputs)
                previous=record['end_ns']
                # No output checking, logging, stats, file export or sleep here.
        if args.capture!='timing':
            recorder.stop();service.call('trace_stop')
        result['runtime']=service.call('stats')
        result['parent_gpu']=dict(allocated_peak=torch.cuda.max_memory_allocated(0),
                                 reserved_peak=torch.cuda.max_memory_reserved(0))
        result['warmup_correctness']='Warmup outputs discarded without validation I/O; all measured requests validated after the continuous batch'
        for index,outputs in enumerate(retained):
            result['correctness'].append(validate(outputs,args.reference,args.output,index,args.vae_mode))
        retained.clear()
        result['passed']=True
    finally:
        if recorder.enabled:
            recorder.stop()
        if service:
            service.close();result['host_budget_after_close']=service.budget.snapshot()
        result['host_monitor']=monitor.close()
        tracing.restore();save()
    print('AUDIT CAUSAL COMPLETE',args.case,args.capture,len(result['requests']),flush=True)


if __name__=='__main__':main()
