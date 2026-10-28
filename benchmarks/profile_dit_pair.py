"""Two cold/warm DiT calls with identical payloads in one persistent TP2 session."""
import argparse
from contextlib import contextmanager
from dataclasses import asdict
import gzip
import json
import os
from pathlib import Path
import sys
import threading
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'ComfyUI'),str(ROOT/'benchmarks')]
import torch
import ref2va_e2e_run as observation
from patches import comfy_tp_runtime as rt
from dit_performance_observer import Phases
from dit_pair_contract import fixture,identity,digest,TOKENS,SEGMENTS

class PairWindow(observation.ObservedWindow):
    @contextmanager
    def use(self,key):
        owner=observation.CURRENT
        phases=owner.phases
        previous=phases.block;phases.block=str(key[1:])
        def counters():
            return dict(host_wait_ms=self.stats['host_wait_ms'],h2d_bytes=self.stats['h2d_bytes'],
                hits=self.stats['hits'],lookahead=self.lookahead,cpu_lookahead=self.cpu_lookahead,
                resident_bytes=self._live_bytes(),pending_keys=[str(k[1:]) for k in self.pending])
        before=counters();start=time.monotonic_ns()
        try:
            with phases.span('weight_acquire',gpu=False):
                lease=super().use(key)
                module=lease.__enter__()
            acquired=time.monotonic_ns()
            try:
                with phases.span('block_compute'):yield module
            except BaseException:
                lease.__exit__(*sys.exc_info());raise
            else:lease.__exit__(None,None,None)
            finally:
                owner.weight_uses.append(dict(run=owner.context['nfe'],key=str(key[1:]),start_ns=start,
                    acquired_ns=acquired,end_ns=time.monotonic_ns(),before=before,after=counters()))
        finally:phases.block=previous

    def _stage(self,key,host,size):
        owner=observation.CURRENT
        with owner.host_span('H2D_submit',str(key[1:]),bytes=size):return super()._stage(key,host,size)

    def _prepare(self,spec,**kwargs):
        owner=observation.CURRENT
        with owner.host_span('pin_and_prepare',str(spec.key[1:])):return super()._prepare(spec,**kwargs)

    def _room(self,size,deadline,blocking):
        owner=observation.CURRENT
        with owner.host_span('memory_admission',str(deadline),bytes=size,blocking=blocking):
            return super()._room(size,deadline,blocking)


class PairRank(observation.ObservedRank):
    def __init__(self,*args):
        super().__init__(*args)
        self.phases=Phases(self.rank);self.phases.install()
        from patches import optimized_tp,dit_column_linear,tilelang_column_epilogue
        self.phases.wrap(optimized_tp,'native_int_mm','column/native_INT8_GEMM')
        self.phases.wrap(dit_column_linear,'_prepare_activation','column/ConvRot_quantize')
        self.phases.wrap(tilelang_column_epilogue,'dequantize','column/TileLang_epilogue')
        self.weight_uses=[];self.host_events=[];self.per_call=[]
        self.memory_audit.history_limit=200000
        from worker_cuda_checkpoint import CUDAFlushCheckpoint
        self.checkpoints=CUDAFlushCheckpoint(self.rank,os.environ['H3_PAIR_OUTPUT'],device=self.device)
        from patches.windowed_tp import CheckpointReader
        original=CheckpointReader.read_tensor
        def read_tensor(reader,name,*args,**kwargs):
            with self.host_span('checkpoint_read_tensor',name,source=reader.path.name):
                return original(reader,name,*args,**kwargs)
        CheckpointReader.read_tensor=read_tensor
        from patches import tp2_bundle,tp2_disk_bake
        for module in (tp2_bundle,tp2_disk_bake):
            original_sha=module.sha
            def hashed(path,_fn=original_sha):
                with self.host_span('file_hash',Path(path).name):return _fn(path)
            module.sha=hashed
        state_proof=tp2_bundle.state_hashes
        def prove(module):
            with self.host_span('native_tensor_state_proof'):
                return state_proof(module)
        tp2_bundle.state_hashes=prove

    def _phase(self,kind):
        with self.host_span('runtime_phase',kind,construction=kind!=self.kind):
            return super()._phase(kind)

    @contextmanager
    def host_span(self,name,key='',**metadata):
        run=self.context.get('nfe',0);start=time.monotonic_ns();cpu=time.thread_time_ns()
        torch.cuda.nvtx.range_push(f'H3_PAIR/run={run}/rank={self.rank}/{name}/{key}')
        try:yield
        finally:
            torch.cuda.nvtx.range_pop()
            self.host_events.append(dict(name=name,key=key,run=run,rank=self.rank,pid=os.getpid(),
                tid=threading.get_native_id(),start_ns=start,end_ns=time.monotonic_ns(),
                thread_cpu_ns=time.thread_time_ns()-cpu,**metadata))

    def dispatch(self,command,args=(),kwargs=None):
        self.context['command']=command
        if command=='denoise':
            run=self.context['nfe']+1
            self.phases.enabled=True
            first=len(self.phases.cpu);first_gpu=len(self.phases.gpu)
            before=dict(self.backends)
            cache_before={k:v for k,v in self.cache.stats.items() if isinstance(v,(int,float))}
            window_before={k:v for k,v in self.window.stats.items() if isinstance(v,(int,float))}
            torch.cuda.nvtx.range_push(f'H3_PAIR_CALL/run={run}/rank={self.rank}')
            try:
                return super().dispatch(command,args,kwargs)
            finally:
                torch.cuda.nvtx.range_pop();self.phases.enabled=False
                self.per_call.append(dict(run=run,phase_cpu_range=[first,len(self.phases.cpu)],
                    phase_gpu_range=[first_gpu,len(self.phases.gpu)],
                    backend_counts={k:v-before.get(k,0) for k,v in self.backends.items()},
                    cache_stats_before=cache_before,window_stats_before=window_before,
                    cache_stats={k:v for k,v in self.cache.stats.items() if isinstance(v,(int,float))},
                    window_stats={k:v for k,v in self.window.stats.items() if isinstance(v,(int,float))}))
        if command=='pair_checkpoint':return self.checkpoints.flush(args[0],resume=args[1])
        result=super().dispatch(command,args,kwargs)
        if command=='stats':
            result.update(pair_calls=self.per_call,performance_phases=self.phases.report(),
                          weight_uses=self.weight_uses,host_events=self.host_events)
        return result


def worker(*args):
    rt.RankRuntime=PairRank;rt.ResidencyWindow=PairWindow
    reply=rt.worker_reply
    def measured_reply(*values):
        with observation.CURRENT.host_span('reply_D2H'):
            return reply(*values)
    rt.worker_reply=measured_reply
    observation.BASE_WORKER(*args[:-1],PairConnection(args[-1]))


class PairConnection(observation.MemoryConnection):
    def send(self,message):
        owner=observation.CURRENT
        if owner and owner.memory_audit and owner.memory_audit.active:
            with owner.host_span('observer/lifetime_export'):
                return super().send(message)
        return self.connection.send(message)


class PairService(observation.Service):
    def close(self):
        if self.closed:return
        try:
            self.call('pair_checkpoint','worker_close',False)
            self.call('close')
            for process in self.processes:process.join(timeout=20)
        finally:self.abort()


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args();sys.argv=sys.argv[:1]
    args.output.mkdir(exist_ok=False)
    os.environ.update(MINIMAX_H3_GPU_MEMORY_PROFILE='1',MINIMAX_H3_MEMORY_DIR=str(args.output/'memory'),H3_PAIR_OUTPUT=str(args.output))
    torch.set_grad_enabled(False);torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    from patches.h3_operator_policy import H3OperatorPolicy
    from patches.h3_tensor_flow import TensorFlowPolicy
    from safetensors.torch import save_file
    empty=args.output/'unused_te.safetensors';save_file({},str(empty))
    config=rt.RuntimeConfig(str(empty),str(args.checkpoint),lora_stack=(),bundle_offline_lora=False,
        cross_request_te=False,cross_model_cpu=False,idle_residency=False,
        operators=H3OperatorPolicy(),tensor_flow=TensorFlowPolicy('fp16'))
    rt._worker=worker
    state=dict(status='starting',runtime=asdict(config),scope='Two identical DiT denoiser calls, no trajectory/TE/VAE',
        operator_warmups=0,observer_enabled_equally_for_both=True,cases=[])
    def save():(args.output/'result.json').write_text(json.dumps(state,indent=2,default=str))
    service=None;monitor=observation.Monitor();save()
    try:
        inputs=fixture();state['input_identity']=identity(inputs);save()
        begin=time.monotonic();service=PairService(config)
        state['service_construction_seconds']=time.monotonic()-begin
        begin=time.monotonic();service.call('begin_sample')
        state['begin_sample_seconds']=time.monotonic()-begin
        begin=time.monotonic();inputs['context']=service.call('refine',inputs['context'])
        state['native_refiner_seconds']=time.monotonic()-begin
        before=[digest(t) for t in inputs['x']]+[digest(inputs['context'])]
        for run in (1,2):
            start=time.monotonic_ns()
            output=service.call('denoise',**inputs)
            end=time.monotonic_ns()
            assert [digest(t) for t in inputs['x']]+[digest(inputs['context'])]==before
            summaries=[dict(shape=list(t.shape),finite=bool(torch.isfinite(t).all()),sha256=digest(t)) for t in output]
            assert all(s['finite'] for s in summaries)
            del output
            state['cases'].append(dict(run=run,start_ns=start,end_ns=end,rpc_seconds=(end-start)/1e9,outputs=summaries));save()
            service.call('pair_checkpoint',f'call_{run}_complete',True)
            print('DIT_PAIR_CALL_COMPLETE',run,(end-start)/1e9,flush=True)
        ranks=service.call('stats');state['ranks']=ranks;state['parent_spans']=service.spans
        for rank in ranks:
            assert rank['calls']==2 and len(rank['pair_calls'])==2
            assert all(x['sequence_tokens']==TOKENS and [list(s) for s in x['segments']]==[list(s) for s in SEGMENTS]
                       for x in rank['benchmark']['layouts'])
            assert all(r['backend_counts'].get('kitchen/int8_attention')==50 for r in rank['pair_calls'])
        state.update(status='complete',outputs_byte_exact=state['cases'][0]['outputs']==state['cases'][1]['outputs'])
        assert state['outputs_byte_exact']
    except BaseException as error:
        state.update(status='failed',error=repr(error));(args.output/'exception.txt').write_text(traceback.format_exc());raise
    finally:
        if service:service.close()
        state['worker_exit']=[dict(pid=p.pid,exit_code=p.exitcode,alive=p.is_alive()) for p in service.processes] if service else []
        monitor.close()
        with gzip.open(args.output/'nvml_samples.json.gz','wt') as stream:json.dump(monitor.rows,stream)
        save()


if __name__=='__main__':main()
