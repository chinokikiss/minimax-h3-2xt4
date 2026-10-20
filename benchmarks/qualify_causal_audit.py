"""Two logical ranks on the 5060 Ti: tracing semantics, not T4 performance."""
from dataclasses import asdict
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import sys
import subprocess
from types import SimpleNamespace

import causal_pipeline_audit as diagnostic
import causal_trace as tracing
import pipeline_residency_audit as audit
from patches import comfy_tp_runtime as runtime
from patches.comfy_sampler import create_clip,create_model
import qualify_comfy_sampler as qualify
import torch


def child(config,queue):
    queue.put(asdict(config))


def main():
    root=audit.ROOT;output=root/'semantic_results/causal_qualification';output.mkdir(exist_ok=True)
    if '--make-reference' in sys.argv:
        torch.backends.cudnn.allow_tf32=False
        sys.argv=[sys.argv[0],'--strategy','native','--layers','2','--steps','4',
            '--width','128','--height','128','--qualification','--output',str(output/'native')]
        audit.main()
        return
    result=dict(passed=False,physical=False,scope='Actual compact Native Ref2VA/4-NFE path on logical ranks; diagnostic equivalence only',cases=[])
    save=lambda:(output/'result.json').write_text(json.dumps(result,indent=2))
    sources=[*list((root/'patches').glob('*.py')),Path(__file__),Path(diagnostic.__file__),Path(tracing.__file__)]
    source_hashes={p.relative_to(root).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    if '--assemble' in sys.argv:
        config=diagnostic.DiagnosticConfig('unused','unused',audit_dir=str(output))
        context=mp.get_context('spawn');queue=context.Queue()
        process=context.Process(target=child,args=(config,queue));process.start()
        assert queue.get(timeout=60)==asdict(config);process.join(60);assert process.exitcode==0;queue.close()
        result['cases'].append(dict(name='importable_spawn_configuration',passed=True))
        for strategy in ('sync','adaptive'):
            for mode,requests in (('native',1),('auto',1)):
                for capture in ('baseline','traced'):
                    case=json.loads((output/f'{strategy}_{mode}_{capture}_qualification.json').read_text())
                    assert case['passed'] and case['requests']==requests and case['source_sha256']==source_hashes
                    result['cases'].append(case)
        result['scope']='Logical ranks: single-request Native/default VAE traces, four real NFEs; unchanged math, no T4 or continuous memory-capacity claim'
        result['continuous_local_limitation']='Windows 32GB repeated-request allocation guard triggered even without Kineto; guards unchanged. Physical continuous trace quality and memory admission are independently gated.'
        result.update(passed=True,source_sha256=source_hashes);save();return
    if '--strategy' not in sys.argv:
        save()
        config=diagnostic.DiagnosticConfig('unused','unused',audit_dir=str(output))
        context=mp.get_context('spawn');queue=context.Queue()
        process=context.Process(target=child,args=(config,queue));process.start()
        assert queue.get(timeout=60)==asdict(config)
        process.join(60);assert process.exitcode==0;queue.close()
        result['cases'].append(dict(name='importable_spawn_configuration',passed=True));save()
        subprocess.run([sys.executable,'-u',str(Path(__file__).resolve()),'--make-reference'],check=True,timeout=300)
        for strategy in ('sync','adaptive'):
            for capture in ('baseline','traced'):
                subprocess.run([sys.executable,'-u',str(Path(__file__).resolve()),
                    '--strategy',strategy,'--capture',capture],check=True,timeout=240)
                result['cases'].append(json.loads((output/f'{strategy}_{capture}_qualification.json').read_text()))
        result['source_sha256']={p.relative_to(root).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [*list((root/'patches').glob('*.py')),Path(__file__),Path(diagnostic.__file__),Path(tracing.__file__)]}
        result['passed']=True;save()
        return
    import comfy_kitchen as ck
    from logical_dual_gpu import isolate_cublas_workspaces
    ck.backends.cuda._max_dynamic_shared_memory_per_block=lambda d:49152
    isolate_cublas_workspaces()
    torch.set_grad_enabled(False);torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    audit.enable('scalemax')
    from comfy_extras import nodes_minimax_h3 as nodes_h3
    original=nodes_h3.MiniMaxH3ReferenceToVideo.execute.__func__
    nodes_h3.MiniMaxH3ReferenceToVideo.execute=classmethod(
        lambda cls,clip,prompt,width,height,length,**kw:original(cls,clip,prompt,width,height,5,**kw))
    args=SimpleNamespace(width=128,height=128,steps=4,vae_mode='auto',paths={
        'te':'D:/minimaxH3/qwen3vl_32b_minimax_h3_int8_convrot.safetensors',
        'dit':'D:/minimaxH3/MiniMax_H3_Ref2VA_pruned_int8_convrot.safetensors',
        'video':'D:/minimaxH3/minimax_h3_video_vae_int8_convrot.safetensors',
        'audio':'D:/minimaxH3/minimax_h3_audio_vae_fp32_clean.safetensors'})
    import psutil
    args.stage_observer=lambda name:print('LOCAL STAGE',name,'rss',psutil.Process().memory_info().rss,
        'available',psutil.virtual_memory().available,flush=True)
    strategy=sys.argv[sys.argv.index('--strategy')+1]
    capture=sys.argv[sys.argv.index('--capture')+1]
    mode=sys.argv[sys.argv.index('--vae-mode')+1] if '--vae-mode' in sys.argv else 'auto'
    requests=int(sys.argv[sys.argv.index('--requests')+1]) if '--requests' in sys.argv else 1
    args.vae_mode=mode
    for strategy in (strategy,):
        config=diagnostic.DiagnosticConfig(args.paths['te'],args.paths['dit'],layers=2,
            weight_bytes=2*audit.GiB,te_weight_bytes=2*audit.GiB,cache_bytes=3*audit.GiB,
            host_bytes=24*audit.GiB,reserve_bytes=256*2**20,
            strategy=strategy,logical=True,audit_dir=str(output),
            cross_request_te=strategy=='adaptive',cross_model_cpu=strategy=='adaptive')
        runtime.RankRuntime=diagnostic.DiagnosticRank
        runtime.ResidencyWindow=diagnostic.PolicyWindow
        qualify.RankRuntime=diagnostic.DiagnosticRank
        recorder=tracing.Recorder(output,'logical','cuda:0')
        if '--cpu-only' in sys.argv:recorder.kineto_activities=[torch.profiler.ProfilerActivity.CPU]
        tracing.ACTIVE=recorder
        class ContextService(qualify.LogicalService):
            sequence=0
            def call(self,command,*a,**kw):
                self.sequence+=1
                if command=='denoise':recorder.context['nfe']+=1
                context=dict(recorder.context,command=self.sequence)
                kw['_diagnostic_context']=context
                return super().call(command,*a,**kw)
        service=ContextService(config)
        clip=create_clip(service);model=create_model(service);vaes={}
        try:
            local_reference=output/'native/native_reference'
            if capture=='traced':
                tracing.install(recorder);recorder.start(strategy,'--no-kineto' not in sys.argv)
            retained=[];records=[]
            for index in range(requests):
                record,traced=diagnostic.request(service,clip,model,vaes,args,recorder,index)
                retained.append(traced);records.append(record)
            if capture=='traced':recorder.stop()
            assert all(r['nfe']==4 for r in records)
            baseline_dir=output/('baseline_'+strategy+'_'+mode);baseline_dir.mkdir(exist_ok=True)
            if capture=='baseline':
                for index,values in enumerate(retained):
                    for name,tensor in values.items():torch.save(tensor,baseline_dir/f'{index}_{name}.pt')
            else:
                assert all(torch.equal(torch.load(baseline_dir/f'{index}_{k}.pt',weights_only=True),v)
                    for index,values in enumerate(retained) for k,v in values.items())
                names={e['name'] for e in recorder.events}
                for required in ('cpu_future_wait','cpu_weight_prepare','cpu_pin','h2d_enqueue',
                                 'block_acquire','stream_wait_event','reference_encode'):
                    assert required in names, (strategy,required)
                assert any(e['name']=='gpu_dependency_wait_upper_bound' for e in json.loads(
                    (output/f'{strategy}_rlogical_causal.json').read_text())['gpu_intervals'])
            checks=[diagnostic.validate(values,local_reference,output,strategy+'_'+capture+'_'+str(index),args.vae_mode)
                    for index,values in enumerate(retained)]
            case=dict(name=strategy+'_'+mode+'_'+capture,passed=True,all_outputs_bit_exact_with_uninstrumented=capture=='traced',
                native_checks=checks,trace_events=len(recorder.events),decode=record['decode'],requests=requests,
                nfe=[r['nfe'] for r in records],source_sha256=source_hashes)
            (output/f'{strategy}_{mode}_{capture}_qualification.json').write_text(json.dumps(case,indent=2))
            print('VERIFIED causal instrumentation',strategy,capture,len(recorder.events),flush=True)
        finally:
            if recorder.enabled:
                recorder.stop()
            service.close();tracing.restore()
            assert service.budget.snapshot()['reserved_bytes']==0
            vaes.clear();audit.retest.cleanup()


if __name__=='__main__':main()
