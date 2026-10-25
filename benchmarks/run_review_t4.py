"""Bounded review operator suite. No H3 model, NFE, sampling or media decode.

Use only four immutable trained block shards, seeded activation fixtures and
seeded online LoRA factors. Both TP2 owners run on distinct physical T4 GPUs.
Steady-state timings are collected without Nsight; a separate warmed replay
is attributed only after full per-worker CUDA/NVTX coverage and clean exits.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'ComfyUI'),str(ROOT/'benchmarks')]
ROWS=(261,4096,51220)
REPETITIONS=5
PRIOR_SHA='437741a15f536925563453eb7463c472a297f3f2c8c7983deed1bd78f1125ebf'
BACKENDS=('old','preallocated','tilelang_exact','cuda_int8_tilelang')


def write(path,value):
    temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(value,indent=2));temporary.replace(path)


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def host_copy(value):
    """Untimed validation storage must not occupy the candidate's GPU budget."""
    if isinstance(value,tuple):return tuple(host_copy(v) for v in value)
    if isinstance(value,list):return [host_copy(v) for v in value]
    return value.detach().to(device='cpu',copy=True)


def reference_on_host(function):
    value=function()
    result=host_copy(value)
    del value
    return result


def recovery_skipped_draws(rows=ROWS,te_rows=1182):
    """Preserve v1's RNG schedule without repeating completed operators."""
    return ([(count,5376) for _ in ('qkv_proj','fc1') for count in rows]+
            [(1,te_rows,5120)])


def fixture_plan(plan):
    """Reject accidental full-bundle loading, mutable URLs or private inputs."""
    names={f'{kind}.block00.rank{rank}.safetensors' for kind in ('dit','te') for rank in (0,1)}
    if set(plan['objects'])!=names or len(plan['revision'])!=40:
        raise ValueError('Only four immutable block-zero shards are allowed')
    if not all(c in '0123456789abcdef' for c in plan['revision']):raise ValueError('Invalid immutable revision')
    if plan.get('token_required') is not False:raise ValueError('This fixture plan must be public')
    if sum(r['bytes'] for r in plan['objects'].values())>1024**3:raise ValueError('Operator asset budget exceeded')
    return plan


def fetch_assets(assets,out):
    from huggingface_hub import hf_hub_download
    plan=fixture_plan(json.loads((ROOT/'benchmarks/review_t4_assets.json').read_text()))
    assets.mkdir(parents=True,exist_ok=True)
    def one(item):
        name,row=item;start=time.monotonic()
        path=Path(hf_hub_download(plan['repository'],name,revision=plan['revision'],token=False,local_dir=str(assets)))
        if path.stat().st_size!=row['bytes'] or sha(path)!=row['sha256']:
            raise ValueError('Immutable trained shard mismatch: '+name)
        print('REVIEW_ASSET_READY',name,flush=True)
        return dict(file=name,bytes=row['bytes'],sha256=row['sha256'],seconds=time.monotonic()-start)
    start=time.monotonic()
    with ThreadPoolExecutor(max_workers=2) as pool:objects=list(pool.map(one,plan['objects'].items()))
    write(out/'assets.json',dict(repository=plan['repository'],revision=plan['revision'],objects=objects,
        seconds=time.monotonic()-start,activation_distribution='Seeded shape fixture, not captured H3 activations',
        lora_factors='Seeded rank128, not a trained adapter',persistent_baked_weight_bytes=0))


def device_gate():
    import torch
    import comfy_kitchen as ck
    from patches.legacy_runtime import validate_packages,validate_source
    import shutil
    packages=validate_packages();validate_source(ROOT/'ComfyUI')
    devices=[dict(name=torch.cuda.get_device_name(i),capability=list(torch.cuda.get_device_capability(i)),
        uuid=str(torch.cuda.get_device_properties(i).uuid),total_bytes=torch.cuda.get_device_properties(i).total_memory)
        for i in range(torch.cuda.device_count())]
    if (len(devices)!=2 or len({d['uuid'] for d in devices})!=2 or
            any(d['capability']!=[7,5] or 'T4' not in d['name'] for d in devices)):
        raise RuntimeError('Two distinct actual T4 SM75 GPUs required; no substitute')
    if packages['comfy-kitchen']!='0.2.37' or packages['tilelang']!='0.1.13':raise RuntimeError('Core pin drift')
    if not ck.backends.cuda._EXT_AVAILABLE:raise RuntimeError('Pinned Kitchen native extension unavailable')
    nvcc=shutil.which('nvcc')
    if not nvcc:raise RuntimeError('CUDA compiler unavailable')
    compiler=subprocess.check_output([nvcc,'--version'],text=True,timeout=10)
    if 'release 12.8' not in compiler or torch.version.cuda!='12.8':
        raise RuntimeError('Pinned Kitchen/T4 CUDA 12.8 compiler ABI unqualified: '+compiler)
    nvcc=Path(nvcc).resolve();os.environ.update(CUDACXX=str(nvcc),CUDA_HOME=str(nvcc.parent.parent),CUDA_PATH=str(nvcc.parent.parent))
    return dict(devices=devices,packages=packages,torch=torch.__version__,cuda=torch.version.cuda,compiler=compiler)


def launch_workers(args,out,trace=False):
    workers=[]
    try:
        for rank in (0,1):
            log=(out/f'rank{rank}.log').open('w')
            argv=[sys.executable,'-u',str(Path(__file__)), '--output',str(out),'--assets',str(args.assets),
                  '--rank',str(rank)]
            if trace:argv+=['--trace-only']
            if args.retry_lora_only and not trace:argv+=['--retry-lora-only']
            workers.append((rank,subprocess.Popen(argv,stdout=log,stderr=subprocess.STDOUT),log))
        deadline=time.monotonic()+480
        exits=[dict(rank=rank,exit_code=proc.wait(timeout=max(1,deadline-time.monotonic()))) for rank,proc,_ in workers]
        if any(r['exit_code'] for r in exits):raise RuntimeError('Operator worker failed; inspect rank logs')
        return exits
    finally:
        for _,proc,log in workers:
            if proc.poll() is None:proc.kill();proc.wait(timeout=10)
            log.close()


def profile(args,out,state):
    from launch_dit_profile import find_tool
    from worker_trace_coverage import check
    nsys=find_tool('nsys')
    if not nsys:
        state['profiling']=dict(status='blocked',blocker='Nsight Systems unavailable; timing results remain unattributed')
        return
    path=out/'profile';path.mkdir()
    def command(argv,name,timeout):
        with (path/(name+'.log')).open('w') as log:
            result=subprocess.run(argv,stdout=log,stderr=subprocess.STDOUT,timeout=timeout)
        if result.returncode:raise RuntimeError(name+' failed; no profiler attribution')
    try:
        command([sys.executable,str(ROOT/'benchmarks/qualify_worker_cuda_trace.py'),'--nsys',nsys,
            '--output',str(path/'minimal_gate'),'--checkpoint-restart'],'minimal_gate',120)
        gate=json.loads((path/'minimal_gate/capture_gate.json').read_text())
        if not gate['passed']:raise RuntimeError('Minimal worker trace capture failed')
        command([nsys,'profile','--trace=cuda,nvtx,osrt','--sample=none','--cpuctxsw=none',
            '--trace-fork-before-exec=true','--flush-on-cudaprofilerstop=true','--wait=all',
            '-o',str(path/'operators'),sys.executable,'-u',str(Path(__file__)),
            '--output',str(path/'replay'),'--assets',str(args.assets),'--trace-launch'], 'capture',480)
        command([nsys,'export','--type=sqlite','--output',str(path/'operators.sqlite'),
            str(path/'operators.nsys-rep')],'export',60)
        coverage=check(path/'operators.sqlite',prefix='H3_REVIEW/',minimum=8,minimum_per_label=1)
        replay=json.loads((path/'replay/result.json').read_text())
        workers=[json.loads((path/'replay'/f'rank{r}.json').read_text()) for r in (0,1)]
        complete=(coverage['passed'] and replay['status']=='trace_workers_complete' and
            all(w['status']=='complete' and w['trace_labels']==expected_labels() for w in workers) and
            all(r['exit_code']==0 for r in replay['worker_exits']) and
            all(json.loads((path/'replay'/f'rank{r}_cuda_checkpoints.json').read_text())['records'][-1]['stopped'] for r in (0,1)))
        state['profiling']=dict(status='complete' if complete else 'blocked',coverage=coverage,
            clean_exit=replay['worker_exits'],counter_permission='Prior T4 ERR_NVGPUCTRPERM; NCU not reprobed')
        if not complete:raise RuntimeError('Incomplete operator labels / worker CUDA coverage; no attribution')
        command([nsys,'stats','--report','cuda_gpu_kern_sum,cuda_api_sum,cuda_gpu_mem_time_sum,nvtx_sum',
            '--format','csv','--output',str(path/'operator_stats'),str(path/'operators.sqlite')],'stats',60)
    except Exception as error:
        state.setdefault('profiling',{}).update(status='blocked',blocker=str(error))
        (path/'exception.txt').write_text(traceback.format_exc())


def expected_labels():
    return ([f'column/{projection}/{backend}' for projection in ('qkv_proj','fc1') for backend in BACKENDS]+
        [f'te/{group}/{backend}' for group in ('qkv','gate_up') for backend in ('independent','shared')]+
        ['qkv_lora/full','qkv_lora/owned'])


def worker(args):
    import torch
    import comfy_kitchen as ck
    from safetensors.torch import load_file
    from patches.dit_column_linear import _eager_column,column_linear,shared_column_linears
    from patches.h3_lora import NativeAlignedLoRA
    from patches.tilelang_column_epilogue import dequantize
    torch.cuda.set_device(args.rank);torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.manual_seed(20261024);torch.cuda.manual_seed(20261024)
    out=args.output;out.mkdir(parents=True,exist_ok=True)
    record=dict(status='running',rank=args.rank,physical_device=torch.cuda.get_device_name(args.rank),
        capability=list(torch.cuda.get_device_capability(args.rank)),pid=os.getpid(),cases=[],trace_labels=[],
        warmups=3,repetitions=REPETITIONS,fixture_distribution='Seeded FP16 inputs; no captured distribution claim')
    record.update(retry_lora_only=args.retry_lora_only,correctness_reference_device='cpu',
        prior_input_identity='Same seed and scripted draw order; v1 did not export input hashes')
    path=out/f'rank{args.rank}.json'
    def save():write(path,record)
    def exact(a,b):
        if isinstance(a,(tuple,list)):return len(a)==len(b) and all(exact(x,y) for x,y in zip(a,b))
        return a.shape==b.shape and a.dtype==b.dtype and torch.equal(a.contiguous().view(torch.uint8),b.contiguous().view(torch.uint8))
    def finite(value):
        return all(finite(v) for v in value) if isinstance(value,(tuple,list)) else bool(torch.isfinite(value).all())
    def compare_case(label,functions,reference_name,details):
        case=dict(label=label,**details,backends={});record['cases'].append(case);save()
        reference=None if args.trace_only else reference_on_host(functions[reference_name])
        if reference is not None and not finite(reference):raise AssertionError('Nonfinite fixture reference: '+label)
        # V1 retained a 1.026-GiB LoRA oracle during full-B warm-up, then
        # fragmented the available allocator segments. Release it before any
        # repeat; CPU oracle copies and cache cleanup are outside timing.
        torch.cuda.synchronize();torch.cuda.empty_cache()
        for name,fn in functions.items():
            started=time.monotonic();output=fn();torch.cuda.synchronize()
            case['backends'][name]=dict(first_call_seconds=time.monotonic()-started,measurements=[])
            if reference is not None:
                equal=exact(host_copy(output),reference);case['backends'][name]['byte_exact']=equal
                if not equal:save();raise AssertionError('Raw storage mismatch: '+label+'/'+name)
            del output
            for _ in range(2):output=fn();torch.cuda.synchronize();del output
        # The large reference output is deliberately released before measuring
        # memory, so reported incremental peaks do not include validation data.
        del reference
        torch.cuda.synchronize();torch.cuda.empty_cache()
        if args.trace_only:
            for name,fn in functions.items():
                full=label+'/'+name;torch.cuda.nvtx.range_push('H3_REVIEW/'+full)
                try:output=fn();torch.cuda.synchronize();del output
                finally:torch.cuda.nvtx.range_pop()
                record['trace_labels'].append(full);save()
            return
        for repetition in range(REPETITIONS):
            names=list(functions)
            if repetition%2:names.reverse()
            for name in names:
                torch.cuda.synchronize();base=torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats()
                begin=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
                started=time.perf_counter();begin.record();output=functions[name]();end.record();end.synchronize()
                case['backends'][name]['measurements'].append(dict(repetition=repetition,
                    cuda_ms=begin.elapsed_time(end),host_seconds=time.perf_counter()-started,
                    baseline_allocated_bytes=base,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                    incremental_peak_bytes=torch.cuda.max_memory_allocated()-base,
                    reserved_bytes=torch.cuda.memory_reserved()))
                del output;save()
        for value in case['backends'].values():
            value['median_cuda_ms']=statistics.median(r['cuda_ms'] for r in value['measurements'])
            value['median_host_seconds']=statistics.median(r['host_seconds'] for r in value['measurements'])
            value['max_incremental_peak_bytes']=max(r['incremental_peak_bytes'] for r in value['measurements'])
        print('REVIEW_CASE_COMPLETE',args.rank,label,details,flush=True);save()
    try:
        prior_path=ROOT/'benchmarks/review_prior_column.py'
        if sha(prior_path)!=PRIOR_SHA:raise ValueError('Pre-edit baseline source mismatch')
        spec=importlib.util.spec_from_file_location('prior_column',prior_path)
        prior=importlib.util.module_from_spec(spec);spec.loader.exec_module(prior)
        with torch.no_grad(),ck.registry.use_backend('eager'):
            if not args.trace_only and not args.retry_lora_only:
                values=torch.tensor([-2147483647,-16777217,-3,0,3,16777217,2147483647],device='cuda',dtype=torch.int32)
                acc=values.repeat(17,1);a=torch.logspace(-8,2,17,device='cuda').reshape(-1,1)
                record['epilogue']=[]
                for scalar in (True,False):
                    w=torch.tensor([.314159],device='cuda') if scalar else torch.linspace(.01,.9,7,device='cuda')
                    for dtype in (torch.float16,torch.float32):
                        same=exact((acc.float()*(a*w)).to(dtype),dequantize(acc,a,w,dtype))
                        record['epilogue'].append(dict(scalar=scalar,dtype=str(dtype),byte_exact=same))
                        if not same:raise AssertionError('Ordered column epilogue rounding failed')
                del acc,a,w,values
            state={} if args.retry_lora_only else load_file(str(args.assets/f'dit.block00.rank{args.rank}.safetensors'))
            for projection in (() if args.retry_lora_only else ('qkv_proj','fc1')):
                weight=state[projection+'.weight'].cuda();scale=state[projection+'.weight_scale'].cuda()
                if tuple(weight.shape)!=(10752 if projection=='qkv_proj' else 14336,5376):raise AssertionError('TP2 column shape changed')
                options=dict(convrot=True,convrot_groupsize=256,out_dtype=torch.float16)
                for count in ((51220,) if args.trace_only else ROWS):
                    x=torch.randn(count,5376,device='cuda',dtype=torch.float16)*.2
                    functions={'old':lambda:prior.column_linear(x,weight,scale,**options)}
                    functions.update({n:(lambda n=n:column_linear(x,weight,scale,backend=n,**options)) for n in BACKENDS[1:]})
                    probe_seconds={}
                    if not args.trace_only:
                        # Confirm the public SM75 dispatcher really enters each
                        # selected route; do not benchmark silent fallbacks.
                        from unittest.mock import patch
                        import patches.dit_column_linear as module
                        for backend in BACKENDS[1:]:
                            with patch.object(module,'_eager_column',wraps=_eager_column) as called:
                                started=time.monotonic();probe=functions[backend]();torch.cuda.synchronize();del probe
                                probe_seconds[backend]=time.monotonic()-started
                                if called.call_args is None or called.call_args.kwargs['backend']!=backend:
                                    raise AssertionError('Production column dispatcher fell back: '+backend)
                    compare_case('column/'+projection,functions,'old',dict(rows=count,shape=[count,weight.shape[0],5376],
                        dispatch='public, pinned eager',first_dispatch_including_compilation_seconds=probe_seconds))
                    del x,functions;torch.cuda.empty_cache()
                del weight,scale
            del state;torch.cuda.empty_cache()
            state={} if args.retry_lora_only else load_file(str(args.assets/f'te.block00.rank{args.rank}.safetensors'))
            if not args.retry_lora_only:x=torch.randn(1,1182,5120,device='cuda',dtype=torch.float16)*.2
            for group,columns in ([] if args.retry_lora_only else [('qkv',('q_proj','k_proj','v_proj')),('gate_up',('gate_proj','up_proj'))]):
                weights=[(state[n+'.weight'].cuda(),state[n+'.weight_scale'].cuda()) for n in columns]
                def independent():
                    return tuple(ck.int8_linear(x.view(-1,5120),w,s,None,x.dtype,convrot=True,convrot_groupsize=256)
                                 .view(1,1182,w.shape[0]) for w,s in weights)
                compare_case('te/'+group,dict(independent=independent,shared=lambda:shared_column_linears(x,weights)),
                    'independent',dict(rows=1182,widths=[w.shape[0] for w,s in weights],preparation_calls=[len(weights),1]))
                del weights;torch.cuda.empty_cache()
            del state
            if not args.retry_lora_only:del x
            else:
                # These are RNG draws only, with no weight loads, GEMMs,
                # correctness repeats or timing of the already completed cases.
                for shape in recovery_skipped_draws():
                    discarded=torch.randn(*shape,device='cuda',dtype=torch.float16)
                    del discarded
            torch.cuda.empty_cache()
            terms=[(torch.randn(128,5376)*.02,torch.randn(21504,128)*.01,s) for s in (.7,-.25)]
            rows=[(c*7168+args.rank*3584,c*7168+(args.rank+1)*3584) for c in range(3)]
            full=NativeAlignedLoRA(terms,rows=rows).cuda();full.fp32_compute=True
            selected=NativeAlignedLoRA([(a,torch.cat([b[lo:hi] for lo,hi in rows]),s) for a,b,s in terms]).cuda()
            selected.fp32_compute=True
            for count in ((51220,) if args.trace_only else ROWS):
                x=torch.randn(count,5376,device='cuda',dtype=torch.float16)*.2
                base=torch.randn(count,10752,device='cuda',dtype=torch.float16)*.2
                if args.retry_lora_only and count!=51220:
                    del x,base;continue
                record.setdefault('lora_inputs',[]).append(dict(rows=count,
                    x_sha256=hashlib.sha256(memoryview(x.cpu().view(torch.uint8).numpy()).cast('B')).hexdigest(),
                    base_sha256=hashlib.sha256(memoryview(base.cpu().view(torch.uint8).numpy()).cast('B')).hexdigest()))
                compare_case('qkv_lora',dict(full=lambda:full.add_to(x,base.clone()),owned=lambda:selected.add_to(x,base.clone())),
                    'full',dict(rows=count,lora_rank=128,adapters=2,fp32_compute=True,base_clone_included=True))
                del x,base;torch.cuda.empty_cache()
        if args.trace_only:
            from worker_cuda_checkpoint import CUDAFlushCheckpoint
            CUDAFlushCheckpoint(args.rank,out).flush('worker_close',resume=False)
        record.update(status='complete',all_outputs_byte_exact=None if args.trace_only else
            all(v.get('byte_exact',False) for c in record['cases'] for v in c['backends'].values()))
    except BaseException as error:
        record.update(status='failed',error=repr(error));(out/f'rank{args.rank}_exception.txt').write_text(traceback.format_exc());raise
    finally:save()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--assets',type=Path,required=True);parser.add_argument('--rank',type=int,choices=(0,1))
    parser.add_argument('--trace-only',action='store_true');parser.add_argument('--trace-launch',action='store_true')
    parser.add_argument('--retry-lora-only',action='store_true',help='Retry only the unfinished 51220-row LoRA case; retain v1 completed timings')
    args=parser.parse_args()
    if args.retry_lora_only and (args.trace_only or args.trace_launch):parser.error('Recovery timing scope is distinct from complete trace replay')
    if args.rank is not None:return worker(args)
    out=args.output;out.mkdir(parents=True,exist_ok=True)
    state=dict(status='starting',scope='Operator-only, both actual TP2 owners; no H3 pipeline',
        excluded=['Offline LoRA cross-rank full-matrix CPU deduplication','Scalar Sparse kernels'],
        pending=['Full learned Video VAE remote decode and historical 261-token projection discrepancy',
                 'Generation distribution and semantic acceptance'],commands=[])
    state['retry_lora_only']=args.retry_lora_only
    def save():write(out/'result.json',state)
    try:
        if args.trace_launch:
            state['worker_exits']=launch_workers(args,out,trace=True)
            state['status']='trace_workers_complete';return
        with (out/'batch.claimed').open('x') as claim:claim.write('One bounded execution; do not rerun after failure')
        state['hardware']=device_gate();save()
        fetch_assets(args.assets,out);state['status']='assets_ready';save()
        started=time.monotonic();state['worker_exits']=launch_workers(args,out)
        state['operator_seconds']=time.monotonic()-started
        state['ranks']=[json.loads((out/f'rank{rank}.json').read_text()) for rank in (0,1)]
        state['all_outputs_byte_exact']=all(r['all_outputs_byte_exact'] for r in state['ranks'])
        state['status']='operators_complete';save()
        profile(args,out,state)
        state['status']='complete' if state['profiling']['status']=='complete' else 'operators_complete_profiler_blocked'
    except BaseException as error:
        state.update(status='failed',error=repr(error));(out/'exception.txt').write_text(traceback.format_exc());raise
    finally:save();print('REVIEW_FINAL',state['status'],flush=True)


if __name__=='__main__':main()
