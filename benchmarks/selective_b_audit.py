"""Benchmark-only Selective B feasibility. No production/UI mutation."""
import argparse
from datetime import timedelta
import gc,gzip,hashlib,json,math,os
from pathlib import Path
import sys,time
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),os.environ.get('COMFY_ROOT',str(ROOT/'ComfyUI')),str(ROOT/'benchmarks')]
if '--physical' not in sys.argv:import simulate_weight_residency
import torch
import torch.nn.functional as F
import torch.distributed as dist
import retest
from logical_dual_gpu import LogicalDualGPU,isolate_cublas_workspaces
from physical_h3_reduction import timed,ProfileComm
from patches.windowed_tp import CheckpointReader,prepare_tp_block
from patches.h3_lora import H3LoRAStore
from patches.optimized_tp import enable,disable,global_row_quantize
from selective_b_prototype import SelectiveBFC1

PROTECTED=('patches/h3_lora.py','patches/optimized_tp.py','patches/tp_dit.py','patches/comfy_tp_runtime.py',
           'custom_nodes/minimax_h3_tp2/__init__.py','patches/windowed_tp.py','patches/rank_local_int8.py')
SOURCES=PROTECTED+('benchmarks/selective_b_prototype.py','benchmarks/selective_b_audit.py')


def byte_equal(a,b):return torch.equal(a.contiguous().view(torch.uint8),b.contiguous().view(torch.uint8))


def metrics(a,b):
    finite=bool(torch.isfinite(a).all()) and bool(torch.isfinite(b).all())
    exact=byte_equal(a,b)
    if exact:return dict(finite=finite,bytes_exact=True,max_abs=0.,rel_l2=0.,cosine=1.)
    a=a.flatten();b=b.flatten();aa=bb=ab=dd=0.;maximum=0.
    for start in range(0,a.numel(),4*2**20):
        x=a[start:start+4*2**20].double();y=b[start:start+4*2**20].double();delta=x-y
        aa+=float(x.square().sum());bb+=float(y.square().sum());ab+=float((x*y).sum());dd+=float(delta.square().sum());maximum=max(maximum,float(delta.abs().max()))
    return dict(finite=finite,bytes_exact=False,max_abs=maximum,rel_l2=math.sqrt(dd/max(bb,1e-300)),cosine=ab/max(math.sqrt(aa*bb),1e-300))


def profile_pass(fn,output,label,rank):
    dist.barrier();torch.cuda.synchronize()
    # Shapes are diagnostic only; all performance/peak measurements are separate.
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],
        record_shapes=True,profile_memory=False,with_stack=False) as prof:
        with torch.profiler.record_function(label):fn()
        torch.cuda.synchronize()
    path=output/f'{label}_rank{rank}.json';prof.export_chrome_trace(str(path))
    with path.open('rb') as src,gzip.open(str(path)+'.gz','wb',compresslevel=5) as dst:
        import shutil;shutil.copyfileobj(src,dst)
    path.unlink();return str(path.name)+'.gz'


def main():
    p=argparse.ArgumentParser();p.add_argument('--physical',action='store_true');p.add_argument('--profile',action='store_true')
    p.add_argument('--checkpoint',default='D:/minimaxH3/MiniMax_H3_Ref2VA_pruned_int8_convrot.safetensors')
    p.add_argument('--lora',default='D:/minimaxH3/loras/dmad_minimax_h3_4step_full_critic_comfyui.safetensors')
    p.add_argument('--output',type=Path,required=True);p.add_argument('--indices',type=int,nargs='+',default=[0,25,49]);args=p.parse_args()
    import comfy_kitchen as ck
    import patches.tp_dit as dit
    rank=int(os.environ['LOCAL_RANK']) if args.physical else 0
    torch.set_grad_enabled(False);torch.backends.cuda.matmul.allow_tf32=False;torch.set_num_threads(2)
    if args.physical:
        torch.cuda.set_device(rank);dist.init_process_group('nccl',timeout=timedelta(seconds=180))
        assert dist.get_world_size()==2 and all('T4' in torch.cuda.get_device_name(i) for i in (0,1))
    else:
        ck.backends.cuda._max_dynamic_shared_memory_per_block=lambda _:49152;isolate_cublas_workspaces()
    saved=dit.dist;enable('scalemax');args.output.mkdir(parents=True,exist_ok=True)
    source_hashes={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in SOURCES}
    checkpoint_signatures={p:(Path(p).stat().st_size,Path(p).stat().st_mtime_ns) for p in (args.checkpoint,args.lora)}
    state=dict(status='running',physical_dual_t4=args.physical,torch=torch.__version__,cuda=torch.version.cuda,
        scope='FC1-only Selective B prototype against validated Row Partial; no deployment or production changes',
        stack_note='Multi uses two separate entries of the same real DMAD checkpoint, +0.6/-0.2; no second independent checkpoint available',
        numerical_gate='Finite outputs, owner mapping, replica agreement and relative L2 < .01 gross-regression guard; byte-exactness recorded separately, not assumed',
        source_sha256=source_hashes,cases=[],timings=[],profiles=[],stage_memory=[],packing=[])
    def save():
        if rank==0:retest.write_json(args.output/'results.json',state)
    def gather(value):
        if not args.physical:return value
        values=[None,None];dist.all_gather_object(values,value);return values
    def execute(blocks,fn):
        if args.physical:dit.dist=dist;return [fn(blocks[0])]
        comm=LogicalDualGPU(180);dit.dist=comm
        return comm.run([lambda r=r:fn(blocks[r]) for r in (0,1)])
    def choose(blocks,adapters):
        for block,adapter in zip(blocks,adapters):block.fc1.lora_adapter=adapter
    def release():
        gc.collect();torch.cuda.empty_cache()
        if args.physical:dist.barrier()
    def compare_list(actual,reference):
        rows=[metrics(a,b) for a,b in zip(actual,reference)]
        return gather(rows[0]) if args.physical else rows
    def replicas(values):
        hashes=[hashlib.sha256(memoryview(v.cpu().numpy()).cast('B')).hexdigest() for v in values]
        if args.physical:hashes=gather(hashes[0])
        return len(set(hashes))==1
    save()
    try:
        for strengths in ((1.,),(.6,-.2)):
            stack='single' if len(strengths)==1 else 'multi';base=CheckpointReader(args.checkpoint)
            store=H3LoRAStore(tuple((args.lora,s) for s in strengths),base,row_partial=True);base.lora_store=store;identity=store.identity
            for index in args.indices:
                ranks=(rank,) if args.physical else (0,1)
                blocks=[prepare_tp_block(base,'dit',index,r).cuda() for r in ranks]
                original=[b.fc1.lora_adapter for b in blocks]
                torch.cuda.synchronize();a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True)
                a.record();start=time.perf_counter();selective=[SelectiveBFC1(adapter,r) for adapter,r in zip(original,ranks)];b.record();b.synchronize()
                packing=dict(stack=stack,block=index,cuda_ms=a.elapsed_time(b),wall_ms=(time.perf_counter()-start)*1000,
                    per_rank_packed_weight_bytes=[s.pack_bytes for s in selective],copies_per_rank=[s.pack_copies for s in selective])
                state['packing'].append(dict(per_rank=gather(packing) if args.physical else [packing]))
                assert store.identity==identity
                for count in (513,8192):
                    # Other blocks qualify real learned values; time/profile block0 only.
                    torch.manual_seed(900);x=torch.randn(count,5376,device='cuda')
                    hidden=[F.linear(x,adapter.terms[0].down) for adapter in original]
                    choose(blocks,original);baseline_fc1=execute(blocks,lambda block:block.fc1(x))
                    baseline_ffn=execute(blocks,lambda block:block.fc2(block.fc1(x),input_act='swiglu'))
                    baseline_q=execute(blocks,lambda block:global_row_quantize(baseline_fc1[0 if args.physical else dit.dist.get_rank()],dit.dist,2,256,'swiglu'))
                    choose(blocks,selective);candidate_fc1=execute(blocks,lambda block:block.fc1(x))
                    candidate_ffn=execute(blocks,lambda block:block.fc2(block.fc1(x),input_act='swiglu'))
                    candidate_q=execute(blocks,lambda block:global_row_quantize(candidate_fc1[0 if args.physical else dit.dist.get_rank()],dit.dist,2,256,'swiglu'))
                    fc1_metrics=compare_list(candidate_fc1,baseline_fc1);ffn_metrics=compare_list(candidate_ffn,baseline_ffn)
                    qmetrics=[];bmetrics=[]
                    for offset in range(len(blocks)):
                        q,s=candidate_q[offset];qr,sr=baseline_q[offset]
                        qmetrics.append(dict(changed_payload=int((q!=qr).sum()),payload_elements=q.numel(),
                            changed_scale_rows=int((s.view(torch.uint8).reshape(count,-1)!=sr.view(torch.uint8).reshape(count,-1)).any(-1).sum()),scale=metrics(s,sr)))
                        delta=F.linear(hidden[offset],original[offset].terms[0].up)
                        picked=torch.cat([delta[:,lo:hi] for lo,hi in original[offset].rows],-1);del delta
                        direct=F.linear(hidden[offset],selective[offset].terms[0].up)
                        bmetrics.append(metrics(direct,picked));del picked,direct
                    # Do not retain quantizer diagnostic tensors in timing residency.
                    del q,s,qr,sr
                    if args.physical:qmetrics=gather(qmetrics[0]);bmetrics=gather(bmetrics[0])
                    row=dict(stack=stack,block=index,rows=count,fc1_per_rank=fc1_metrics,ffn_per_rank=ffn_metrics,
                        b_gemm_per_rank=bmetrics,downstream_quant_per_rank=qmetrics,
                        baseline_replicas_agree=replicas(baseline_ffn),selective_replicas_agree=replicas(candidate_ffn),
                        column_rank_rule='Different owned features; each rank compared to its own baseline, no cross-rank FC1 equality expected',
                        B_baseline=[28672,128],B_selective=[14336,128],A=[128,5376])
                    state['cases'].append(row);save()
                    all_scores=fc1_metrics+ffn_metrics+bmetrics
                    if not row['baseline_replicas_agree'] or not row['selective_replicas_agree'] or any(not m['finite'] or m['rel_l2']>=.01 for m in all_scores):
                        raise RuntimeError('Feasibility sanity/ownership gate failed; reject performance')
                    del baseline_fc1,baseline_ffn,candidate_fc1,candidate_ffn,baseline_q,candidate_q;release()
                    if args.physical and index==0:
                        for turn in (0,1):
                            for variant in (('baseline','selective') if turn==0 else ('selective','baseline')):
                                adapters=original if variant=='baseline' else selective;choose(blocks,adapters)
                                for scope in ('fc1','ffn','b_gemm'):
                                    if scope=='fc1':fn=lambda:blocks[0].fc1(x)
                                    elif scope=='ffn':fn=lambda:blocks[0].fc2(blocks[0].fc1(x),input_act='swiglu')
                                    else:fn=lambda:F.linear(hidden[0],adapters[0].terms[0].up)
                                    torch.cuda.empty_cache();timing=timed(fn,warmups=2,repeats=5)
                                    state['timings'].append(dict(stack=stack,rows=count,variant=variant,round=turn,scope=scope,timing=timing));save()
                        if count==8192 and stack=='multi':
                            for variant,adapters in (('baseline',original),('selective',selective)):
                                choose(blocks,adapters);torch.cuda.synchronize();entry=torch.cuda.memory_allocated()
                                torch.cuda.reset_peak_memory_stats();packed=blocks[0].fc1(x);torch.cuda.synchronize()
                                fc1_peak=torch.cuda.max_memory_allocated()-entry
                                torch.cuda.reset_peak_memory_stats();out=blocks[0].fc2(packed,input_act='swiglu');torch.cuda.synchronize()
                                fc2_peak=torch.cuda.max_memory_allocated()-entry;del out,packed
                                state['stage_memory'].append(dict(variant=variant,per_rank=gather(dict(fc1_peak_above_entry=fc1_peak,fc2_peak_above_entry=fc2_peak))))
                                if args.profile:
                                    class Comm(ProfileComm):pass
                                    transport=Comm();dit.dist=transport
                                    def traced_ffn():
                                        with torch.profiler.record_function('stage/FC1'):packed=blocks[0].fc1(x)
                                        with torch.profiler.record_function('stage/FC2'):return blocks[0].fc2(packed,input_act='swiglu')
                                    for scope,fn in (('b_gemm',lambda:F.linear(hidden[0],adapters[0].terms[0].up)),('fc1',lambda:blocks[0].fc1(x)),('ffn',traced_ffn)):
                                        label=f'SelectiveB_{variant}_{scope}';trace=profile_pass(fn,args.output,label,rank)
                                        state['profiles'].append(dict(variant=variant,scope=scope,rank=rank,trace=trace,collectives=transport.calls.copy()))
                                        transport.calls.clear()
                                    dit.dist=dist
                                save()
                    print('SELECTIVE_B_CASE',stack,index,count,'fc1_exact',all(m['bytes_exact'] for m in fc1_metrics),'ffn_exact',all(m['bytes_exact'] for m in ffn_metrics),flush=True)
                    del x,hidden;release()
                del blocks,original,selective;release()
            del store,base;release()
        protected_unchanged=all(hashlib.sha256((ROOT/n).read_bytes()).hexdigest()==source_hashes[n] for n in PROTECTED)
        checkpoint_unchanged=all((Path(p).stat().st_size,Path(p).stat().st_mtime_ns)==sig for p,sig in checkpoint_signatures.items())
        if not protected_unchanged or not checkpoint_unchanged:raise RuntimeError('Protected source or checkpoint changed during study')
        state.update(status='complete',passed=True,production_unchanged=True,checkpoint_signatures_unchanged=True,
            fc1_byte_exact=all(m['bytes_exact'] for c in state['cases'] for m in c['fc1_per_rank']),
            ffn_byte_exact=all(m['bytes_exact'] for c in state['cases'] for m in c['ffn_per_rank']))
        save()
    except BaseException as error:state.update(status='failed',passed=False,error_type=type(error).__name__);save();raise
    finally:
        dit.dist=saved;disable()
        if args.physical:torch.cuda.synchronize();dist.destroy_process_group()


if __name__=='__main__':main()
