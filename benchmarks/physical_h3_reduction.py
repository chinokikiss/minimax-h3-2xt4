"""Physical dual-T4 qualification; learned H3 FFN and complete block diagnostics.

Results are block/projection measurements, never full-pipeline estimates.
Failed numerical gates keep raw diagnostic timings but never earn speedups.
"""
import argparse
from datetime import timedelta
import gc
import gzip
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),os.environ.get('COMFY_ROOT',str(ROOT/'ComfyUI')),str(ROOT/'benchmarks')]
import torch
import torch.distributed as dist
from torch import nn
import torch.nn.functional as F
import comfy_kitchen as ck
import retest
from fixture_inputs import block_input
from patches.windowed_tp import CheckpointReader,prepare_tp_block
from patches.h3_lora import H3LoRAStore,NativeLoraLinear,NativeLoraMLP
from patches.optimized_tp import enable
import patches.optimized_tp as tp
from patches.parallel_int8 import quantize,dequantize
from patches.quantized_collectives import int8_sum

MODES=[('scalemax','native_int32'),('scalemax_streamed','native_int32'),
       ('scalemax_streamed','fp32'),('scalemax_streamed','int8')]
GATE=dict(rel_l2_max=1e-5,cosine_min=.999999,max_abs_max=.005)


def agree(value):
    t=torch.tensor(int(value),device='cuda');dist.all_reduce(t,op=dist.ReduceOp.MIN);return bool(t.item())


def score(value,reference):
    m=retest.metrics(value,reference)
    # Degenerate zero inputs are valid for explicit transport safety tests.
    m['passed']=retest.passes(m,GATE) if m.get('valid') else bool(m.get('is_bit_exact'))
    return m


class ProfileComm:
    ReduceOp=dist.ReduceOp
    def __init__(self):self.calls=[]
    def get_rank(self):return dist.get_rank()
    def get_world_size(self):return dist.get_world_size()
    def is_initialized(self):return True
    def all_gather(self,*args):return dist.all_gather(*args)
    def all_reduce(self,value,op=None):
        label='MAX' if op==dist.ReduceOp.MAX else 'SUM'
        self.calls.append(dict(op=label,dtype=str(value.dtype),elements=value.numel(),payload_bytes=value.numel()*value.element_size()))
        with torch.profiler.record_function(f'collective/{label}/{value.dtype}'):
            return dist.all_reduce(value,op=op)


def timed(fn,warmups=2,repeats=5):
    for _ in range(warmups):fn()
    torch.cuda.synchronize();dist.barrier();torch.cuda.synchronize()
    start_bytes=torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats()
    walls=[];events=[]
    for _ in range(repeats):
        a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True)
        a.record();start=time.perf_counter();result=fn();b.record();b.synchronize()
        walls.append((time.perf_counter()-start)*1000);events.append(a.elapsed_time(b));del result
    result=dict(wall_ms=walls,cuda_event_ms=events,wall_median_ms=statistics.median(walls),
        cuda_median_ms=statistics.median(events),peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        peak_reserved_bytes=torch.cuda.max_memory_reserved(),baseline_allocated_bytes=start_bytes,
        temporary_peak_bytes=torch.cuda.max_memory_allocated()-start_bytes,warmups=warmups,repeats=repeats)
    gathered=[None,None];dist.all_gather_object(gathered,result)
    return dict(per_rank=gathered,critical_wall_median_ms=statistics.median([max(gathered[r]['wall_ms'][i] for r in (0,1)) for i in range(repeats)]),
                critical_cuda_median_ms=statistics.median([max(gathered[r]['cuda_event_ms'][i] for r in (0,1)) for i in range(repeats)]))


def profile(fn,output,label):
    dist.barrier();torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],
                                record_shapes=False,profile_memory=False,with_stack=False) as prof:
        with torch.profiler.record_function(label):fn()
        torch.cuda.synchronize()
    rank=dist.get_rank();path=output/f'{label}_rank{rank}.json'
    prof.export_chrome_trace(str(path))
    with path.open('rb') as source,gzip.open(str(path)+'.gz','wb',compresslevel=5) as dest:
        import shutil;shutil.copyfileobj(source,dest)
    path.unlink()
    # Intervals overlap; self GPU sums are activity totals, not critical path.
    top=sorted(prof.key_averages(),key=lambda e:e.self_device_time_total,reverse=True)
    rows=[dict(name=e.key,calls=e.count,self_cpu_us=e.self_cpu_time_total,
               self_gpu_us=e.self_device_time_total,total_gpu_us=e.device_time_total) for e in top[:45]]
    (output/f'{label}_rank{rank}_summary.json').write_text(json.dumps(rows,indent=2))
    return str(path.name)+'.gz'


class SequentialNative(nn.Module):
    def __init__(self,terms):
        super().__init__()
        from comfy.weight_adapter.lora import LoRAAdapter
        self.adapters=[]
        for a,b,scale in terms:
            adapter=LoRAAdapter(set(),(b.cuda().float(),a.cuda().float(),None,None,None,None))
            adapter.multiplier=scale;self.adapters.append(adapter)
    def add_to(self,x,out,input_act=None,**kwargs):
        if input_act=='swiglu':
            gate,up=x.chunk(2,-1);x=F.silu(gate)*up
        for adapter in self.adapters:out=out+adapter.h(x,out)
        return out


def native_adapt(block,store):
    for n in ('qkv_proj','out_proj'):
        setattr(block.attn,n,NativeLoraLinear(getattr(block.attn,n),SequentialNative(store.terms('blocks.0.attn.'+n))))
    block.mlp=NativeLoraMLP(block.mlp,SequentialNative(store.terms('blocks.0.mlp.fc1')),
                          SequentialNative(store.terms('blocks.0.mlp.fc2')))
    return block


def block_diagnostics(base,lora,output,save):
    rank=dist.get_rank();rows=[]
    curve=base.read_tensor('adaln_t_table')
    merged=output/'offline_block0.safetensors'
    offline_bytes=[]
    if rank==0:
        from merge_h3_lora import convert
        from comfy_kitchen.tensor.int8 import TensorWiseINT8Layout as Layout
        convert(str(base.path),((str(lora),1.),),merged)
        merged_reader=CheckpointReader(merged);single=H3LoRAStore(((str(lora),1.),),base)
        for target in sorted(single.targets):
            q=base.read_tensor(target+'.weight').cuda();s=base.read_tensor(target+'.weight_scale').cuda()
            p=Layout.Params(scale=s,orig_dtype=torch.float32,orig_shape=tuple(q.shape),is_weight=True,convrot=True,convrot_groupsize=256)
            value=Layout.dequantize(q,p)
            for a,b,scale in single.terms(target):value.add_(torch.mm(b.cuda().float(),a.cuda().float())*scale)
            expected,params=Layout.quantize(value,per_channel=True,convrot=True,convrot_groupsize=256,stochastic_rounding=0)
            exact=torch.equal(expected.cpu(),merged_reader.read_tensor(target+'.weight')) and torch.equal(params.scale.cpu(),merged_reader.read_tensor(target+'.weight_scale'))
            offline_bytes.append(dict(target=target,native_bytes_exact=exact))
            del q,s,p,value,expected,params;torch.cuda.empty_cache()
        save(offline_native_bytes=offline_bytes)
        if not all(r['native_bytes_exact'] for r in offline_bytes):raise RuntimeError('Physical offline Native bytes differ')
    dist.barrier()
    variants=[('base',base.path,()),('online_single',base.path,(1.,)),('online_multi',base.path,(.6,-.2)),('offline',merged,())]
    for adapter_mode,checkpoint,strengths in variants:
        store=H3LoRAStore(tuple((str(lora),s) for s in strengths),base) if strengths else None
        reader=CheckpointReader(checkpoint)
        if store is not None:reader.lora_store=store
        block=prepare_tp_block(reader,'dit',0,rank).cuda()
        native=retest.load_native_block('dit',checkpoint)
        if store is not None:native=native_adapt(native,store)
        for count in (33,257,512):
            data=block_input('dit',count,900,curve);x=data['input'].cuda()
            args=(data['t_emb'].cuda(),data['segments'],data['rope'].cuda())
            reference=native(x.clone(),*args).cpu();torch.cuda.synchronize()
            for mode,reduction in MODES:
                enable(mode,swiglu_reduction=reduction)
                actual=block(x.clone(),*args)
                m=score(actual.cpu(),reference);m['passed']=agree(m['passed'])
                replica=actual.clone();dist.broadcast(replica,src=0)
                m['replicas_equal']=agree(torch.equal(replica,actual));del replica,actual
                item=dict(scope='complete_DiT_block0',rows=count,adapter_mode=adapter_mode,lora_strengths=list(strengths),mode=mode,reduction=reduction,
                          **m,accepted_as_native=m['passed'] and m['replicas_equal'])
                # Only qualified candidates can be assigned accepted latency.
                if item['accepted_as_native']:
                    item['timing']=timed(lambda:block(x.clone(),*args),warmups=1,repeats=3)
                rows.append(item);save(block_cases=rows)
                if rank==0:print('BLOCK',count,adapter_mode,mode,reduction,m['passed'],m.get('rel_l2'),flush=True)
            del data,x,args,reference
        del native,block,store,reader;gc.collect();torch.cuda.empty_cache();dist.barrier()
    return rows


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--checkpoint',required=True);parser.add_argument('--lora',required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--rows',nargs='+',type=int,default=[257,512,8192,16384,32768])
    parser.add_argument('--profile',action='store_true');args=parser.parse_args()
    rank=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(rank);torch.set_num_threads(2)
    torch.set_grad_enabled(False);torch.backends.cuda.matmul.allow_tf32=False
    dist.init_process_group('nccl',timeout=timedelta(seconds=180))
    assert dist.get_world_size()==2
    assert torch.cuda.device_count()==2 and all('T4' in torch.cuda.get_device_name(i) for i in range(2))
    args.output.mkdir(parents=True,exist_ok=True)
    state=dict(status='running',physical_dual_t4=True,scope='Learned H3 block0 FFN/SwiGLU/FC2 and compact complete DiT blocks; not full-model inference',
        torch=torch.__version__,cuda=torch.version.cuda,kitchen=__import__('importlib.metadata',fromlist=['version']).version('comfy-kitchen'),
        devices=[dict(name=torch.cuda.get_device_name(i),bytes=torch.cuda.get_device_properties(i).total_memory) for i in (0,1)],
        gate=GATE,projection_cases=[],native_ffn_cases=[],block_cases=[],safety=[],profiles=[],
        memory_scope='Per-rank absolute peaks include resident full Native reference weights and TP shards. Temporary deltas isolate execution; not full-model residency.',
        source_sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in
            ('patches/optimized_tp.py','patches/quantized_collectives.py','patches/h3_lora.py','patches/tp_dit.py','benchmarks/physical_h3_reduction.py')})
    def save(**kwargs):
        state.update(kwargs)
        if rank==0:retest.write_json(args.output/'results.json',state)
    save()
    try:
        for label,v0,v1 in [('overflow',1.e4,1.e4),('zero',0.,0.),('cancellation',1.e4,-1.e4)]:
            x=torch.full((3,7),v0 if rank==0 else v1,device='cuda')
            actual=int8_sum(x,dist,2);passed=agree(torch.equal(actual,torch.full_like(x,v0+v1)))
            state['safety'].append(dict(label=label,passed=passed));save()
            if not passed:raise RuntimeError('INT8 collective overflow/safety failure')
        base=CheckpointReader(args.checkpoint)
        weights={n:base.read_tensor('blocks.0.mlp.'+n).cuda() for n in
            ('fc1.weight','fc1.weight_scale','fc2.weight','fc2.weight_scale')}
        local_fc1=torch.cat([part.chunk(2,0)[rank] for part in weights['fc1.weight'].chunk(2,0)],0).contiguous()
        local_scale=torch.cat([part.chunk(2,0)[rank] for part in weights['fc1.weight_scale'].chunk(2,0)],0).contiguous()
        local_fc2=weights['fc2.weight'].chunk(2,-1)[rank].contiguous()
        curve=base.read_tensor('adaln_t_table')
        for count in args.rows:
            gc.collect();torch.cuda.empty_cache();free,_=torch.cuda.mem_get_info()
            # Conservative bound for full Native reference plus input/shard
            # materialization. Skip unsafe sizes, never catch an avoidable OOM.
            estimated=count*(28672*4+5376*4+14336*4*4+5376*4*2+14336)*1.15+GiB
            if not agree(estimated<free-GiB):
                state['projection_cases'].append(dict(rows=count,status='memory_guard_skipped',estimated_bytes=int(estimated),free_bytes=free))
                save();continue
            enable('scalemax')
            x=block_input('dit',count,900,curve)['input'].cuda()
            packed=ck.int8_linear(x,weights['fc1.weight'],weights['fc1.weight_scale'],None,torch.float32,convrot=True,convrot_groupsize=256)
            q,s=quantize(packed,input_act='swiglu')
            reference=dequantize(tp.native_int_mm(q,weights['fc2.weight']),s,weights['fc2.weight_scale'],torch.float32).cpu()
            gate,up=packed.chunk(2,-1)
            owned=torch.cat([gate.chunk(2,-1)[rank],up.chunk(2,-1)[rank]],-1).contiguous()
            del q,s,gate,up,packed;torch.cuda.empty_cache();dist.barrier()
            def native_ffn_call():
                if rank!=0:return None
                p=ck.int8_linear(x,weights['fc1.weight'],weights['fc1.weight_scale'],None,torch.float32,convrot=True,convrot_groupsize=256)
                return ck.int8_linear(p,weights['fc2.weight'],weights['fc2.weight_scale'],None,torch.float32,
                                      convrot=True,convrot_groupsize=256,input_act='swiglu')
            native_output=native_ffn_call()
            native_score=score(native_output.cpu(),reference) if rank==0 else None
            dist.broadcast_object_list(holder:=[native_score],src=0);native_score=holder[0]
            del native_output
            if not native_score['passed']:raise RuntimeError('Native FFN differs from quantize/GEMM reference')
            state['native_ffn_cases'].append(dict(rows=count,score=native_score,timing=timed(native_ffn_call),
                placement='Original single-GPU FFN on rank0; rank1 idle'))
            save()
            def local_fc1_call():
                return ck.int8_linear(x,local_fc1,local_scale,None,torch.float32,convrot=True,convrot_groupsize=256)
            generated=local_fc1_call()
            if not agree(torch.equal(generated,owned)):raise RuntimeError('Column FC1 changed Native packed projection')
            del generated
            for round_index in (0,1):
                modes=MODES if round_index==0 else MODES[::-1]
                for mode,reduction in modes:
                    enable(mode,swiglu_reduction=reduction)
                    def row_call(comm=dist):return tp.row_linear(owned,local_fc2,weights['fc2.weight_scale'],comm,2,input_act='swiglu')
                    def ffn_call():
                        value=local_fc1_call()
                        return tp.row_linear(value,local_fc2,weights['fc2.weight_scale'],dist,2,input_act='swiglu')
                    actual=row_call();cpu=actual.cpu();m=score(cpu,reference);m['passed']=agree(m['passed'])
                    replicas=[None,None];dist.all_gather_object(replicas,hashlib.sha256(memoryview(cpu.numpy()).cast('B')).hexdigest())
                    m['replicas_equal']=replicas[0]==replicas[1];m['passed']=m['passed'] and m['replicas_equal'];del cpu
                    exact=agree(m.get('is_bit_exact',False));del actual
                    row=dict(scope='FC2_projection',rows=count,round=round_index,mode=mode,reduction=reduction,**m,
                             native_bytes_exact=exact,accepted_as_native=m['passed'])
                    row['diagnostic_timing']=timed(row_call)
                    ffn=ffn_call();fm=score(ffn.cpu(),reference);fm['passed']=agree(fm['passed']);del ffn
                    row['ffn_score']=fm;row['ffn_diagnostic_timing']=timed(ffn_call)
                    if m['passed']:row['accepted_timing']=row['diagnostic_timing']
                    if fm['passed']:row['accepted_ffn_timing']=row['ffn_diagnostic_timing']
                    if count==8192 and round_index==0 and args.profile:
                        comm=ProfileComm()
                        trace=profile(lambda:row_call(comm),args.output,f'S8192_{mode}_{reduction}')
                        state['profiles'].append(dict(mode=mode,reduction=reduction,rank=rank,trace=trace,
                            collective_calls=comm.calls,payload_bytes=sum(c['payload_bytes'] for c in comm.calls),
                            bytes_scope='Sum of tensor payload per collective invocation; not physical PCIe link bytes'))
                    state['projection_cases'].append(row);save()
                    if rank==0:print('PROJECTION',count,round_index,mode,reduction,'pass',m['passed'],'rel',m.get('rel_l2'),
                                     'critical_ms',row['diagnostic_timing']['critical_wall_median_ms'],flush=True)
            del x,owned,reference;torch.cuda.empty_cache()
        del weights,local_fc1,local_scale,local_fc2;gc.collect();torch.cuda.empty_cache();dist.barrier()
        save(stage='block_diagnostics')
        block_diagnostics(base,args.lora,args.output,save)
        exact_cases=[r for r in state['projection_cases'] if r.get('reduction')=='native_int32']
        state['native_exact_passed']=bool(exact_cases) and all(r['native_bytes_exact'] for r in exact_cases)
        state['status']='complete';save()
    except BaseException as error:
        save(status='failed',error_type=type(error).__name__);raise
    finally:
        torch.cuda.synchronize();dist.destroy_process_group()


GiB=2**30
if __name__=='__main__':main()
