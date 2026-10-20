"""Actual SM75 replay, repeated timing and diagnostic ablations on native H3 captures.

This file deliberately rejects other architectures. Timings are operator timings;
they do not claim to measure the full two-GPU H3 deployment.
"""
import argparse
from contextlib import contextmanager
import gc
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'benchmarks')]
import torch
import torch.nn.functional as F
from patches.tilelang_sparse_attention import prepare, make_routes, execute, get_kernel, reference
from attention_root_causes import diagnose, metrics

MODES = ['native', 'fp16_cast', 'dense_int8', 'dense_int8_pv', 'sol_tau0', 'sol_tau0_int8_pv', 'topk50']


@contextmanager
def label(name):
    torch.cuda.nvtx.range_push(name)
    try:
        yield
    finally:
        torch.cuda.nvtx.range_pop()


def save(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def load(directory, record, start=0, heads=None, verify=True,expand_gqa=True):
    path = directory / record['file']
    if verify:
        with path.open('rb') as stream:
            assert hashlib.file_digest(stream, 'sha256').hexdigest() == record['sha256']
    data = torch.load(path, map_location='cpu', weights_only=True)
    total = data['q'].shape[1]
    heads = total-start if heads is None else min(heads, total-start)
    ids = torch.arange(start, start+heads)
    groups = total // data['k'].shape[1]
    key_ids=ids//groups if expand_gqa else (ids//groups).unique(sorted=True)
    values = [data['q'][:, ids], data['k'][:, key_ids], data['v'][:, key_ids], data['native_output'][:, ids]]
    if record['component'] == 'te':
        mask = data['mask']
        n = data['q'].shape[2]
        assert mask is not None
        assert ((mask.reshape(-1,n,n)>=0)==torch.ones(n,n,dtype=torch.bool).tril()).all()
    else:
        assert data['mask'] is None
    assert data['scale'] is None or abs(data['scale']-values[0].shape[-1]**-.5)<1e-9
    return [x.cuda().contiguous() for x in values], ids.tolist()


def routes_for(p, mode, component):
    kind = 'sol' if mode.startswith('sol') else ('topk' if mode=='topk50' else 'dense')
    routes = make_routes(p, kind, tau=0., density=.5)
    if component == 'dit':
        # Exact fixed-workload packing: 353 text + 221 reference + 300 audio.
        prefix = math.ceil(874/64)
        routes[..., :prefix] = 1
        routes[..., :prefix, :] = 1
    return routes


def config(mode, component):
    return dict(pooled=mode.startswith('sol'), int8_pv=mode in ('dense_int8_pv','sol_tau0_int8_pv'),
                causal=component=='te', sm=75)


def raw_args(p, routes):
    return (p.q,p.qi,p.ki,p.v,p.qs,p.ks,p.kc,p.vc,p.counts,routes,p.vi,p.vs)


def measure(fn, warmup=5, repeats=15):
    # A few short calls do not reliably leave T4's idle clock state. Warm the
    # actual operation for >=1 second as well as satisfying the call minimum.
    warm_start=time.perf_counter();warm_count=0
    while warm_count<warmup or time.perf_counter()-warm_start<1.0:
        for _ in range(5):fn();warm_count+=1
        torch.cuda.synchronize()
    warm_seconds=time.perf_counter()-warm_start
    gpu, wall = [], []
    for _ in range(repeats):
        first,last = torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        begin = time.perf_counter()
        first.record(); fn(); last.record(); last.synchronize()
        wall.append((time.perf_counter()-begin)*1000)
        gpu.append(first.elapsed_time(last))
    return {'warmup_minimum':warmup,'warmup_actual_calls':warm_count,'warmup_seconds':warm_seconds,
            'repeats':repeats,'gpu_ms':gpu,'wall_ms':wall,
            'gpu_median_ms':statistics.median(gpu),'wall_median_ms':statistics.median(wall),
            'gpu_min_ms':min(gpu),'gpu_max_ms':max(gpu),
            'scope':'GPU events include host enqueue gaps; wall includes final synchronization. Nsight traces identify CPU overhead.'}


def benchmark(directory, record, output, heads=None, modes=MODES, repeats=15, profile=False):
    begin = time.perf_counter()
    (q,k,v,native), head_ids = load(directory,record,heads=heads,expand_gqa=False)
    torch.cuda.synchronize()
    item={'capture':record,'head_ids':head_ids,'q_shape':list(q.shape),'input_load_and_h2d_seconds':time.perf_counter()-begin,
          'modes':{},'device':torch.cuda.current_device()}
    causal=record['component']=='te';gqa=q.shape[1]!=k.shape[1]
    for mode in modes:
        compile_seconds=0.
        if mode in ('native','fp16_cast'):
            tensors=(q,k,v) if mode=='native' else (q.half(),k.half(),v.half())
            def raw():
                with label(mode+'/native_sdpa'):
                    return F.scaled_dot_product_attention(*tensors,is_causal=causal,enable_gqa=gqa)
            end_to_end=raw
            phase={}
        else:
            with label(mode+'/preparation'):
                p=prepare(q.half(),k.half(),v.half(),enable_gqa=gqa)
            with label(mode+'/routing'):
                routes=routes_for(p,mode,record['component'])
            begin=time.perf_counter()
            kernel=get_kernel(*q.shape,**config(mode,record['component']))
            torch.cuda.synchronize();compile_seconds=time.perf_counter()-begin
            raw_inputs=raw_args(p,routes)
            def raw():
                with label(mode+'/fused_QK_softmax_PV'):
                    return kernel(*raw_inputs)[0][:,:,:p.tokens]
            def end_to_end():
                with label(mode+'/input_cast'):
                    half=[x.half() for x in (q,k,v)]
                with label(mode+'/preparation_quantization_and_packing'):
                    current=prepare(*half,enable_gqa=gqa)
                with label(mode+'/routing'):
                    selected=routes_for(current,mode,record['component'])
                with label(mode+'/public_API_QK_softmax_PV_unpack'):
                    return execute(current,selected,pooled=config(mode,record['component'])['pooled'],
                                   int8_pv=config(mode,record['component'])['int8_pv'],causal=causal)[0]
            phase={'preparation_quantization_and_packing':measure(lambda:prepare(q.half(),k.half(),v.half(),enable_gqa=gqa),repeats=repeats),
                   'routing':measure(lambda:routes_for(p,mode,record['component']),repeats=repeats),
                   'selected_block_density':routes.float().mean().item()}
            kernel.export_ptx(str(output/f"{record['component']}_{record['layer']}_{record['denoise_call']}_{len(head_ids)}_{mode}.ptx"))
            (output/f"{record['component']}_{record['layer']}_{record['denoise_call']}_{len(head_ids)}_{mode}.cu").write_text(kernel.get_kernel_source())
        torch.cuda.reset_peak_memory_stats()
        # First execution/lazy vendor initialization is always outside measurements.
        begin=time.perf_counter();actual=raw();torch.cuda.synchronize()
        first_execution=time.perf_counter()-begin
        if profile:
            for _ in range(5):end_to_end()
            torch.cuda.synchronize()
            torch.cuda.profiler.start()
            with label('steady_state/'+mode):
                for _ in range(repeats):end_to_end()
            torch.cuda.synchronize();torch.cuda.profiler.stop()
            timing={}
        else:
            timing={'raw_kernel_or_native_SDPA':measure(raw,repeats=repeats),
                    'full_attention_API':measure(end_to_end,repeats=repeats)}
        item['modes'][mode]={'compile_or_cache_lookup_seconds':compile_seconds,
            'first_execution_seconds':first_execution,'vs_saved_native_output':metrics(actual,native),
            'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),
            'phases':phase,**timing}
        print('BENCHMARK',record['file'],len(head_ids),mode,json.dumps(item['modes'][mode]),flush=True)
    return item


def accuracy(directory, records, output, chunk=8):
    result={'status':'running','hardware':hardware(),'query_diagnostics':'64 deterministic query rows per capture/head; full-sequence operator output error separately',
            'records':[]}
    for record in records:
        row={'capture':record,'head_chunks':[]};result['records'].append(row)
        total=record['q_shape'][1]
        for start in range(0,total,chunk):
            (q,k,v,native),ids=load(directory,record,start,chunk,verify=start==0)
            p=prepare(q.half(),k.half(),v.half());causal=record['component']=='te'
            routes=routes_for(p,'sol_tau0' if record['component']=='dit' else 'dense_int8',record['component'])
            diagnostic=diagnose(q,k,v,p,routes,causal=causal)
            outputs={}
            available=MODES if record['component']=='dit' else MODES[:4]
            for mode in available:
                if mode in ('native','fp16_cast'):
                    tensors=(q,k,v) if mode=='native' else (q.half(),k.half(),v.half())
                    actual=F.scaled_dot_product_attention(*tensors,is_causal=causal)
                else:
                    selection=routes_for(p,mode,record['component']);cfg=config(mode,record['component'])
                    actual,_=execute(p,selection,cfg['pooled'],cfg['int8_pv'],causal=causal)
                outputs[mode]=metrics(actual,native)
            row['head_chunks'].append({'head_ids':ids,'outputs':outputs,'root_causes':diagnostic})
            save(output/'accuracy.json',result)
            print('ACCURACY',record['file'],ids,flush=True)
            del q,k,v,native,p,routes,actual;gc.collect();torch.cuda.empty_cache()
    result['status']='complete';save(output/'accuracy.json',result)


def hardware():
    return {'torch':torch.__version__,'cuda':torch.version.cuda,
        'devices':[{'name':torch.cuda.get_device_name(i),'capability':list(torch.cuda.get_device_capability(i)),
                    'total_memory':torch.cuda.get_device_properties(i).total_memory}
                   for i in range(torch.cuda.device_count())]}


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--data',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--accuracy',action='store_true');parser.add_argument('--profile',action='store_true')
    parser.add_argument('--file');parser.add_argument('--heads',type=int);parser.add_argument('--mode',choices=MODES)
    parser.add_argument('--repeats',type=int,default=15);parser.add_argument('--device',type=int,default=0)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    torch.set_grad_enabled(False);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.cuda.set_device(args.device)
    hw=hardware();save(args.output/'hardware.json',hw)
    assert len(hw['devices'])==2 and all(g['capability']==[7,5] and 'T4' in g['name'] for g in hw['devices']), hw
    records=json.loads((args.data/'capture_manifest.json').read_text())['records']
    if args.accuracy:
        accuracy(args.data,records,args.output)
    else:
        wanted={'te_layer10_call-1.pt','dit_layer00_call00.pt','dit_layer25_call09.pt','dit_layer49_call19.pt',
                'video_layer00_call00.pt','video_layer18_call15.pt','video_layer35_call29.pt'}
        selected=[r for r in records if r['file'] in wanted] if not args.file else [r for r in records if r['file']==args.file]
        assert selected
        result={'status':'running','hardware':hw,'scope':'one-GPU operator replay, full fixed-workload sequence; DiT head slice 28 is one TP rank, not full-pipeline throughput',
                'cases':[]}
        try:
            for record in selected:
                counts=[args.heads] if args.heads else ([56,28] if record['component']=='dit' else [record['q_shape'][1]])
                for heads in counts:
                    modes=[args.mode] if args.mode else (MODES if record['component']=='dit' else MODES[:4])
                    result['cases'].append(benchmark(args.data,record,args.output,heads,modes,args.repeats,args.profile))
                    save(args.output/'benchmark.json',result);gc.collect();torch.cuda.empty_cache()
            result['status']='complete'
        except Exception:
            result['status']='failed';result['traceback']=traceback.format_exc();raise
        finally:
            save(args.output/'benchmark.json',result)
