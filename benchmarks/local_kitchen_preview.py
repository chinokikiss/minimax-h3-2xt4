"""Isolated SM120 preview of the frozen SM75 experiment; never installs hooks."""
import argparse
import gc
import hashlib
import importlib
import importlib.metadata
import json
from pathlib import Path
import random
import statistics
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'benchmarks')]
import torch


def digest(data):return hashlib.sha256(data).hexdigest()


def save(path,value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2));tmp.replace(path)


def adapt(output,compatibility):
    package=output/'adapted'/'preview_patches';package.mkdir(parents=True,exist_ok=True)
    (package/'__init__.py').write_text('')
    changes={}
    for name in ('tilelang_grouped_tp_optimized','tilelang_grouped_tp_kitchen','tilelang_grouped_fused',
                 'tilelang_grouped_qk','tilelang_dense_preparation','tilelang_sparse_attention'):
        source=ROOT/'patches'/f'{name}.py';original=source.read_text(encoding='utf-8')
        changed=original.replace('from patches.','from preview_patches.').replace("'sm_75'","'sm_120'").replace('!=(7,5)','!=(12,0)')
        changed=changed.replace('Actual SM75','Local preview SM120').replace('actual SM75','local preview SM120')
        if name=='tilelang_grouped_tp_kitchen' and compatibility=='scalar':
            # Installed TileLang 0.1.13 emits an invalid cutlass::half_t ->
            # __half conversion for vectorized V-padding on Windows SM120.
            # Restrict the compatibility workaround to fused preparation.
            changed=changed.replace("out_idx=[4,5,6,7,8],target={'kind':'cuda','arch':'sm_120'},pass_configs={'tl.enable_async_copy':False}",
                "out_idx=[4,5,6,7,8],target={'kind':'cuda','arch':'sm_120'},pass_configs={'tl.enable_async_copy':False,'tirx.disable_vectorize':True}")
        (package/source.name).write_text(changed,encoding='utf-8')
        changes[str(source)]=dict(original_normalized_sha256=digest(original.encode()),adapted_sha256=digest(changed.encode()))
    save(output/'source_adaptation.json',dict(changes=changes,compatibility=compatibility,policy='Import namespace, architecture target/guards/messages changed. Scalar compatibility disables fused-prep vectorization; pack compatibility preserves vectorization and repairs the generated half-packing helper signature using the same bit packing. Originals untouched.'))
    sys.path.insert(0,str(package.parent))
    return (importlib.import_module('preview_patches.tilelang_grouped_tp_optimized'),
            importlib.import_module('preview_patches.tilelang_grouped_tp_kitchen'))


def timing(fn,repeats):
    for _ in range(5):fn()
    torch.cuda.synchronize();gpu=[];wall=[]
    for _ in range(repeats):
        a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True)
        start=time.perf_counter();a.record();fn();b.record();b.synchronize()
        gpu.append(a.elapsed_time(b));wall.append((time.perf_counter()-start)*1000)
    return dict(gpu_median_ms=statistics.median(gpu),wall_median_ms=statistics.median(wall),gpu_ms=gpu,wall_ms=wall,warmup_calls=5,repeats=repeats)


def graph_timing(fn,repeats):
    # Diagnostic ablation: remove per-operation Python dispatch. This is not
    # a claim that the current production TP2 pipeline uses CUDA Graphs.
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):fn()
    torch.cuda.current_stream().wait_stream(stream);torch.cuda.synchronize()
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):captured=fn()
    result=timing(graph.replay,repeats)
    result['scope']='Captured whole attention API replay; excludes Python prep/dispatch and graph capture. Diagnostic only, not current TP2 deployment.'
    del captured,graph
    return result


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',required=True,type=Path)
    ap.add_argument('--data',type=Path,default=Path('D:/attention_eval/t4_redo_20261013/dataset'))
    ap.add_argument('--matrix',type=Path,default=Path('D:/attention_eval/operator_optimization_tp2_20261017/kitchen_inspired/offline/shortlist.json'))
    ap.add_argument('--components',nargs='+',default=['video','te','dit']);ap.add_argument('--ranks',nargs='+',type=int,default=[0,1])
    ap.add_argument('--compatibility',choices=['scalar','pack'],default='pack')
    ap.add_argument('--names',nargs='+');ap.add_argument('--graphs',action='store_true')
    ap.add_argument('--repeats',type=int,default=10);args=ap.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    assert torch.cuda.get_device_capability()==(12,0),'This harness explicitly targets the local SM120 preview only'
    torch.set_grad_enabled(False);torch.backends.cuda.matmul.allow_tf32=False
    # Localized MSVC diagnostics otherwise hide the actual compiler error.
    # This changes decoding in this benchmark process only, never codegen.
    from tilelang.contrib import nvcc
    def diagnostic_text(value):
        if not isinstance(value,bytes):return str(value)
        try:return value.decode('utf-8')
        except UnicodeDecodeError:return value.decode('gb18030',errors='replace')
    nvcc.py_str=diagnostic_text
    if args.compatibility=='pack':
        compile_cuda=nvcc.compile_cuda
        def compatible_compile(code,*pos,**kw):
            if 'tl::pack_float16x4(' in code:
                # The installed helper accepts CUDA __half but codegen passes
                # cutlass::half_t. Both contain exactly the same 16 bits.
                helper='''template <typename H>
__device__ __forceinline__ unsigned long long preview_pack_float16x4(const H x, const H y, const H z, const H w) {
  static_assert(sizeof(H)==2, "Expected 16-bit half representation");
  unsigned long long a=*((const unsigned short *)&x), b=*((const unsigned short *)&y);
  unsigned long long c=*((const unsigned short *)&z), d=*((const unsigned short *)&w);
  return a | (b<<16) | (c<<32) | (d<<48);
}
'''
                signature=digest(code.encode())[:16]
                (args.output/f'{signature}_original.cu').write_text(code)
                code=code.replace('tl::pack_float16x4(','preview_pack_float16x4(')
                at=code.index('extern "C"');code=code[:at]+helper+code[at:]
                (args.output/f'{signature}_compatible.cu').write_text(code)
            return compile_cuda(code,*pos,**kw)
        nvcc.compile_cuda=compatible_compile
    old,new=adapt(args.output,args.compatibility)
    from tp2_kitchen_attention import dense_sample,dense_error,identity
    import comfy_kitchen as ck
    result=dict(status='running',scope='Single SM120 GPU; TP2 logical partitions run sequentially. No distributed latency, sparse-policy or full-generation evaluation.',
                hardware=torch.cuda.get_device_name(),capability=list(torch.cuda.get_device_capability()),
                versions={x:importlib.metadata.version(x) for x in ('torch','tilelang','comfy-kitchen')},
                cuda_runtime=torch.version.cuda,cases=[],seed=20261017,comparison_scope='Local installed Kitchen, not the queued T4 package version.')
    result['local_compatibility_workaround']=('Fused preparation only: tirx.disable_vectorize=True.' if args.compatibility=='scalar' else
        'Generated CUDA only: equivalent 16-bit pack_float16x4 helper with a generic 2-byte input type. Preserves vectorized stores and attention scheduling. Both CUDA sources saved when compiled.')
    try:result['nvidia_smi']=subprocess.check_output(['nvidia-smi','--query-gpu=name,driver_version,memory.total,memory.used,utilization.gpu','--format=csv'],text=True)
    except Exception:pass
    save(args.output/'results.json',result)
    matrix=json.loads(args.matrix.read_text())['configurations']
    records=json.loads((args.data/'capture_manifest.json').read_text())['records']
    files=dict(te='te_layer00_call-1.pt',dit='dit_layer25_call09.pt',video='video_layer00_call00.pt')
    # Warm clocks outside any timing; no model execution.
    x=torch.randn(1024,1024,device='cuda',dtype=torch.float16)
    until=time.perf_counter()+1
    while time.perf_counter()<until:
        torch.mm(x,x);torch.cuda.synchronize()
    del x
    for component in args.components:
        record=next(r for r in records if r['file']==files[component]);path=args.data/record['file']
        with path.open('rb') as handle:assert hashlib.file_digest(handle,'sha256').hexdigest()==record['sha256']
        data=torch.load(path,map_location='cpu',weights_only=True)
        for rank in args.ranks:
            if component=='te':values=[data['q'][:,rank*32:(rank+1)*32],data['k'][:,rank*4:(rank+1)*4],data['v'][:,rank*4:(rank+1)*4]];offset=0
            elif component=='dit':values=[data[z][:,rank*28:(rank+1)*28] for z in ('q','k','v')];offset=0
            else:
                lo,hi=((0,1024),(1024,1797))[rank];values=[data['q'][:,:,lo:hi],data['k'],data['v']];offset=lo
            original=[v.to('cuda').contiguous() for v in values];del values
            if component=='dit':
                b,h,n,d=original[2].shape;storage=torch.empty((b,n,3*h*d),device='cuda',dtype=original[2].dtype)
                packed=storage.view(b,n,3,h,d)[:,:,2].permute(0,2,1,3);packed.copy_(original[2]);original[2]=packed
            b,h,n,d=original[0].shape;causal=component=='te';baseline=matrix[component][0]
            start=time.perf_counter()
            try:
                p=old.prepare(*(x.half() for x in original),fused=True,query_tile=baseline['tile'])
                expected=old.execute(p,causal=causal,query_offset=offset,cache=baseline['cache'],layout='BND',output_dtype=original[0].dtype)
                if component=='video':expected=torch.nan_to_num(expected)
                torch.cuda.synchronize()
            except Exception:
                result['cases'].append(dict(component=component,rank=rank,status='baseline_failed',traceback=traceback.format_exc()))
                save(args.output/'results.json',result);print('BASELINE_FAILED',component,rank,traceback.format_exc(),flush=True);continue
            baseline_jit=time.perf_counter()-start
            indices,reference=dense_sample(*original,offset,causal)
            configs=[c for c in matrix[component] if not args.names or c['name'] in args.names];random.Random(20261017+rank).shuffle(configs)
            successful=[]
            for cfg in configs:
                native=cfg.get('native',False)
                row=dict(component=component,rank=rank,name=cfg['name'],config=cfg,q_shape=list(original[0].shape),k_shape=list(original[1].shape),input_strides=[list(x.stride()) for x in original],fixture_sha256=record['sha256'],baseline_first_call_JIT_seconds=baseline_jit)
                def prep():return new.prepare(*original,query_tile=cfg['tile']) if cfg.get('fused') else old.prepare(*(x.half() for x in original),fused=True,query_tile=cfg['tile'])
                def raw(prepared):
                    function=new.execute if 'threads' in cfg else old.execute
                    kw=dict(threads=cfg['threads'],warp_policy=cfg['warp']) if 'threads' in cfg else {}
                    out=function(prepared,causal=causal,query_offset=offset,cache=cfg['cache'],layout='BND',output_dtype=original[0].dtype,**kw)
                    return torch.nan_to_num(out) if component=='video' else out
                def api():
                    if native:
                        out=ck.int8_attention(*original)
                        return torch.nan_to_num(out.transpose(1,2).reshape(b,n,h*d).contiguous())
                    return raw(prep())
                start=time.perf_counter()
                try:
                    actual=api();torch.cuda.synchronize();row['first_call_JIT_seconds']=time.perf_counter()-start
                    assert torch.isfinite(actual).all()
                    if not native:
                        prepared=prep();own=old.prepare(*(x.half() for x in original),fused=True,query_tile=cfg['tile'])
                        row['operand_identity']=identity(prepared,own);assert all(row['operand_identity'].values()),row['operand_identity']
                        torch.testing.assert_close(actual,expected,rtol=.001,atol=.0002)
                        row['bitwise_equal_to_current']=torch.equal(actual.view(torch.uint8),expected.view(torch.uint8))
                        row['preparation']=timing(prep,args.repeats);row['attention_and_output']=timing(lambda:raw(prepared),args.repeats)
                        del prepared,own
                    row['dense_sample_error']=dense_error(actual.reshape(b,n,h,d).transpose(1,2).index_select(2,indices),reference)
                    row['output_sha256']=digest(actual.cpu().contiguous().numpy().tobytes())
                    row['API']=timing(api,args.repeats);row['status']='complete';successful.append(cfg)
                    if args.graphs:
                        try:row['graph_API']=graph_timing(api,args.repeats)
                        except Exception:row['graph_error']=traceback.format_exc()
                    del actual
                    print('LOCAL_PREVIEW',component,rank,cfg['name'],row['API']['gpu_median_ms'],flush=True)
                except Exception:
                    row.update(status='failed',traceback=traceback.format_exc());print('CANDIDATE_FAILED',component,rank,cfg['name'],row['traceback'],flush=True)
                result['cases'].append(row);save(args.output/'results.json',result)
            # An already-compiled second round, in reverse order, exposes first-round drift.
            for cfg in reversed(successful):
                native=cfg.get('native',False)
                row=next(r for r in result['cases'] if r.get('component')==component and r.get('rank')==rank and r.get('name')==cfg['name'])
                row['API_repeat']=timing(api,args.repeats)
            save(args.output/'results.json',result)
            del p,expected,original,reference;gc.collect();torch.cuda.empty_cache()
        del data;gc.collect()
    result['status']='complete';save(args.output/'results.json',result)


if __name__=='__main__':main()
