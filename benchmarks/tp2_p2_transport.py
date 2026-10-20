"""Real H3 row operators: immutable transports and exact-contract fused epilogues."""
import gc
import hashlib
import json
from pathlib import Path
import sys
import time
sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parent)]
import numpy as np
import torch
import torch.distributed as dist
import tp2_tradeoffs as base
from patches import tilelang_transport_p2 as fused
from test_tp2_p2_epilogue import run as units


def diagnostic(actual,reference,parts):
    """Full-output CPU diagnostics. No unstable division by individual near-zero outputs."""
    a=actual.cpu().numpy().reshape(-1);b=reference.cpu().numpy().reshape(-1)
    p=parts.cpu().numpy().reshape(2,-1);delta=a-b;absolute=np.abs(delta)
    bb=np.dot(b.astype(np.float64),b.astype(np.float64));aa=np.dot(a.astype(np.float64),a.astype(np.float64))
    error2=np.dot(delta.astype(np.float64),delta.astype(np.float64));rms=np.sqrt(bb/b.size)
    severity=np.abs(p[0].astype(np.float64)+p[1])/np.maximum(np.abs(p[0].astype(np.float64))+np.abs(p[1].astype(np.float64)),1.)
    bins=[]
    for lo,hi in ((0,1e-4),(1e-4,.001),(.001,.01),(.01,.1),(.1,1.000001)):
        mask=(severity>=lo)&(severity<hi);count=int(mask.sum())
        if count:
            e=delta[mask].astype(np.float64);r=b[mask].astype(np.float64)
            bins.append(dict(cancellation_ratio_interval=[lo,hi],count=count,RMSE=float(np.sqrt(np.mean(e*e))),
                             relative_RMSE=float(np.sqrt(np.sum(e*e)/max(np.sum(r*r),1e-300))),max_abs=float(absolute[mask].max()),
                             reference_RMS=float(np.sqrt(np.mean(r*r)))))
    near=np.abs(b)<=rms*1e-4
    return dict(relative_RMSE=float(np.sqrt(error2/max(bb,1e-300))),RMSE=float(np.sqrt(error2/b.size)),reference_RMS=float(rms),
        max_abs=float(absolute.max()),cosine_similarity=float(np.dot(a.astype(np.float64),b.astype(np.float64))/max(np.sqrt(aa*bb),1e-300)),
        absolute_error_percentiles=dict(zip(('p50','p90','p99','p99_9','p100'),map(float,np.percentile(absolute,[50,90,99,99.9,100])))),
        near_zero=dict(rule='abs(reference)<=reference_RMS*1e-4',count=int(near.sum()),max_abs_error=float(absolute[near].max()) if near.any() else None),
        cancellation_bins=bins,partial_abs_percentiles=list(map(float,np.percentile(np.abs(p.astype(np.float64)),[50,90,99,99.9,100]))),
        scope='Real native activations + learned weights at selected sites; arithmetic only, not generation-quality acceptance.')


def memory(fn):
    gc.collect();torch.cuda.synchronize();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
    start=dict(allocated=torch.cuda.memory_allocated(),reserved=torch.cuda.memory_reserved())
    y=fn();torch.cuda.synchronize()
    result=dict(start_bytes=start,peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        retained_output_bytes=y.numel()*y.element_size(),allocated_with_output_bytes=torch.cuda.memory_allocated(),
        scope='Single warmed API after releasing references and emptying allocator cache; NCCL/internal CUDA allocations excluded from PyTorch counters.')
    del y;gc.collect();torch.cuda.synchronize();result['allocated_after_output_release_bytes']=torch.cuda.memory_allocated()
    return result


def communication(args,rank,result,target):
    if args.round==0 and not args.profile:
        records=units();base.save(args.output/f'unit_rank{rank}.json',records)
    fixtures=json.loads((args.data/'real_transport_manifest.json').read_text())
    for spec in fixtures['records']:
        if args.profile and spec['name'] not in ('o_l25_s9','fc2_l0_s0'):continue
        p=args.data/spec['file']
        assert hashlib.file_digest(p.open('rb'),'sha256').hexdigest()==spec['sha256']
        data=torch.load(p,map_location='cpu',weights_only=True);full=data['x'];weights=data['weights'];act=spec['name'].startswith('fc2')
        m=full.shape[0];k=full.shape[1]//(4 if act else 2);n=weights['weight'].shape[0]
        assert (m,n,k)==(6814,5376,7168 if act else 3584)
        local=torch.cat([z[:,rank*k:(rank+1)*k] for z in full.chunk(2,-1)],-1) if act else full[:,rank*k:(rank+1)*k]
        x=local.to(rank).contiguous();w=weights['weight'][:,rank*k:(rank+1)*k].to(rank).contiguous();ws=weights['weight_scale'].to(rank).contiguous()
        del data,full,local,weights;gc.collect()
        def quant():
            with base.label('linear/activation_rotation_quantization'):
                return base.swiglu.global_row_quantize(x,dist,2) if act else base.row.global_row_quantize(x,dist,2,256,None,'packed')
        variants=['int32','fp16','fp16_fused','int8_gather','int8_gather_fused']
        if args.round%2:variants.reverse()
        if not args.profile:
            aq,asc=quant();partial=base.native.native_int_mm(aq,w)
            both=torch.empty((2*m,n),device='cuda',dtype=torch.int32);dist.all_gather_into_tensor(both,partial)
            parts=both.reshape(2,m,n);cpu_parts=parts.cpu();native_sum=parts[0]+parts[1]
            expected=base.pi.dequantize(native_sum,asc,ws,torch.float32);cpu_expected=expected.cpu()
            partial_hash=hashlib.sha256(partial.cpu().numpy().tobytes()).hexdigest()
            del aq,partial,both,parts,native_sum,expected;gc.collect()
        for variant in variants:
            scheme=variant.removesuffix('_fused');is_fused=variant.endswith('_fused');name='communication/'+spec['name']+'/'+variant
            record=dict(family='communication/'+spec['name'],name=variant,shape=dict(M=m,N=n,K=k),fixture=spec,approximate=scheme!='int32')
            def api():
                with base.label(name+'/API'):
                    q,s=quant()
                    with base.label('linear/INT8_GEMM_to_INT32'):acc=base.native.native_int_mm(q,w)
                    del q
                    if is_fused:
                        with base.label('linear/encode_'+scheme):payload=fused.encode(acc,scheme,k)
                        # Release the original accumulator before allocating receive/output buffers.
                        del acc
                        with base.label('linear/collect_'+scheme):received=fused.collect(payload,scheme,dist)
                        if received is not payload:del payload
                        with base.label('linear/fused_reconstruction_dequantize'):return fused.finish(received,s,ws,scheme,k)
                    with base.label('linear/transport_'+scheme):reduced=base.transport.reduce(acc,dist,scheme,k)
                    with base.label('linear/dequantize'):return base.pi.dequantize(reduced,s,ws,torch.float32)
            error=None;start=time.perf_counter()
            if not args.profile:
                try:
                    # Reference quantization, NCCL rounding and native dequantization are independent of the fused factory.
                    independent=base.transport_reference(cpu_parts.to(rank),scheme,k)
                    reference=base.pi.dequantize(independent,asc,ws,torch.float32)
                    actual=api();assert base.byte_equal(actual,reference),'Transport/native-dequant contract mismatch'
                    assert torch.isfinite(actual).all()
                    # Count clamping separately from ordinary absmax endpoint use.
                    saturation={}
                    for ri,part in enumerate(cpu_parts):
                        if scheme=='fp16':saturation[str(ri)]=dict(overflow_count=int(((part.float()*2.**-base.transport.shift_for(k)).abs()>65504).sum()))
                        elif scheme=='int8_gather':
                            values=part.float().reshape(m,-1,128);scale=(values.abs().amax(-1)*(1./127.)).clamp_min(1e-8)
                            z=(values/scale[...,None]).round()
                            saturation[str(ri)]=dict(clamp_count=int((z.abs()>127).sum()),endpoint_count=int((z.abs()==127).sum()),total=part.numel())
                    record.update(independent_contract_gate='bitwise_pass',output_sha256=hashlib.sha256(actual.cpu().numpy().tobytes()).hexdigest(),partial_sha256=partial_hash,
                        diagnostics=diagnostic(actual,cpu_expected,cpu_parts),saturation=saturation,gate_and_first_JIT_seconds=time.perf_counter()-start)
                    del actual,reference,independent
                except Exception as e:error=type(e).__name__+': '+str(e)
            errors=base.agree(error)
            if any(errors):
                record.update(status='failed',errors=errors);result['cases'].append(record);base.save(target,result);continue
            # Only central trajectory O and first-step FC2 consume performance budget.
            if args.profile or spec['name'] in ('o_l25_s9','fc2_l0_s0'):
                if not args.profile:
                    record['memory']=memory(api)
                    record['workspace_model']=dict(full_INT32_reconstruction_bytes=0 if is_fused else m*n*4,
                        extra_reconstruction_write_read_bytes_avoided=2*m*n*4 if is_fused else 0,
                        epilogue_launches=1 if is_fused else 2 if scheme!='int32' else 1,
                        lifetime='Fused releases original INT32 partial after encoding, before collective; unfused immutable control retains it through dequantization.')
                record['API']=base.timed(api,args)
            else:record['timing_scope']='accuracy-only trajectory endpoint'
            record['status']='complete';result['cases'].append(record);base.save(target,result);print('P2_TRANSPORT',rank,name,'complete',flush=True)
        if not args.profile:del asc,cpu_parts,cpu_expected
        del x,w,ws;gc.collect();torch.cuda.empty_cache()


if __name__=='__main__':
    base.communication=communication
    # Existing main's summary assumes API records; this harness writes richer raw records
    # and uses the common main with a profiled flag only for actual profiling.
    import argparse,datetime,os
    parser=argparse.ArgumentParser();parser.add_argument('--manifest',type=Path,required=True);parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--round',type=int,default=0);parser.add_argument('--profile',action='store_true');args=parser.parse_args()
    rank=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(rank);torch.set_grad_enabled(False);torch.backends.cuda.matmul.allow_tf32=False
    assert torch.cuda.device_count()==2 and all(torch.cuda.get_device_capability(i)==(7,5) and 'T4' in torch.cuda.get_device_name(i) for i in range(2))
    assert os.environ.get('NCCL_P2P_DISABLE')=='1' and os.environ.get('NCCL_MIN_NCHANNELS')==os.environ.get('NCCL_MAX_NCHANNELS')=='4' and 'NCCL_PROTO' not in os.environ
    dist.init_process_group('nccl',device_id=torch.device('cuda',rank),timeout=datetime.timedelta(seconds=120))
    for method in ('all_reduce','all_gather_into_tensor'):
        original=getattr(dist,method)
        def wrapped(*a,_method=method,_original=original,**kw):
            with base.label('NCCL/'+_method+'/'+str(a[0].dtype)+'/'+str(kw.get('op',''))):return _original(*a,**kw)
        setattr(dist,method,wrapped)
    base.pi.int_mm=base.native.native_int_mm;args.output.mkdir(parents=True,exist_ok=True)
    result=dict(status='running',rank=rank,hardware=torch.cuda.get_device_name(),torch=torch.__version__,cases=[],scope='Real H3 operator fixtures; generation unqualified.')
    target=args.output/f'rank{rank}.json';base.save(target,result)
    communication(args,rank,result,target);result['status']='complete';base.save(target,result)
    dist.barrier();dist.destroy_process_group()
