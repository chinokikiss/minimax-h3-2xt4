"""Local operator preview and TP2 arithmetic gates; no model or NCCL emulation timing."""
import argparse
from contextlib import contextmanager
import gc
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'benchmarks')]
import torch
import torch.distributed as dist
import comfy_kitchen as ck
from patches import tilelang_swiglu_pad as fused,optimized_tp as original,parallel_int8 as pi
from local_kitchen_preview import timing
from tp2_timing import warm_device


def save(path,value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2),encoding='utf8');tmp.replace(path)


def equal(a,b):return a.shape==b.shape and a.dtype==b.dtype and torch.equal(a.view(torch.uint8),b.view(torch.uint8))


def dependencies(output):
    """Process-only local architecture/signature adapters; original files untouched."""
    import patches
    path=ROOT/'patches/tilelang_row_quant.py';source=path.read_text(encoding='utf8')
    capability=torch.cuda.get_device_capability()
    if capability==(12,0):
        changed=source.replace("'sm_75'","'sm_120'").replace('!= (7, 5)','!= (12, 0)')
        adapted=output/'row_dependency_sm120.py';adapted.write_text(changed,encoding='utf8')
        spec=importlib.util.spec_from_file_location('swiglu_row_preview',adapted);module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
        sys.modules['patches.tilelang_row_quant']=module;patches.tilelang_row_quant=module
    elif capability==(7,5):
        from patches import tilelang_row_quant as module
        changed=source
    else:raise ValueError('Explicit SM75/SM120 operator harness only')
    backend=ck.backends.cuda
    if not hasattr(backend,'_gemm_vector_arg'):
        # Pinned 0.2.37's argument normalization helper, absent in local .33.
        def gemm_vector(value,device,dtype):
            if value.device==device and value.dtype==dtype and value.is_contiguous() and value.data_ptr()%16==0:return value
            value=value.to(device=device,dtype=dtype).contiguous()
            return value if value.data_ptr()%16==0 else value.clone()
        backend._gemm_vector_arg=gemm_vector
    if len(inspect.signature(backend._apply_input_act).parameters)==2:
        old=backend._apply_input_act
        def activation(x,act,weight=None,eps=0.):
            if weight is not None or eps!=0.:raise ValueError('Local activation adapter only covers this test')
            return old(x,act)
        backend._apply_input_act=activation
    pi.int_mm=original.native_int_mm
    return module,dict(original_normalized_sha256=hashlib.sha256(source.encode()).hexdigest(),adapted_normalized_sha256=hashlib.sha256(changed.encode()).hexdigest(),
        scope='Only local dependency guard/target, 0.2.33 activation signature and missing .37 vector argument normalization helper adapted in this process; shared files/environment untouched.')


def eager_pad(gate,up,rank,world=2):
    activated=torch.nn.functional.silu(gate).mul_(up)
    out=torch.zeros((*gate.shape[:-1],gate.shape[-1]*world),device=gate.device,dtype=gate.dtype)
    out[...,rank*gate.shape[-1]:(rank+1)*gate.shape[-1]].copy_(activated)
    return out


def units(output):
    torch.manual_seed(20261017);records=[]
    for width in (1,257):
        base=torch.randn(7,width*2,device='cuda')
        for kind in ('normal','zero','subnormal','large','nonfinite'):
            packed=base.clone()
            if kind=='zero':packed.zero_();packed[0,0]=-0.;packed[1,-1]=-0.
            elif kind=='subnormal':
                packed[:,:width]*=1.e-40;packed[:,width:]=1.;packed[1,width:]=-1.
            elif kind=='large':packed*=100.;packed[:,0]=-100.
            elif kind=='nonfinite':
                packed[0,0]=float('nan');packed[1,0]=float('inf');packed[2,0]=-float('inf')
                packed[3,-1]=float('nan');packed[4,-1]=float('inf');packed[5,-1]=-float('inf')
            before=packed.clone();g,u=packed.chunk(2,-1)
            for separate in (False,True):
                gate,up=(g.contiguous(),u.contiguous()) if separate else (g,u)
                for rank in (0,1):
                    actual=fused.swiglu_pad(gate,up,rank=rank);expected=eager_pad(gate,up,rank)
                    same=equal(actual,expected);assert same,(width,kind,separate,rank,int((actual.view(torch.uint8)!=expected.view(torch.uint8)).sum()))
                    assert equal(packed,before),'Input mutated'
                    records.append(dict(width=width,pattern=kind,separate=separate,rank=rank,bitwise_equal=True,input_unchanged=True))
    packed=torch.randn(2,3,14,device='cuda');g,u=packed.chunk(2,-1)
    actual=fused.swiglu_pad_packed(packed,rank=1,world_size=3)
    assert equal(actual,eager_pad(g,u,1,3));records.append(dict(test='leading_dimensions_world3',bitwise_equal=True))
    save(output/'unit_gates.json',records);print('SWIGLU_UNIT_GATES',len(records),'pass',flush=True)


class FixedTP2:
    """Validate actual local contributions then inject precomputed collective results.

    Arithmetic gate only: no communication latency/throughput measurement.
    """
    ReduceOp=dist.ReduceOp
    def __init__(self,rank,local_max,global_max,partial,total):
        self.rank=rank;self.local_max=local_max;self.global_max=global_max;self.partial=partial;self.total=total;self.calls=[]
    def get_rank(self):return self.rank
    def get_world_size(self):return 2
    def is_initialized(self):return True
    def all_reduce(self,x,op):
        if op==self.ReduceOp.MAX:
            assert equal(x,self.local_max),'Changed local maximum';x.copy_(self.global_max);self.calls.append('FP32_MAX')
        elif op==self.ReduceOp.SUM:
            assert x.dtype==torch.int32 and equal(x,self.partial),'Changed local INT32 partial';x.copy_(self.total);self.calls.append('INT32_SUM')
        else:raise ValueError('Unexpected collective')


@contextmanager
def full_width_route_fixture(required):
    # SM120 has enough shared memory for a native rotation route that the
    # target SM75 fixture does not use. This override is confined to arithmetic
    # gates; it changes neither device capability nor any performance timing.
    previous=original.use_fallback_rotation
    try:
        if required:original.use_fallback_rotation=lambda *args,**kw:True
        yield
    finally:original.use_fallback_rotation=previous


def chain_gate(component,values,padded,row,output):
    width=values[0][0].shape[-1];rows=values[0][0].shape[0];n=5120 if component=='te' else 5376
    backend=ck.backends.cuda;rotation=backend._build_hadamard(256,device='cuda',dtype=torch.float32)
    rotated=[backend._rotate_activation(p,rotation,256) for p in padded]
    maxima=[r[:,i*width:(i+1)*width].abs().amax(-1,keepdim=True) for i,r in enumerate(rotated)]
    maximum=torch.maximum(*maxima)
    qs=[row.quantize(r,maximum,width,i*width) for i,r in enumerate(rotated)]
    weights=[torch.randint(-127,128,(n,width),device='cuda',dtype=torch.int8) for _ in (0,1)]
    partial=[original.native_int_mm(qs[i][0],weights[i]) for i in (0,1)]
    total=partial[0]+partial[1];scale=torch.full((n,1),.001,device='cuda');bias=torch.randn(n,device='cuda')*.01
    residual=torch.randn(rows,n,device='cuda')*.01;residual_scale=torch.full((rows,1),.5,device='cuda')
    records=[]
    for rank,(gate,up,packed) in enumerate(values):
        comm=FixedTP2(rank,maxima[rank],maximum,partial[rank],total)
        if component=='dit':actual_q,actual_s=fused.global_row_quantize(packed,comm,2)
        else:
            p=fused.swiglu_pad(gate,up,rank=rank);actual_q,actual_s=fused._quantize_padded(p,width,comm,2,256);del p
        assert equal(actual_q,qs[rank][0]) and equal(actual_s,qs[rank][1])
        comm=FixedTP2(rank,maxima[rank],maximum,partial[rank],total)
        native_fallback=original.use_fallback_rotation(gate,width*2,256)
        override=torch.cuda.get_device_capability()==(12,0) and not native_fallback
        if torch.cuda.get_device_capability()==(7,5):assert native_fallback,'Target dispatch differs from TP2 manifest'
        with full_width_route_fixture(override):
            if component=='dit':
                actual=fused.row_linear(packed,weights[rank],scale,comm,2,input_act='swiglu',bias=bias,residual=residual,residual_scale=residual_scale)
            else:actual=fused.row_linear_parts(gate,up,weights[rank],scale,comm,2,bias=bias,residual=residual,residual_scale=residual_scale)
        expected=pi.dequantize(total,qs[rank][1],scale,torch.float32,bias)
        expected=torch.addcmul(residual,expected,residual_scale)
        assert equal(actual,expected) and comm.calls==['FP32_MAX','INT32_SUM']
        record=dict(component=component,rank=rank,M=rows,N=n,local_K=width,global_K=width*2,padded_equal=True,q_equal=True,scale_equal=True,
            local_MAX_equal=True,local_INT32_partial_equal=True,output_with_bias_and_residual_bitwise_equal=True,
            output_sha256=hashlib.sha256(actual.cpu().contiguous().numpy().tobytes()).hexdigest(),
            local_full_width_route_fixture_override=override,
            scope='Full-size arithmetic only. Both ranks evaluated sequentially on one GPU; collective results precomputed and each local contribution checked. No NCCL timing.')
        records.append(record);print('SWIGLU_CHAIN_GATE',component,rank,'pass',flush=True)
        del actual,expected,actual_q,actual_s;gc.collect()
    save(output/(component+'_chain_gates.json'),records)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',required=True,type=Path);ap.add_argument('--repeats',type=int,default=15);ap.add_argument('--skip-chain',action='store_true');ap.add_argument('--units-only',action='store_true');a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    torch.cuda.set_device(0);torch.set_grad_enabled(False);torch.backends.cuda.matmul.allow_tf32=False
    row,adaptation=dependencies(a.output)
    result=dict(status='running',hardware=torch.cuda.get_device_name(),capability=list(torch.cuda.get_device_capability()),torch=torch.__version__,
        scope='Local isolated SwiGLU/padding timing, synthetic current TP2 shapes; arithmetic chain gates with validated fixed collectives. No H3 or distributed performance claim.',
        adaptation=adaptation,cases=[])
    save(a.output/'results.json',result);units(a.output)
    if a.units_only:
        result['status']='units_complete';save(a.output/'results.json',result);return
    for component,rows,width in (('te',353,12800),('dit',6814,7168)):
        torch.manual_seed(20261017+(99 if component=='dit' else 0));values=[];expected_all=[]
        for rank in (0,1):
            if component=='dit':
                packed=torch.randn(rows,2*width,device='cuda')*.25;gate,up=packed.chunk(2,-1)
            else:
                packed=None;gate=torch.randn(rows,width,device='cuda')*.25;up=torch.randn_like(gate)*.25
            values.append((gate,up,packed));expected=eager_pad(gate,up,rank);expected_all.append(expected)
            actual=fused.swiglu_pad(gate,up,rank=rank);assert equal(actual,expected);del actual
            def eager():return eager_pad(gate,up,rank)
            def packed_padding():
                activated=torch.nn.functional.silu(gate).mul_(up)
                return row.packing_kernel(rows,width,2,rank)(activated)
            def combined():return fused.swiglu_pad(gate,up,rank=rank)
            warm_device();record=dict(component=component,rank=rank,M=rows,local_width=width,output_width=width*2,
                gate_stride=list(gate.stride()),up_stride=list(up.stride()),padded_output_bitwise_equal=True)
            for name,fn in (('eager',eager),('packed_padding',packed_padding),('fused',combined)):
                torch.cuda.reset_peak_memory_stats();before=torch.cuda.memory_allocated();start=time.perf_counter()
                fn();torch.cuda.synchronize();first=time.perf_counter()-start
                record[name]=timing(fn,a.repeats);record[name]['first_call_including_possible_JIT_seconds']=first
                record[name]['peak_extra_allocated_bytes']=torch.cuda.max_memory_allocated()-before
            # Repeat in reverse order without compilation.
            for name,fn in (('fused',combined),('packed_padding',packed_padding),('eager',eager)):record[name+'_repeat']=timing(fn,a.repeats)
            result['cases'].append(record);save(a.output/'results.json',result)
            print('SWIGLU_BENCH',component,rank,{name:record[name]['gpu_median_ms'] for name in ('eager','packed_padding','fused')},flush=True)
        if not a.skip_chain:chain_gate(component,values,expected_all,row,a.output)
        del values,expected_all,gate,up,packed,expected;gc.collect();torch.cuda.empty_cache()
    result['status']='complete';save(a.output/'results.json',result)


if __name__=='__main__':main()
