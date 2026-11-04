"""Physical SM75 CUDA 13.0 operator smoke. No checkpoint, dataset, NCCL or H3 generation.

Each case compiles and launches a real kernel. Fail closed with per-case receipts.
The qualified scope is the exact source, Torch/cu130, NVCC and SM75 reported.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys
import time
import traceback


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--device', type=int, default=0)
    args = parser.parse_args()
    source = args.source.resolve()
    sys.path[:0] = [str(source), str(source/'ComfyUI'), str(source/'benchmarks')]
    os.environ['H3_ALL_OPS_CUDA130_T4_QUALIFY'] = '1'
    os.environ.setdefault('TILELANG_CACHE_DIR','/kaggle/temp/h3_cuda130_operator_cache')
    args.output.mkdir(parents=True, exist_ok=True)
    result = dict(status='running', scope='CUDA13 source-compiled H3 operators and pinned native Kitchen/Sol',
                  no_model=True, no_dataset=True, no_production_promotion=True,
                  cases=[], started=time.time())
    def save():
        (args.output/'result.json').write_text(json.dumps(result, indent=2, default=str)+'\n')
    def check(name, fn):
        started = time.monotonic()
        record = dict(name=name, status='started')
        result['cases'].append(record)
        save()
        print('CU130_SM75_CASE_START',name,flush=True)
        try:
            detail = fn()
            torch.cuda.synchronize()
            record.update(status='passed', seconds=time.monotonic()-started, detail=detail)
            print('CU130_SM75_CASE_PASS',name,round(record['seconds'],3),flush=True)
        except BaseException as exc:
            record.update(status='failed',seconds=time.monotonic()-started,error=repr(exc),
                          traceback=traceback.format_exc()[-10000:])
            result['status'] = 'failed'
            save()
            print('CU130_SM75_CASE_FAIL',name,repr(exc),flush=True)
            raise
        save()

    try:
        import comfy.options
        comfy.options.enable_args_parsing()
        import torch
        from importlib.metadata import version
        from patches.h3_operator_dependencies import (
            ALL_OPS_CUDA13_T4_TRIAL, configure_compiler, compiler_qualification)
        from patches.h3_operator_policy import H3OperatorPolicy
        import tilelang
        torch.cuda.set_device(args.device)
        torch.set_grad_enabled(False)
        torch.backends.cuda.matmul.allow_tf32=False
        assert os.environ[ALL_OPS_CUDA13_T4_TRIAL]=='1'
        assert torch.version.cuda=='13.0',(torch.__version__,torch.version.cuda)
        assert torch.cuda.get_device_capability()==(7,5),torch.cuda.get_device_name()
        assert version('comfy-kitchen')=='0.2.37'
        assert version('tilelang')=='0.1.13'
        policy=H3OperatorPolicy()
        compiler=configure_compiler(policy)
        assert compiler['qualification']=='experimental_cuda_13_0_all_ops_t4_only',compiler
        result['environment']=dict(torch=torch.__version__,cuda=torch.version.cuda,
            tilelang=version('tilelang'),kitchen=version('comfy-kitchen'),
            device=torch.cuda.get_device_name(),gpu_count=torch.cuda.device_count(),
            compiler=compiler)
        save()

        from patches.h3_operator_aot import plan
        from importlib import import_module
        rows=8
        def run_factory(module,name,arguments):
            fn=getattr(import_module('patches.'+module),name)(*arguments)
            w=arguments[1]
            if module=='tilelang_column_epilogue':
                return fn(torch.zeros(rows,w,device='cuda',dtype=torch.int32),
                          torch.ones(rows,1,device='cuda'),torch.ones(w,device='cuda')).shape
            if module=='tilelang_row_quant':
                if name=='packing_kernel':
                    out=fn(torch.zeros(rows,w,device='cuda'))
                elif name=='maximum_kernel':
                    out=fn(torch.zeros(rows,arguments[2],device='cuda'))
                else:
                    out=fn(torch.zeros(rows,arguments[2],device='cuda'),
                           torch.ones(rows,1,device='cuda'))
                return str(tuple(x.shape for x in out)) if isinstance(out,(tuple,list)) else str(tuple(out.shape))
            if module=='tilelang_swiglu_quant':
                x=torch.rand(rows,2*w,device='cuda',dtype=torch.float16)*.01
                phase=arguments[-1]
                if phase=='recompute':
                    out=fn(x,torch.ones(rows,1,device='cuda'))
                else:
                    out=fn(x)
                assert all(torch.isfinite(y).all() for y in out)
                return phase
            if module=='tilelang_transport_tradeoffs':
                return tuple(fn(torch.zeros(rows,w,device='cuda',dtype=torch.int32)).shape)
            if module=='tilelang_transport_p2':
                return tuple(fn(torch.zeros(rows,w,device='cuda',dtype=torch.float16),
                    torch.ones(rows,1,device='cuda'),torch.ones(w,device='cuda')).shape)
            if module=='tilelang_fp32_rms_rope':
                packed=torch.zeros(rows,10752,device='cuda',dtype=torch.float16)
                freqs=torch.zeros(rows,48,2,2,device='cuda',dtype=torch.float16)
                freqs[...,0,0]=1
                freqs[...,1,1]=1
                out=fn(packed,freqs,torch.ones(128,device='cuda'),
                       torch.ones(128,device='cuda'))
                assert all(torch.isfinite(y).all() for y in out)
                return [tuple(y.shape) for y in out]
            if module=='tilelang_dit_fusion':
                x=torch.zeros(rows,w,device='cuda',dtype=torch.float16)
                weights=torch.ones(w,device='cuda')
                table=torch.zeros(9,w,device='cuda')
                labels=torch.zeros(rows,device='cuda',dtype=torch.int32)
                if name=='norm_kernel':
                    out=fn(x,weights,table,table,labels)
                    assert torch.isfinite(out).all()
                    return tuple(out.shape)
                if name=='gate_kernel':
                    fn(x,torch.ones_like(x),table,labels)
                    return 'inplace'
                if name=='row_gate_kernel':
                    fn(x,torch.ones(rows,1,device='cuda'),weights,x,table,labels)
                    return 'inplace'
            raise RuntimeError('Missing actual forward for '+module+'.'+name)

        # These 2 stages cover every selected default custom TileLang AOT factory.
        # Both TP ranks are exercised, avoiding false admission of rank-local offsets.
        for stage,rank in (('te',0),('te',1),('dit',0),('dit',1)):
            for module,name,arguments in plan(rows,rank,policy,stage):
                label=f'{stage}_r{rank}.{module}.{name}.{str(arguments)}'
                check(label,lambda m=module,n=name,a=arguments:run_factory(m,n,a))
        from patches import tilelang_swiglu_quant as sq, tilelang_transport_tradeoffs as tt
        from patches import tilelang_transport_p2 as p2
        for phase in ('maximum','recompute'):
            arguments=(rows,7168,'float16','sm_75',phase)
            check('swiglu_'+phase,lambda a=arguments:run_factory('tilelang_swiglu_quant','fused_kernel',a))
        check('row_fp16_decode',lambda:tuple(tt.fp16_codec(rows,5376,tt.shift_for(3584),True)(
            torch.zeros(rows,5376,device='cuda',dtype=torch.float16)).shape))
        check('transport_int8_pack_unpack',lambda:(lambda p:tuple(tt.unpack_int8(rows,5376)(
            torch.cat((p,p),dim=0)).shape))(tt.pack_int8(rows,5376)(
            torch.zeros(rows,5376,device='cuda',dtype=torch.int32))))
        check('transport_int8_epilogue',lambda:tuple(p2.epilogue(rows,5376,'int8_gather')(
            torch.zeros(2*rows,5376//128*132,device='cuda',dtype=torch.uint8),
            torch.ones(rows,1,device='cuda'),torch.ones(5376,device='cuda')).shape))

        # Compile and execute the exact CUDA C++/Kitchen/SM75 binaries, not just
        # their compiler version check or a generic Kitchen prebuilt wheel.
        from patches import kitchen_attention as kc
        def native_kitchen_fp32():
            lib=kc.prepare(arch='75',nvcc=os.environ['CUDACXX'])
            q=torch.randn(1,2,64,128,device='cuda',dtype=torch.float32)*.02
            out=kc.int8_attention(q,q,q)
            assert out.shape==q.shape and torch.isfinite(out).all()
            return str(lib)
        check('kitchen_fp32_native_compiled_forward',native_kitchen_fp32)

        from patches import kitchen_te_attention as te
        def native_te():
            lib=te.prepare_native(nvcc=os.environ['CUDACXX'])
            q=torch.randn(1,2,64,64,device='cuda',dtype=torch.float16)*.02
            mask=torch.ones(64,64,dtype=torch.bool,device='cuda').tril()[None,None]
            out=te.attention(q,q,q,mask=mask,use_native=True)
            assert out.shape==(1,64,128) and torch.isfinite(out).all()
            # Explicit native path should be available, not silently skipped.
            assert te._backend_for_device(q.device) is not None
            return str(lib)
        check('kitchen_te_native_compiled_forward',native_te)

        from patches import sol_attention as sol
        check('sol_native_compilation',lambda:sol.prepare_native())
        for dim in (64,128):
            def sol_forward(d=dim):
                q=torch.randn(1,2,128,d,device='cuda',dtype=torch.float16)*.05
                out=sol.attention(q,q,q,allow_d64=d==64,protected_tokens=64)
                assert out.shape==q.shape and torch.isfinite(out).all()
                return tuple(out.shape)
            check(f'sol_all_int8_d{dim}_forward',sol_forward)
        result['status']='passed'
    except BaseException as exc:
        result['status']='failed'
        result['fatal_error']=repr(exc)
        result['fatal_traceback']=traceback.format_exc()[-13000:]
        raise
    finally:
        result['finished']=time.time()
        save()
        print('CU130_SM75_FINAL',result['status'],len(result['cases']),flush=True)


if __name__=='__main__':
    main()
