"""TE causal/precision audit of the pinned installed Kitchen operator.

Synthetic inputs only. A failed prefix-invariance property is a finding, not
an excuse to change quantization, the production policy or package versions.
"""
import argparse
from dataclasses import replace
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback
from types import ModuleType

import torch


def reference(q,k,v,mask,*,dtype=torch.float32):
    """Explicit GQA QK/softmax/PV at the requested arithmetic precision."""
    if q.ndim!=4 or k.shape!=v.shape or q.shape[1]%k.shape[1]:
        raise ValueError('Compatible BHND/GQA required')
    groups=q.shape[1]//k.shape[1]
    q=q.to(dtype);k=k.to(dtype).repeat_interleave(groups,1);v=v.to(dtype).repeat_interleave(groups,1)
    logits=(q@k.transpose(-1,-2))*q.shape[-1]**-.5
    logits=logits.masked_fill(~mask,float('-inf')) if mask.dtype==torch.bool else logits+mask.to(dtype)
    return logits.softmax(-1)@v


def mask_for(n,kind,device,dtype):
    allowed=torch.ones(n,n,device=device,dtype=torch.bool).tril()
    if kind=='bool':return allowed
    mask=torch.zeros(n,n,device=device,dtype=dtype).masked_fill(~allowed,torch.finfo(dtype).min/4)
    if kind=='padding_bias':
        mask[:,17]=torch.finfo(dtype).min/4
        mask[31:,:1]+=.125
        return mask[None,None]
    if kind!='additive':raise ValueError(kind)
    return mask


def metrics(actual,expected):
    actual=actual.double();expected=expected.double();delta=actual-expected
    flat=delta.abs().flatten()
    return dict(max_abs=float(flat.max()),mean_abs=float(flat.mean()),rmse=float(delta.square().mean().sqrt()),
        relative_l2=float(delta.norm()/expected.norm().clamp_min(1e-30)),p99_abs=float(torch.quantile(flat,.99)),
        changed_fraction=float((delta!=0).double().mean()),byte_equal=torch.equal(actual,expected),
        finite=bool(torch.isfinite(actual).all() and torch.isfinite(expected).all()))


def fixture(n,dtype,device,seed,amplitude=1.,h=32,hk=4):
    generator=torch.Generator(device=device).manual_seed(seed)
    return tuple((torch.randn(1,heads,n,128,generator=generator,device=device,dtype=torch.float32)*amplitude).to(dtype)
                 for heads in (h,hk,hk))


def save(path,state):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(state,indent=2)+'\n',encoding='utf8');tmp.replace(path)


def worker(rank,output):
    import comfy_kitchen as ck
    from experiments.kitchen_attention.attention import dequantized_operands,value_storage_indices
    # Supplied separately from the immutable qualified capsule, hash-bound by
    # the notebook. The final production adapter is tested without editing it.
    source=Path(__file__).with_name('kitchen_te_attention.py')
    te=ModuleType('audit_te_adapter');exec(compile(source.read_bytes(),str(source),'exec'),te.__dict__)
    torch.cuda.set_device(rank);torch.set_grad_enabled(False)
    assert torch.cuda.get_device_capability(rank)==(7,5) and 'T4' in torch.cuda.get_device_name(rank)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.set_float32_matmul_precision('highest')
    state=dict(status='running',rank=rank,gpu=torch.cuda.get_device_name(rank),uuid=str(torch.cuda.get_device_properties(rank).uuid),
        prefix=[],precision=[],fixed_packed=[],controls=[],scope='Synthetic TE attention only, not model/generation quality',
        reference='Explicit FP32 QK/softmax/PV, TF32 disabled; input values preserved after dtype conversion',
        production_adapter_sha256=hashlib.sha256(source.read_bytes()).hexdigest())
    started=time.monotonic();deadline=started+220
    def check():
        if time.monotonic()>deadline:raise TimeoutError('TE audit deadline')
    try:
        with torch.inference_mode():
            q,k,v=fixture(65,torch.float32,rank,108,h=4,hk=2)
            mask=mask_for(65,'bool',q.device,q.dtype)
            fp32=reference(q,k,v,mask)
            fp64=reference(q.cpu(),k.cpu(),v.cpu(),mask.cpu(),dtype=torch.float64)
            oracle=metrics(fp32.cpu(),fp64)
            torch.testing.assert_close(fp32.double().cpu(),fp64,rtol=2e-5,atol=2e-6)
            state['controls'].append(dict(kind='FP32_vs_CPU_FP64',**oracle))
            # Precision covers query/key tiling boundaries and the real N353 TE
            # geometry; RMS~1 follows Q/K normalization. .2 and 4 are diagnostics.
            for n,amplitude in ((65,1.),(129,1.),(353,.2),(353,1.),(353,4.),(513,1.),(1025,1.)):
                for dtype in (torch.float16,torch.float32):
                    for seed in (821,1709):
                        q,k,v=fixture(n,dtype,rank,seed,amplitude)
                        for kind in ('bool','additive','padding_bias'):
                            check();mask=mask_for(n,kind,q.device,dtype)
                            expected=reference(q,k,v,mask)
                            actual=ck.int8_attention(q,k,v,attn_mask=mask)
                            adapted=te.attention(q,k,v,mask).reshape(1,n,32,128).transpose(1,2)
                            assert torch.equal(actual,adapted),'Production adapter changed pinned output'
                            assert torch.isfinite(actual).all()
                            p=ck.prequantize_int8_attention(q,k,v,attn_mask=mask)
                            packed=ck.int8_attention_from_prequantized(p)
                            assert torch.equal(actual,packed),'Pinned raw/prequantized paths differ'
                            dq,dk,dv=dequantized_operands(p)
                            quantized_ref=reference(dq,dk,dv,mask)
                            store_dtype=torch.bfloat16 if dtype==torch.float32 else dtype
                            state['precision'].append(dict(n=n,h=32,hk=4,d=128,input_dtype=str(dtype),seed=seed,
                                amplitude=amplitude,mask=kind,cta_k=p.cta_k,output_dtype=str(actual.dtype),
                                matches_storage_lattice=torch.equal(actual,actual.to(store_dtype).to(actual.dtype)),
                                vs_FP32=metrics(actual,expected),row0_vs_FP32=metrics(actual[:,:,:1],expected[:,:,:1]),
                                ideal_storage_rounding_vs_FP32=metrics(expected.to(store_dtype).float(),expected),
                                dequantized_FP32_vs_FP32=metrics(quantized_ref,expected),
                                vs_dequantized_FP32=metrics(actual,quantized_ref),adapter_byte_equal=True))
                        save(output,state)
            # Keep lengths fixed for future-only edits. Prefix shortening is a
            # separate result because it can change tiling/anchor choices too.
            for dtype in (torch.float16,torch.float32):
                for seed in (821,1709):
                    q,k,v=fixture(353,dtype,rank,seed)
                    for kind in ('bool','additive'):
                        mask=mask_for(353,kind,q.device,dtype)
                        base=ck.int8_attention(q,k,v,attn_mask=mask)
                        base_ref=reference(q,k,v,mask)
                        p=ck.prequantize_int8_attention(q,k,v,attn_mask=mask)
                        dq,dk,dv=dequantized_operands(p)
                        for prefix in (31,128,257):
                            for index,name in enumerate(('q','k','v')):
                                for mutation in ('resample','scale32'):
                                    check();changed=[x.clone() for x in (q,k,v)]
                                    tail=changed[index][:,:,prefix:]
                                    if mutation=='scale32':tail.mul_(32)
                                    else:
                                        generator=torch.Generator(device=q.device).manual_seed(9053+index)
                                        tail.copy_(torch.randn(tail.shape,generator=generator,device=q.device,dtype=tail.dtype))
                                    assert all(torch.equal(x[:,:,:prefix],y[:,:,:prefix]) for x,y in zip((q,k,v),changed))
                                    actual=ck.int8_attention(*changed,attn_mask=mask)
                                    control=reference(*changed,mask)
                                    torch.testing.assert_close(control[:,:,:prefix],base_ref[:,:,:prefix],rtol=1e-6,atol=1e-6)
                                    cp=ck.prequantize_int8_attention(*changed,attn_mask=mask)
                                    cq,ckey,cv=dequantized_operands(cp)
                                    state['prefix'].append(dict(n=353,prefix=prefix,input_dtype=str(dtype),seed=seed,mask=kind,
                                        edit=name,mutation=mutation,mode='fixed_length_future_edit',
                                        output=metrics(actual[:,:,:prefix],base[:,:,:prefix]),
                                        FP32_control=metrics(control[:,:,:prefix],base_ref[:,:,:prefix]),
                                        dequantized_prefix_changes={nm:metrics(after[:,:,:prefix],before[:,:,:prefix])
                                            for nm,after,before in zip(('q','k','v'),(cq,ckey,cv),(dq,dk,dv))}))
                            short=tuple(x[:,:,:prefix].contiguous() for x in (q,k,v))
                            smask=mask_for(prefix,kind,q.device,dtype)
                            result=ck.int8_attention(*short,attn_mask=smask)
                            control=reference(*short,smask)
                            torch.testing.assert_close(control,base_ref[:,:,:prefix],rtol=2e-5,atol=2e-6)
                            state['prefix'].append(dict(n=353,prefix=prefix,input_dtype=str(dtype),seed=seed,mask=kind,
                                edit='qkv',mutation='truncate',mode='prefix_only_vs_full',
                                output=metrics(result,base[:,:,:prefix]),FP32_control=metrics(control,base_ref[:,:,:prefix])))
                            # Fixed packed scales/operands isolate score masking.
                            before=ck.int8_attention_from_prequantized(p)
                            for name in ('q','k','v'):
                                field=getattr(p,name).clone()
                                if name=='v':
                                    storage=field.reshape(1,4,128,-1)
                                    storage[...,value_storage_indices(353,rank)[prefix:]]=0
                                else:field[:,:,prefix:]=0
                                after=ck.int8_attention_from_prequantized(replace(p,**{name:field}))
                                assert torch.equal(before[:,:,:prefix],after[:,:,:prefix]),'Fixed packed future input leaked'
                                state['fixed_packed'].append(dict(input_dtype=str(dtype),seed=seed,mask=kind,prefix=prefix,
                                    edit=name,byte_equal=True))
                        save(output,state)
            state['strict_prefix_invariance']=all(r['output']['byte_equal'] for r in state['prefix'])
            state['FP32_input_outputs_on_BF16_lattice']=all(r['matches_storage_lattice'] for r in state['precision'] if r['input_dtype']=='torch.float32')
            state['status']='completed'
    except Exception:
        state.update(status='failed',error=traceback.format_exc());raise
    finally:state['seconds']=time.monotonic()-started;save(output,state)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--rank',type=int);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.rank is not None:return worker(args.rank,args.output)
    args.output.mkdir(parents=True,exist_ok=True)
    assert torch.cuda.device_count()==2
    state=dict(status='running',torch=torch.__version__,cuda=torch.version.cuda,
        packages={n:importlib.metadata.version(n) for n in ('comfy-kitchen','tilelang','numpy')},ranks=[])
    assert state['packages']['comfy-kitchen']=='0.2.37' and state['packages']['tilelang']=='0.1.13'
    procs=[];logs=[];started=time.monotonic()
    try:
        for rank in (0,1):
            log=(args.output/f'rank{rank}.log').open('w');logs.append(log)
            procs.append(subprocess.Popen([sys.executable,'-u',str(Path(__file__)), '--rank',str(rank),'--output',str(args.output/f'rank{rank}.json')],stdout=log,stderr=subprocess.STDOUT))
        for proc in procs:assert proc.wait(timeout=max(1,270-(time.monotonic()-started)))==0,'TE precision worker failed'
        state['ranks']=[json.loads((args.output/f'rank{rank}.json').read_text()) for rank in (0,1)]
        state['status']='completed'
    finally:
        for proc in procs:
            if proc.poll() is None:proc.kill();proc.wait()
        for log in logs:log.close()
        state['seconds']=time.monotonic()-started
        setup=Path('/kaggle/working/light_setup/delivery_binding.json')
        sources=Path(__file__).with_name('source_hashes.json')
        if setup.exists():state['delivery_binding']=json.loads(setup.read_text())
        if sources.exists():state['supplemental_sources']=json.loads(sources.read_text())
        save(args.output/'batch_results.json',state)
        if Path('/kaggle/working').exists():save('/kaggle/working/000_te_causal_precision.json',state)


if __name__=='__main__':main()
