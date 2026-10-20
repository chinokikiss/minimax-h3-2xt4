"""Explicit SM120 codec/compaction unit preview; no T4 performance inference."""
import json
from pathlib import Path
import sys
sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parent)]
import torch
import tilelang
from patches import tilelang_transport_tradeoffs as c,tilelang_attention_tradeoffs as a
from tp2_tradeoffs import transport_reference,pv_reference
from patches.tilelang_grouped_tp_optimized import TPGrouped


def main():
    assert torch.cuda.get_device_capability()==(12,0)
    original=tilelang.compile
    def compile_preview(fn,**kw):
        kw['target']={'kind':'cuda','arch':'sm_120'}
        return original(fn,**kw)
    tilelang.compile=compile_preview;torch.set_grad_enabled(False);torch.backends.cuda.matmul.allow_tf32=False
    torch.manual_seed(818);records=[]
    for pattern in ('normal','zero','bounds','cancellation'):
        parts=torch.randint(-10000000,10000000,(2,17,256),device='cuda',dtype=torch.int32)
        if pattern=='zero':parts.zero_()
        if pattern=='bounds':parts[0].fill_(7168*127*127);parts[1].fill_(7168*127*127)
        if pattern=='cancellation':parts[1]=-parts[0]+17
        shift=c.shift_for(7168)
        payloads=[c.fp16_codec(17,256,shift)(x) for x in parts]
        total=(payloads[0].float()+payloads[1].float()).half()
        decoded=c.fp16_codec(17,256,shift,True)(total)
        assert torch.equal(decoded,transport_reference(parts,'fp16',7168))
        packed=[c.pack_int8(17,256)(x) for x in parts]
        decoded=c.unpack_int8(17,256)(torch.cat(packed,0))
        assert torch.equal(decoded,transport_reference(parts,'int8_gather',7168)),pattern
        records.append(dict(pattern=pattern,fp16='pass',packed_int8='pass'))
    for kind in ('empty','full','alternating','random'):
        route=torch.randint(0,2,(1,2,3,107),device='cuda',dtype=torch.int32)
        if kind=='empty':route.zero_()
        elif kind=='full':route.fill_(1)
        elif kind=='alternating':route[:]=torch.arange(107,device='cuda').remainder(2)
        ids,lengths=a.route_compactor(1,2,3,107)(route)
        expected=torch.where(route.bool(),torch.arange(107,device='cuda'),107).sort(-1).values.int()
        assert torch.equal(ids,expected) and torch.equal(lengths,route.sum(-1).int())
        records.append(dict(compaction=kind,gate='pass'))
    # Independent operands for a small direct INT8-PV invocation. This preview
    # changes only the compilation target; hardware identity is never spoofed.
    h,hk,n,nk,d=2,1,65,65,64;np=kp=128
    qi=torch.randint(-127,128,(1,h,np,d),device='cuda',dtype=torch.int8)
    ki=torch.randint(-127,128,(1,hk,kp,d),device='cuda',dtype=torch.int8)
    qs=torch.full((1,h,np,d//32),.001,device='cuda');ks=torch.full((1,hk,kp,d//32),.001,device='cuda')
    value=torch.randn(1,hk,kp,d,device='cuda').half()*.2;value[:,:,nk:]=0
    vi,vs=a.value_quantizer(1,hk,kp,d)(value)
    values=value.float().reshape(1,hk,kp//64,64,d)
    scales=(values.abs().amax(-2)*torch.tensor(1./127.,device='cuda')).clamp_min(1e-8)
    assert torch.equal(vs,scales) and torch.equal(vi,(values/scales.unsqueeze(-2)).round().clamp(-127,127).to(torch.int8).reshape_as(vi))
    dh=torch.empty(1,device='cuda',dtype=torch.float16);di=torch.empty(1,device='cuda',dtype=torch.int32)
    counts=torch.tensor([64,1],device='cuda',dtype=torch.int32)
    p=TPGrouped(dh,qi,ki,value,qs,ks,dh,dh,counts,n,nk,'dense',32)
    lengths=torch.empty(1,h,2,device='cuda',dtype=torch.int32)
    actual=a.kernel(1,h,hk,n,nk,d,query_tile=32,layout='BND',pv_int8=True)(dh,qi,ki,vi,qs,ks,dh,dh,di,di,vs,lengths)
    expected=pv_reference(p,vi,vs)
    torch.testing.assert_close(actual,expected,rtol=.005,atol=.002)
    records.append(dict(INT8_PV_independent_reference='pass'))
    output=Path('D:/attention_eval/operator_optimization_tp2_20261018/tradeoffs');output.mkdir(parents=True,exist_ok=True)
    (output/'local_units.json').write_text(json.dumps(dict(hardware=torch.cuda.get_device_name(),capability=[12,0],scope='Unit arithmetic preview only; explicit SM120 codegen, no T4 performance data',records=records),indent=2))
    print('EXPLICIT_SM120_UNITS',len(records),'pass')


if __name__=='__main__':main()
