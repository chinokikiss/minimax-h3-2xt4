"""Independent codec/dequantization contract; direct factories allow a local preview.

Only --local-preview retargets code generation. It makes no T4 performance claim.
"""
import argparse
import json
from pathlib import Path
import sys
sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parent)]
import torch
from patches import tilelang_transport_tradeoffs as old,tilelang_transport_p2 as p2,parallel_int8 as pi
from tp2_tradeoffs import transport_reference,byte_equal


def run():
    rows,width,k=17,256,7168
    records=[];torch.manual_seed(744)
    for pattern in ('random','zero','bounds','cancellation','ties_outliers'):
        parts=torch.randint(-10000000,10000000,(2,rows,width),device='cuda',dtype=torch.int32)
        if pattern=='zero':parts.zero_()
        if pattern=='bounds':parts.fill_(k*128*128)
        if pattern=='cancellation':parts[1]=-parts[0]+17
        if pattern=='ties_outliers':
            parts.zero_();parts[0,:,0]=16777217;parts[1,:,1]=-8388609
            parts[:,:,2:10]=torch.tensor([1,-1,3,-3,127,-127,65535,-65535],device='cuda',dtype=torch.int32)
        a=torch.rand(rows,1,device='cuda')*.1;w=torch.rand(width,device='cuda')*.001
        for scheme in ('fp16','int8_gather'):
            encoded=[p2.encode(part,scheme,k) for part in parts]
            payload=(encoded[0].float()+encoded[1].float()).half() if scheme=='fp16' else torch.cat(encoded,0)
            independent=transport_reference(parts,scheme,k)
            # Separate eager FP32 kernels guarantee left-associated products.
            reference=(independent.float()*a)*w
            native=pi.dequantize(independent,a,w,torch.float32)
            fused=p2.epilogue(rows,width,scheme,old.shift_for(k))(payload,a,w)
            assert byte_equal(native,reference),('native FP32 contract',pattern,scheme)
            assert byte_equal(fused,reference),('fused reconstruction + dequantization',pattern,scheme)
            records.append(dict(pattern=pattern,scheme=scheme,gate='bitwise_pass'))
    return records


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--local-preview',action='store_true');parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    if args.local_preview:
        import tilelang
        assert torch.cuda.get_device_capability()==(12,0)
        original=tilelang.compile
        def compile_preview(fn,**kw):kw['target']={'kind':'cuda','arch':'sm_120'};return original(fn,**kw)
        tilelang.compile=compile_preview
    else:assert torch.cuda.get_device_capability()==(7,5)
    records=run();args.output.write_text(json.dumps(dict(hardware=torch.cuda.get_device_name(),capability=torch.cuda.get_device_capability(),
        scope='Arithmetic preview only' if args.local_preview else 'Physical T4 independent arithmetic gates',records=records),indent=2))
    print('P2_EPILOGUE_UNITS',len(records),'pass',flush=True)
