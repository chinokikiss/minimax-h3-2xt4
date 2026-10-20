"""Generate an isolated second pass from the frozen, measured T4 sources."""
import ast
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
BASE=Path('D:/attention_eval/operator_optimization_tp2_20261017/kitchen_inspired')


def main():
    hashes=json.loads((BASE/'submitted_sources.json').read_text())
    source=ROOT/'patches/tilelang_grouped_tp_kitchen.py'
    text=source.read_text(encoding='utf8');assert hashlib.sha256(text.encode()).hexdigest()==hashes['patches/tilelang_grouped_tp_kitchen.py']
    text=text.replace('scales=T.alloc_shared((32,),\'float32\')','scales=T.alloc_fragment((32,),\'float32\')')
    text=text.replace('def prepare(q,k,v,query_tile=32):','def prepare(q,k,v,query_tile=32,mean_mode=\'eager\'):')
    old="raw=F.pad(k.half().float(),(0,0,0,kp-nk)).reshape(b,hk,nb,64,d).sum(-2)/counts[None,None,:,None]"
    replacement="""if mean_mode not in ('eager','castpad'):raise ValueError('Unknown mean preparation')
    padded=(cast_pad_key(b,hk,nk,d,str(k.dtype).removeprefix('torch.'),*k.stride()[:3])(k)
            if mean_mode=='castpad' else F.pad(k.half().float(),(0,0,0,kp-nk)))
    raw=padded.reshape(b,hk,nb,64,d).sum(-2)/counts[None,None,:,None]"""
    assert old in text;text=text.replace(old,replacement)
    cast='''@lru_cache(maxsize=32)
def cast_pad_key(b,h,n,d,dtype,s0,s1,s2):
    import tilelang
    import tilelang.language as T
    padded=math.ceil(n/64)*64
    @T.prim_func
    def main(K:T.StridedTensor((b,h,n,d),(s0,s1,s2,1),dtype),P:T.Tensor((b,h,padded,d),'float32')):
        with T.Kernel(T.ceildiv(padded*d,512),h,b,threads=128) as (bx,hy,bz):
            for i in T.Parallel(512):
                index=bx*512+i
                if index<padded*d:
                    P[bz,hy,index//d,index%d]=T.if_then_else(index//d<n,
                        T.cast(T.cast(K[bz,hy,index//d,index%d],'float16'),'float32'),T.float32(0.))
    return tilelang.compile(main,out_idx=[1],target={'kind':'cuda','arch':'sm_75'},pass_configs={'tl.enable_async_copy':False})


'''
    text=text.replace('def prepare(q,k,v,query_tile=32,mean_mode=',cast+'def prepare(q,k,v,query_tile=32,mean_mode=')
    text=text.replace("threads=128,warp_policy='square'):","threads=128,warp_policy='square',row_scores=False,staged_scales=False):")
    annotation="""            if row_scores:
                row_fragment=T.alloc_fragment((bm,64),'float32')
                row_max=T.alloc_fragment((bm,),'float32');row_sum=T.alloc_fragment((bm,),'float32')
                exchange=T.alloc_shared((bm,64),'float32');row_exchange=T.alloc_shared((bm,),'float32')
                T.annotate_layout({row_fragment:T.Fragment((bm,64),
                    forward_thread_fn=lambda i,j:(i%(threads//32))*32+j%32,
                    forward_index_fn=lambda i,j:(i//(threads//32))*2+j//32)})
"""
    text=text.replace("            m=T.alloc_fragment((bm,),'float32');hm=",annotation+"            m=T.alloc_fragment((bm,),'float32');hm=")
    text=text.replace('T.copy(m,old);T.reduce_max(scores,hm,dim=1)', '''T.copy(m,old)
                    if row_scores:
                        T.copy(scores,exchange);T.copy(exchange,row_fragment)
                        T.reduce_max(row_fragment,row_max,dim=1)
                        T.copy(row_max,row_exchange);T.copy(row_exchange,hm)
                    else:T.reduce_max(scores,hm,dim=1)''')
    text=text.replace('T.reduce_sum(scores,psum,dim=1)', '''if row_scores:
                        T.copy(scores,exchange);T.copy(exchange,row_fragment)
                        T.reduce_sum(row_fragment,row_sum,dim=1)
                        T.copy(row_sum,row_exchange);T.copy(row_exchange,psum)
                    else:T.reduce_sum(scores,psum,dim=1)''')
    text=text.replace('                    T.clear(scores)\n','                    T.clear(scores)\n                    if staged_scales and not ck:T.copy(KS[bz,kh,kb*64:(kb+1)*64,:],kscale)\n')
    text=text.replace('*KS[bz,kh,kb*64+j,g]*scale','*(kscale[j,g] if staged_scales else KS[bz,kh,kb*64+j,g])*scale')
    text=text.replace('output_dtype=\'float16\',finite_output=False,threads=128,warp_policy=\'square\'):',
                      'output_dtype=\'float16\',finite_output=False,threads=128,warp_policy=\'square\',row_scores=False,staged_scales=False):')
    text=text.replace('layout,dtype,finite_output,threads,warp_policy)(', 'layout,dtype,finite_output,threads,warp_policy,row_scores,staged_scales)(')
    text=text.replace('"""SM75 Kitchen-inspired scheduling candidates.', '"""SM75 second-pass candidates: register scales, cast/pad fusion and fragment layout.')
    ast.parse(text)
    dest=ROOT/'patches/tilelang_grouped_tp_kitchen_v2.py';dest.write_text(text,encoding='utf8')
    print(dest)


if __name__=='__main__':main()
