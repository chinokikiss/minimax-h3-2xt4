"""Derive opt-in P2 kernels from the frozen experimental implementation.

The baseline file is never edited. Every replacement has an assertion so the
ablation's causal scope remains auditable in the generated source diff.
"""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
source=(ROOT/'patches/tilelang_attention_tradeoffs.py').read_text(encoding='utf8')


def replace(old,new):
    global source
    assert source.count(old)==1,(old,source.count(old));source=source.replace(old,new)


replace('pv_int8=False,compact_drop=False):','pv_int8=False,compact_drop=False,restore_unit=False,running_max=False,pooled_scores=False):')
replace("    scale=d**-.5*math.log2(math.e)","    scale=d**-.5*math.log2(math.e)\n    cs_shape=(b,h,np,nb) if pooled_scores else (1,)")
replace("             O:T.Tensor(output_shape,output_dtype)):","             CS:T.Tensor(cs_shape,'float32'),O:T.Tensor(output_shape,output_dtype)):")
replace("            if mode=='scalar':T.copy(Q[bz,hy,bx*bm:(bx+1)*bm,:],q16)","            if mode=='scalar' and not pooled_scores:T.copy(Q[bz,hy,bx*bm:(bx+1)*bm,:],q16)")
replace("for i in T.Parallel(bm):beta[i]=T.exp2(hm[i]-m[i])","for i in T.Parallel(bm):beta[i]=T.float32(1.) if running_max else T.exp2(hm[i]-m[i])")
replace("scores[i,j]=T.exp2(scores[i,j]-hm[i])","scores[i,j]=T.exp2(scores[i,j]-(m[i] if running_max else hm[i]))")
replace("for i,x in T.Parallel(bm,d):out[i,x]=out[i,x]*alpha[i]+T.cast(pv[i,x],'float32')*VS[bz,kh,kb,x]*T.float32(1./127.)*beta[i]", "for i,x in T.Parallel(bm,d):out[i,x]=out[i,x]*alpha[i]+(T.cast(pv[i,x],'float32')*T.float32(1./127.) if restore_unit else T.cast(pv[i,x],'float32')*VS[bz,kh,kb,x]*T.float32(1./127.)*beta[i])")
replace("""                    for i,x in T.Parallel(bm,d):products[i,x]=T.cast(q16[i,x],'float32')*T.cast(KC[bz,kh,kb,x],'float32')
                    T.reduce_sum(products,summary,dim=1);T.copy(summary,summary_shared)""", """                    if pooled_scores:
                        for i in T.Parallel(bm):summary_shared[i]=CS[bz,hy,bx*bm+i,kb]
                    else:
                        for i,x in T.Parallel(bm,d):products[i,x]=T.cast(q16[i,x],'float32')*T.cast(KC[bz,kh,kb,x],'float32')
                        T.reduce_sum(products,summary,dim=1);T.copy(summary,summary_shared)""")
replace('main,out_idx=[12]','main,out_idx=[13]')
replace('threads=128,warp_policy=\'square\',pv_int8=False,compact_drop=False,value=None,scales=None,lengths=None):',
        'threads=128,warp_policy=\'square\',pv_int8=False,compact_drop=False,value=None,scales=None,lengths=None,restore_unit=False,running_max=False,centroids=None):')
replace('threads,warp_policy,pv_int8,compact_drop)(p.q if scalar else dh', 'threads,warp_policy,pv_int8,compact_drop,restore_unit,running_max,centroids is not None)(p.q if scalar else dh')
replace('scales if pv_int8 else ds,lengths if compact_drop else dl)', 'scales if pv_int8 else ds,lengths if compact_drop else dl,centroids if centroids is not None else ds.reshape(-1)[:1])')
source+='''

@lru_cache(maxsize=32)
def centroid_scores(b,h,np,nb,d,bm=32,batch=4,omitted=False):
    """Batched leaf summaries; unchanged FP32 products/reduction math, new layout.

    FP32 reduction order can change. The complete operator must be checked
    against the same-route baseline; this is not bitwise-equivalence by fiat.
    """
    import tilelang
    import tilelang.language as T
    route_shape=(b,h,T.ceildiv(np,64),nb) if omitted else (1,)
    @T.prim_func
    def main(Q:T.Tensor((b,h,np,d),'float16'),KC:T.Tensor((b,h,nb,d),'float16'),Routes:T.Tensor(route_shape,'int32'),CS:T.Tensor((b,h,np,nb),'float32')):
        with T.Kernel(T.ceildiv(np,bm),T.ceildiv(nb,batch),h*b,threads=128) as (qx,kx,hb):
            q=T.alloc_shared((bm,d),'float16')
            product=T.alloc_fragment((bm,d),'float32');summary=T.alloc_fragment((bm,),'float32')
            T.copy(Q[hb//h,hb%h,qx*bm:(qx+1)*bm,:],q)
            for leaf in T.serial(batch):
                kb=kx*batch+leaf
                if kb<nb:
                    if not omitted or Routes[hb//h,hb%h,(qx*bm)//64,kb]==0:
                        for i,x in T.Parallel(bm,d):product[i,x]=T.cast(q[i,x],'float32')*T.cast(KC[hb//h,hb%h,kb,x],'float32')
                        T.reduce_sum(product,summary,dim=1)
                        for i in T.Parallel(bm):
                            if qx*bm+i<np:CS[hb//h,hb%h,qx*bm+i,kb]=summary[i]
    return tilelang.compile(main,out_idx=[3],target={'kind':'cuda','arch':'sm_75'},pass_configs={'tl.enable_async_copy':False})
'''
(ROOT/'patches/tilelang_attention_p2.py').write_text(source,encoding='utf8')
print('Generated opt-in P2 attention source')
