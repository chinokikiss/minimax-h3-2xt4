// Experimental SM75 extension. Frozen Kitchen headers retain their licenses.
#include <cassert>
#include "h3_attention.cuh"
#include <mma.h>
#include <cstdio>
#include <stdexcept>

#ifdef _WIN32
#define H3_EXPORT extern "C" __declspec(dllexport)
#else
#define H3_EXPORT extern "C" __attribute__((visibility("default")))
#endif

struct AttentionArgs {
  void *q, *k, *v, *o, *qs, *ks, *vs, *lse, *ids, *counts;
  int b, h, hk, n, nk, dim, cta_k, variant, causal, query_offset, dtype, capacity;
  float scale;
  void *stream;
};
static_assert(sizeof(AttentionArgs)==144, "Attention ctypes ABI mismatch");

template<int D, int CQ, int CK, bool ROUTED, bool CAUSAL, bool CACHE, typename Out,
         bool N8=false, typename Offset=uint64_t>
__global__ void h3_attention(AttentionArgs a) {
  constexpr int WQ = D == 128 ? 16 : 32;
  const uint64_t kp = div_ceil(a.nk, CK) * CK;
  qk_int_sv_i8_attn_body<false, CQ, CK, WQ, CK, D, DataType::kInt8,
      QuantGranularity::kPerThread, QuantGranularity::kPerThread, float,
      false, Out, ComputeUnit::kCudaCore, MaskMode::kNone, false, true,
      false, false, true, false, Offset, ROUTED, CAUSAL, CACHE, N8>(
      (int8_t*)a.q, (int8_t*)a.k, (int8_t*)a.v, (Out*)a.o, nullptr,
      (float*)a.qs, (float*)a.ks, (float*)a.vs, nullptr, nullptr, 0, 0, 0, 0, 0,
      a.n, a.nk, a.h / a.hk, uint64_t(a.h) * a.n * D, D, uint64_t(a.n) * D,
      uint64_t(a.hk) * a.nk * D, D, uint64_t(a.nk) * D,
      uint64_t(a.hk) * D * kp, D * kp, kp,
      uint64_t(a.h) * a.n * D, D, uint64_t(a.n) * D, a.scale, nullptr,
      (int*)a.ids, (int*)a.counts, a.capacity, a.query_offset, (float*)a.lse);
}

template<int D, int CQ, int CK, bool ROUTED, bool CAUSAL, bool CACHE, typename Out,
         bool N8=false, typename Offset=uint64_t>
cudaError_t launch_attention(const AttentionArgs &a) {
  constexpr int WQ = D == 128 ? 16 : 32;
  constexpr int shared = (CQ + 2 * CK) * D;
  auto fn = h3_attention<D, CQ, CK, ROUTED, CAUSAL, CACHE, Out, N8, Offset>;
  auto error = cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, shared);
  if (error != cudaSuccess) return error;
  fn<<<dim3(div_ceil(a.n, CQ), a.h, a.b), dim3(32, CQ / WQ), shared,
        (cudaStream_t)a.stream>>>(a);
  return cudaGetLastError();
}

template<int D, typename Out>
cudaError_t dispatch_attention(const AttentionArgs &a) {
  if (a.variant == 3) {
    if (a.causal) return launch_attention<D,64,64,true,true,false,Out>(a);
    return launch_attention<D,64,64,true,false,false,Out>(a);
  }
  if (a.cta_k == 64) {
    if (a.variant == 6) return launch_attention<D,128,64,false,false,false,Out,false,uint32_t>(a);
    if (a.variant == 1) return launch_attention<D,64,64,false,false,false,Out>(a);
    return launch_attention<D,128,64,false,false,false,Out>(a);
  }
  if constexpr (D == 128) {
    if (a.variant == 4) return launch_attention<D,128,128,false,false,false,Out,true>(a);
    if (a.variant == 5) return launch_attention<D,64,128,false,false,false,Out,true>(a);
    if (a.variant == 6) return launch_attention<D,128,128,false,false,false,Out,false,uint32_t>(a);
    if (a.variant == 7) return launch_attention<D,128,128,false,false,false,Out,true,uint32_t>(a);
    if (a.variant == 8) return launch_attention<D,64,128,false,false,false,Out,true,uint32_t>(a);
    if (a.variant == 2) return launch_attention<D,128,128,false,false,true,Out>(a);
    if (a.variant == 1) return launch_attention<D,64,128,false,false,false,Out>(a);
    return launch_attention<D,128,128,false,false,false,Out>(a);
  }
  return cudaErrorInvalidValue;
}

H3_EXPORT int h3_launch(const AttentionArgs *args) {
  const auto &a = *args;
  if (a.dim != 64 && a.dim != 128) return int(cudaErrorInvalidValue);
  if (a.h <= 0 || a.hk <= 0 || a.h % a.hk || a.b <= 0 || a.n <= 0 || a.nk <= 0)
    return int(cudaErrorInvalidValue);
  if (a.variant < 0 || a.variant > 8 || a.query_offset < 0 || a.dtype < 1 || a.dtype > 2)
    return int(cudaErrorInvalidValue);
  if ((a.causal || a.ids || a.counts) && (a.variant != 3 || a.cta_k != 64))
    return int(cudaErrorInvalidValue);
  if ((a.ids == nullptr) != (a.counts == nullptr)) return int(cudaErrorInvalidValue);
  if (a.variant == 2 && (a.dim != 128 || a.cta_k != 128)) return int(cudaErrorInvalidValue);
  if (a.cta_k != 64 && a.cta_k != 128) return int(cudaErrorInvalidValue);
  if (a.dim == 64 && a.cta_k != 64) return int(cudaErrorInvalidValue);
  if ((a.variant==4 || a.variant==5 || a.variant==7 || a.variant==8) &&
      (a.dim!=128 || a.cta_k!=128)) return int(cudaErrorInvalidValue);
  if (a.variant>=6) {
    const uint64_t padded=(uint64_t(a.nk)+a.cta_k-1)/a.cta_k*a.cta_k;
    const uint64_t extent=padded>uint64_t(a.n) ? padded : uint64_t(a.n);
    if (uint64_t(a.b)*a.h*extent*a.dim >= (uint64_t(1)<<32))
      return int(cudaErrorInvalidValue);
  }
  cudaError_t e;
  if (a.dim == 64) e = a.dtype == 1 ? dispatch_attention<64,half>(a) : dispatch_attention<64,nv_bfloat16>(a);
  else e = a.dtype == 1 ? dispatch_attention<128,half>(a) : dispatch_attention<128,nv_bfloat16>(a);
  return int(e);
}

// Pooled mode is a NEW approximation in Kitchen's rotated/centered quantized
// space. Fine and coarse logits share that space, avoiding a missing anchor-shift
// correction. It is not the previous Group32 Scalar numerical operator.
struct PoolArgs {
  void *q, *k, *v, *qs, *ks, *vs, *qh, *kc, *vc, *routes, *out, *lse;
  int b, h, hk, n, nk, dim, dtype;
  float scale;
  void *stream;
};
static_assert(sizeof(PoolArgs)==136, "Pool ctypes ABI mismatch");

__global__ void h3_pool_prepare_kv(PoolArgs a) {
  const int nb = div_ceil(a.nk,64), kp = nb * 64;
  const uint64_t x = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (x >= uint64_t(a.b) * a.hk * nb * a.dim) return;
  const int col = x % a.dim, leaf = (x / a.dim) % nb;
  const uint64_t bh = x / (uint64_t(a.dim) * nb);
  const int count = min(64, a.nk - leaf * 64);
  float k = 0, v = 0;
  for (int t=0; t<count; ++t) {
    const int token = leaf * 64 + t;
    const int physical = (token & ~15) | (token & 1) | (((token>>3)&1)<<1) |
        (((token>>1)&1)<<2) | (((token>>2)&1)<<3);
    k += float(((int8_t*)a.k)[(bh*a.nk+token)*a.dim+col]) *
        ((float*)a.ks)[bh*nb*4 + leaf*4 + (t%8)/2];
    v += float(((int8_t*)a.v)[(bh*a.dim+col)*kp+physical]) * ((float*)a.vs)[bh*a.dim+col];
  }
  ((half*)a.kc)[x] = __float2half_rn(k / count);
  ((half*)a.vc)[x] = __float2half_rn(v / count);
}

template<int D, typename Out>
__global__ void h3_pool_attention(PoolArgs a) {
  using namespace nvcuda;
  // One warp computes 16 queries against centroid tiles of 16 keys.
  __shared__ __align__(32) half q[16*D], k[16*D], v[16*D], p[256];
  __shared__ __align__(32) float scores[256], out[16*D];
  __shared__ float maximum[16], denom[16], alpha[16];
  const int lane = threadIdx.x, nb = div_ceil(a.nk,64), nq = div_ceil(a.n,64);
  const int start = blockIdx.x * 16, h = blockIdx.y, batch = blockIdx.z;
  const uint64_t bh = uint64_t(batch)*a.h+h;
  const uint64_t bkh = uint64_t(batch)*a.hk+h/(a.h/a.hk);
  for (int x=lane; x<16*D; x+=32) {
    const int token=start+x/D;
    const uint64_t scale_index=bh*div_ceil(a.n,128)*32+(token/32)*8+token%8;
    q[x] = token<a.n ? __float2half_rn(float(((int8_t*)a.q)[(bh*a.n+token)*D+x%D])*
        ((float*)a.qs)[scale_index]) : __float2half(0);
    out[x] = 0;
  }
  if (lane<16) { maximum[lane]=-CUDART_INF_F; denom[lane]=0; }
  __syncwarp();
  for (int first=0; first<nb; first+=16) {
    for (int x=lane; x<16*D; x+=32) {
      const int leaf=first+x/D;
      k[x]=leaf<nb ? ((half*)a.kc)[(bkh*nb+leaf)*D+x%D] : __float2half(0);
      v[x]=leaf<nb ? ((half*)a.vc)[(bkh*nb+leaf)*D+x%D] : __float2half(0);
    }
    __syncwarp();
    wmma::fragment<wmma::accumulator,16,16,16,float> dot;
    wmma::fill_fragment(dot,0.0f);
    for (int c=0;c<D;c+=16) {
      wmma::fragment<wmma::matrix_a,16,16,16,half,wmma::row_major> qa;
      wmma::fragment<wmma::matrix_b,16,16,16,half,wmma::col_major> kb;
      wmma::load_matrix_sync(qa,q+c,D); wmma::load_matrix_sync(kb,k+c,D);
      wmma::mma_sync(dot,qa,kb,dot);
    }
    wmma::store_matrix_sync(scores,dot,16,wmma::mem_row_major);
    __syncwarp();
    if (lane<16) {
      float top=-CUDART_INF_F;
      for(int j=0;j<16;++j) {
        const int leaf=first+j;
        const bool selected=leaf<nb && ((int*)a.routes)[(bh*nq+(start+lane)/64)*nb+leaf];
        const bool active=leaf<nb && start+lane<a.n && !selected;
        const float z=active ? scores[lane*16+j]*a.scale*math::log2e+
            log2f(float(min(64,a.nk-leaf*64))) : -CUDART_INF_F;
        scores[lane*16+j]=z; top=fmaxf(top,z);
      }
      const float next=fmaxf(maximum[lane],top);
      alpha[lane]=isfinite(next) ? exp2f(maximum[lane]-next) : 1.0f;
      float sum=0;
      for(int j=0;j<16;++j) {
        const float probability=isfinite(scores[lane*16+j]) ? exp2f(scores[lane*16+j]-next) : 0;
        p[lane*16+j]=__float2half_rn(probability); sum+=probability;
      }
      denom[lane]=denom[lane]*alpha[lane]+sum; maximum[lane]=next;
    }
    __syncwarp();
    for(int x=lane;x<16*D;x+=32) out[x]*=alpha[x/D];
    __syncwarp();
    wmma::fragment<wmma::matrix_a,16,16,16,half,wmma::row_major> prob;
    wmma::load_matrix_sync(prob,p,16);
    for(int c=0;c<D;c+=16) {
      wmma::fragment<wmma::matrix_b,16,16,16,half,wmma::row_major> vb;
      wmma::fragment<wmma::accumulator,16,16,16,float> result;
      wmma::load_matrix_sync(vb,v+c,D);
      wmma::load_matrix_sync(result,out+c,D,wmma::mem_row_major);
      wmma::mma_sync(result,prob,vb,result);
      wmma::store_matrix_sync(out+c,result,D,wmma::mem_row_major);
    }
    __syncwarp();
  }
  // Merge into the fine output in this same tile. No full Q-half, coarse output,
  // centroid-score matrix, or separate merge launch is materialized.
  for(int x=lane;x<16*D;x+=32) if(start+x/D<a.n && denom[x/D]>0) {
    const uint64_t row=bh*a.n+start+x/D, index=row*D+x%D;
    const float lf=((float*)a.lse)[row];
    const float lc=maximum[x/D]+log2f(denom[x/D]), top=fmaxf(lf,lc);
    const float wf=isfinite(lf) ? exp2f(lf-top) : 0, wc=exp2f(lc-top);
    const float merged=(float(((Out*)a.out)[index])*wf+(out[x]/denom[x/D])*wc)/(wf+wc);
    ((Out*)a.out)[index]=Out(merged);
  }
}

H3_EXPORT int h3_pool(const PoolArgs *args) {
  const auto &a=*args;
  if ((a.dim!=64 && a.dim!=128) || (a.dtype!=1 && a.dtype!=2)) return int(cudaErrorInvalidValue);
  auto stream=(cudaStream_t)a.stream;
  h3_pool_prepare_kv<<<div_ceil(uint64_t(a.b)*a.hk*div_ceil(a.nk,64)*a.dim,256),256,0,stream>>>(a);
  auto error=cudaGetLastError(); if(error!=cudaSuccess) return int(error);
  auto grid=dim3(div_ceil(a.n,16),a.h,a.b);
  if(a.dim==64) {
    if(a.dtype==1) h3_pool_attention<64,half><<<grid,32,0,stream>>>(a);
    else h3_pool_attention<64,nv_bfloat16><<<grid,32,0,stream>>>(a);
  } else {
    if(a.dtype==1) h3_pool_attention<128,half><<<grid,32,0,stream>>>(a);
    else h3_pool_attention<128,nv_bfloat16><<<grid,32,0,stream>>>(a);
  }
  return int(cudaGetLastError());
}

__global__ void h3_compact_kernel(const int *routes, int *ids, int *counts, int nb) {
  const uint64_t row=blockIdx.x;
  const int lane=threadIdx.x;
  int used=0;
  for(int first=0;first<nb;first+=32) {
    const int leaf=first+lane;
    const bool keep=leaf<nb && routes[row*nb+leaf]!=0;
    const unsigned votes=__ballot_sync(0xffffffff,keep);
    if(keep) ids[row*nb+used+__popc(votes&((1u<<lane)-1))]=leaf;
    used+=__popc(votes);
  }
  if(lane==0) counts[row]=used;
}

H3_EXPORT int h3_compact(const void *routes, void *ids, void *counts, int rows,
                       int nb, void *stream) {
  if(rows<=0 || nb<=0) return int(cudaErrorInvalidValue);
  h3_compact_kernel<<<rows,32,0,(cudaStream_t)stream>>>((int*)routes,(int*)ids,(int*)counts,nb);
  return int(cudaGetLastError());
}

H3_EXPORT const char *h3_error(int code) { return cudaGetErrorString(cudaError_t(code)); }
