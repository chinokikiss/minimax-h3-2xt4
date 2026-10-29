// H3 output-store adaptation of pinned Kitchen 0.2.37; see vendor/NOTICE.
#include <cassert>
#include "h3_kitchen_attention.cuh"

#ifdef _WIN32
#define H3_EXPORT extern "C" __declspec(dllexport)
#else
#define H3_EXPORT extern "C" __attribute__((visibility("default")))
#endif

struct AttentionArgs {
  void *q, *k, *v, *o, *qs, *ks, *vs, *mask, *tile_bias;
  int64_t mb, mh, mq, mk;
  int b, h, hk, n, nk, d, ck, cq, mask_mode, mask_dtype, out_dim;
  float scale;
  void *stream;
};
static_assert(sizeof(AttentionArgs) == 160, "H3 attention ABI mismatch");

template<int D, int CK, int CQ, MaskMode MASK, bool FUSE, typename Offset>
cudaError_t launch_attention(const AttentionArgs &a) {
  constexpr int WQ = D >= 128 ? 16 : 32;
  constexpr int shared = (CQ + 2 * CK) * D;
  auto kernel = qk_int_sv_i8_attn_kernel<CQ, CK, WQ, CK, D, DataType::kInt8,
      QuantGranularity::kPerThread, QuantGranularity::kPerThread, float, false,
      float, ComputeUnit::kCudaCore, MASK, false, true, false, false, FUSE, Offset>;
  auto error = cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shared);
  if (error != cudaSuccess) return error;
  const Offset kp = div_ceil(a.nk, CK) * CK;
  kernel<<<dim3(div_ceil(a.n, CQ), a.h, a.b), dim3(32, CQ / WQ), shared,
           static_cast<cudaStream_t>(a.stream)>>>(
      (int8_t*)a.q, (int8_t*)a.k, (int8_t*)a.v, (float*)a.o, nullptr,
      (float*)a.qs, (float*)a.ks, (float*)a.vs, nullptr, a.mask,
      a.mb, a.mh, a.mq, a.mk, a.mask_dtype, a.n, a.nk, a.h / a.hk,
      Offset(a.h) * a.n * D, D, Offset(a.n) * D,
      Offset(a.hk) * a.nk * D, D, Offset(a.nk) * D,
      Offset(a.hk) * D * kp, D * kp, kp,
      Offset(a.h) * a.n * a.out_dim, a.out_dim, Offset(a.n) * a.out_dim, a.scale, (float*)a.tile_bias);
  return cudaGetLastError();
}

template<int D, typename Offset>
cudaError_t dispatch(const AttentionArgs &a) {
  if (a.mask_mode == 1) return launch_attention<D,64,128,MaskMode::kCustom,true,Offset>(a);
  if (a.mask_mode == 2) return launch_attention<D,64,128,MaskMode::kCustomKey,true,Offset>(a);
  if (a.mask_mode == 3) {
    if constexpr (D == 64)
      return launch_attention<D,64,128,MaskMode::kPreparedKey,true,Offset>(a);
    else {
#ifdef H3_LOCAL_SM120
      if constexpr (D == 128) {
        if (a.cq == 64) return launch_attention<D,128,64,MaskMode::kPreparedKey,true,Offset>(a);
      }
#endif
      return launch_attention<D,128,128,MaskMode::kPreparedKey,true,Offset>(a);
    }
  }
  if constexpr (D >= 128) {
    if (a.ck == 128) {
#ifdef H3_LOCAL_SM120
      if constexpr (D == 128) {
        if (a.cq == 64) return launch_attention<D,128,64,MaskMode::kNone,true,Offset>(a);
      }
#endif
      return launch_attention<D,128,128,MaskMode::kNone,true,Offset>(a);
    }
  }
  if (a.nk <= 512) return launch_attention<D,64,128,MaskMode::kNone,false,Offset>(a);
  return launch_attention<D,64,128,MaskMode::kNone,true,Offset>(a);
}

template<typename Offset>
cudaError_t dispatch_dimension(const AttentionArgs &a) {
  if (a.d == 64) return dispatch<64,Offset>(a);
  if (a.d == 128) return dispatch<128,Offset>(a);
  return dispatch<256,Offset>(a);
}

H3_EXPORT int h3_attention(const AttentionArgs *args) {
  const auto &a = *args;
  if (a.b <= 0 || a.h <= 0 || a.hk <= 0 || a.h % a.hk || a.n <= 0 || a.nk <= 0 ||
      (a.d != 64 && a.d != 128 && a.d != 256) || (a.ck != 64 && a.ck != 128) ||
      a.mask_mode < 0 || a.mask_mode > 3 || a.mask_dtype < 0 || a.mask_dtype > 3)
    return int(cudaErrorInvalidValue);
  if (a.out_dim <= 0 || a.out_dim > a.d) return int(cudaErrorInvalidValue);
  if ((a.mask_mode == 0) != (a.mask == nullptr) ||
      ((a.mask_mode == 3) != (a.tile_bias != nullptr))) return int(cudaErrorInvalidValue);
  if (a.mask_mode == 1 || a.mask_mode == 2) {
    if (a.ck != 64 || a.cq != 128) return int(cudaErrorInvalidValue);
  }
  if (a.d == 64 && (a.ck != 64 || a.cq != 128)) return int(cudaErrorInvalidValue);
  if (a.mask_mode == 3 && a.ck != (a.d == 64 ? 64 : 128)) return int(cudaErrorInvalidValue);
  if (uint64_t(a.n) * a.d > INT_MAX || uint64_t(a.nk + 127) * a.d > INT_MAX)
    return int(cudaErrorInvalidValue);
  const uint64_t qp = uint64_t(a.n) * a.d, kp = uint64_t(a.nk) * a.d;
  const uint64_t vp = uint64_t(div_ceil(a.nk,a.ck)) * a.ck * a.d;
  const bool wide = uint64_t(a.b - 1) * a.h * qp > UINT32_MAX ||
      uint64_t(a.b - 1) * a.hk * (vp > kp ? vp : kp) > UINT32_MAX ||
      uint64_t(a.h - 1) * qp > UINT32_MAX || uint64_t(a.hk - 1) * vp > UINT32_MAX;
  return int(wide ? dispatch_dimension<uint64_t>(a) : dispatch_dimension<uint32_t>(a));
}
