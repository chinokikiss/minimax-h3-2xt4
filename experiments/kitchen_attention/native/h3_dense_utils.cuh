#pragma once
#include "attn_utils.cuh"

// Preserve the four integer dot products and each output's original FP32 FMA.
// The retained n8 helper shortens live V/partial fragments. SM75 tests decide
// whether that scheduling is useful; the original function remains the control.
template<bool H3_N8, uint32_t num_warps_q, uint32_t num_warps_k,
         uint32_t num_tiles_q, uint32_t num_tiles_k, uint32_t num_tiles_v,
         SwizzleMode swizzle_mode, uint32_t stride, bool pipeline_pv=true,
         typename DTypeSVAccum>
__device__ __forceinline__ void h3_compute_int8_sv(
    const smem_t<swizzle_mode,stride> &smem_V,
    int32_t RS_scale[][num_tiles_k][8],
    uint32_t RS_u8[][num_tiles_k/2][4],
    DTypeSVAccum RO[][num_tiles_v][8]) {
  if constexpr(H3_N8 && pipeline_pv && num_warps_k==1 && num_tiles_q==1 &&
               num_tiles_k==8 && num_tiles_v==8) {
    static_assert(std::is_same<DTypeSVAccum,float>::value);
    compute_int8_sv_pipelined_n8<swizzle_mode,stride>(smem_V,RS_scale,RS_u8,RO);
  } else {
    compute_int8_sv<num_warps_q,num_warps_k,num_tiles_q,num_tiles_k,num_tiles_v,
        swizzle_mode,stride,pipeline_pv>(smem_V,RS_scale,RS_u8,RO);
  }
}
