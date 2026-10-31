#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>

#include <cstdint>
#include <climits>

namespace star7_sol_preprocess {

constexpr int kBlock = 64;
constexpr float kLog2E = 1.4426950408889634f;

struct InputStrides { std::int64_t batch, head, token; };

__device__ __forceinline__ bool token_intervals_near(
    int k0, int k1, int t0, int t1, int halo) {
  return k0 < t1 + halo && k1 > t0 - halo;
}

__device__ __forceinline__ float warp_max(float value) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1)
    value = fmaxf(value, __shfl_xor_sync(0xffffffff, value, offset));
  return value;
}

template <int HEAD_DIM>
__global__ void reduce_qkv_centroids(
    const half *__restrict__ q, const half *__restrict__ k,
    const half *__restrict__ v, half *__restrict__ qc,
    half *__restrict__ kc, half *__restrict__ vc, int heads, int length,
    int blocks, int padded_blocks) {
  const int linear = static_cast<int>(blockIdx.x);
  const int bh = linear / padded_blocks;
  const int block = linear - bh * padded_blocks;
  const int dim = static_cast<int>(threadIdx.x);
  if (dim >= HEAD_DIM) return;
  const std::int64_t summary_index =
      (static_cast<std::int64_t>(bh) * padded_blocks + block) * HEAD_DIM + dim;
  if (block >= blocks) {
    qc[summary_index] = __float2half(0.0f);
    kc[summary_index] = __float2half(0.0f);
    vc[summary_index] = __float2half(0.0f);
    return;
  }
  const int start = block * kBlock;
  const int count = min(kBlock, length - start);
  const std::int64_t base =
      static_cast<std::int64_t>(bh) * length * HEAD_DIM;
  float q_sum = 0.0f;
  float k_sum = 0.0f;
  float v_sum = 0.0f;
#pragma unroll
  for (int token = 0; token < kBlock; ++token) {
    if (token < count) {
      const std::int64_t index = base +
          static_cast<std::int64_t>(start + token) * HEAD_DIM + dim;
      q_sum += __half2float(q[index]);
      k_sum += __half2float(k[index]);
      v_sum += __half2float(v[index]);
    }
  }
  const float inverse = 1.0f / static_cast<float>(count);
  qc[summary_index] = __float2half_rn(q_sum * inverse);
  kc[summary_index] = __float2half_rn(k_sum * inverse);
  // Store the mean.  The attention kernel adds log2(block_length), which is
  // mathematically identical to NVIDIA Sol's summed V centroid.
  vc[summary_index] = __float2half_rn(v_sum * inverse);
}

template <int HEAD_DIM>
__global__ void reduce_kc_stats(
    const half *__restrict__ kc, float *__restrict__ mean,
    float *__restrict__ variance, int blocks, int padded_blocks) {
  const int bh = static_cast<int>(blockIdx.x);
  const int dim = static_cast<int>(threadIdx.x);
  if (dim >= HEAD_DIM) return;
  float sum = 0.0f;
  float square_sum = 0.0f;
  const std::int64_t base =
      static_cast<std::int64_t>(bh) * padded_blocks * HEAD_DIM;
  for (int block = 0; block < blocks; ++block) {
    const float value = __half2float(kc[base + block * HEAD_DIM + dim]);
    sum += value;
    square_sum = fmaf(value, value, square_sum);
  }
  const float inverse = 1.0f / static_cast<float>(blocks);
  const float average = sum * inverse;
  mean[static_cast<std::int64_t>(bh) * HEAD_DIM + dim] = average;
  variance[static_cast<std::int64_t>(bh) * HEAD_DIM + dim] =
      fmaxf(square_sum * inverse - average * average, 0.0f);
}

// Preserve centroid accumulation order and the existing Q16/K64 quantizer.
template <int HEAD_DIM>
__global__ void reduce_quantized_qkv(
    const half *q, const half *k, const half *v, half *qc, half *kc, half *vc,
    std::int8_t *q8, std::int8_t *k8, float *qs, float *ks,
    int heads, int length, int blocks, int padded_blocks,
    InputStrides q_stride, InputStrides k_stride, InputStrides v_stride) {
  static_assert(HEAD_DIM == 64 || HEAD_DIM == 128, "Sol preprocessing supports D64/D128");
  constexpr int HEAD_WARPS = HEAD_DIM / 32;
  __shared__ float q_maxima[4][4];
  __shared__ float k_maxima[4];
  // Q/K are reused after the scale reduction. Cache the one global read for
  // the exact FP16 values so quantization never rereads strided BHLD input.
  __shared__ half q_cache[kBlock * HEAD_DIM];
  __shared__ half k_cache[kBlock * HEAD_DIM];
  const int bh = blockIdx.x / padded_blocks;
  const int block = blockIdx.x % padded_blocks;
  const int dim = threadIdx.x;
  const std::int64_t summary =
      (static_cast<std::int64_t>(bh) * padded_blocks + block) * HEAD_DIM + dim;
  if (block >= blocks) {
    qc[summary] = kc[summary] = vc[summary] = __float2half(0.0f);
    return;
  }
  const int start = block * kBlock;
  const int count = min(kBlock, length - start);
  const std::int64_t base = static_cast<std::int64_t>(bh) * length * HEAD_DIM;
  const int batch = bh / heads, head = bh % heads;
  const std::int64_t q_base = batch * q_stride.batch + head * q_stride.head + dim;
  const std::int64_t k_base = batch * k_stride.batch + head * k_stride.head + dim;
  const std::int64_t v_base = batch * v_stride.batch + head * v_stride.head + dim;
  float q_sum = 0.0f, k_sum = 0.0f, v_sum = 0.0f;
  float q_max[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  float k_max = 0.0f;
  for (int token = 0; token < kBlock; ++token) {
    if (token < count) {
      const half q_half = q[q_base + (start + token) * q_stride.token];
      const half k_half = k[k_base + (start + token) * k_stride.token];
      q_cache[token * HEAD_DIM + dim] = q_half;
      k_cache[token * HEAD_DIM + dim] = k_half;
      const float q_value = __half2float(q_half);
      const float k_value = __half2float(k_half);
      q_sum += q_value;
      k_sum += k_value;
      v_sum += __half2float(v[v_base + (start + token) * v_stride.token]);
      q_max[token / 16] = fmaxf(q_max[token / 16], fabsf(q_value));
      k_max = fmaxf(k_max, fabsf(k_value));
    }
  }
  const float inverse = 1.0f / count;
  qc[summary] = __float2half_rn(q_sum * inverse);
  kc[summary] = __float2half_rn(k_sum * inverse);
  vc[summary] = __float2half_rn(v_sum * inverse);
  const int lane = dim & 31;
  const int warp = dim >> 5;
#pragma unroll
  for (int group = 0; group < 4; ++group) {
    const float maximum = warp_max(q_max[group]);
    if (lane == 0) q_maxima[group][warp] = maximum;
  }
  k_max = warp_max(k_max);
  if (lane == 0) k_maxima[warp] = k_max;
  __syncthreads();
  if (warp == 0) {
#pragma unroll
    for (int group = 0; group < 4; ++group) {
      const float maximum = warp_max(lane < HEAD_WARPS ? q_maxima[group][lane] : 0.0f);
      if (lane == 0) q_maxima[group][0] = fmaxf(maximum / 127.0f, 1.0e-8f);
    }
    const float maximum = warp_max(lane < HEAD_WARPS ? k_maxima[lane] : 0.0f);
    if (lane == 0) k_maxima[0] = fmaxf(maximum / 127.0f, 1.0e-8f);
  }
  __syncthreads();
  if (dim < 4) {
    const float scale = start + dim * 16 < length ? q_maxima[dim][0] : 1.0f;
    qs[(static_cast<std::int64_t>(bh) * blocks + block) * 4 + dim] = scale;
  }
  if (dim == 0) ks[static_cast<std::int64_t>(bh) * blocks + block] = k_maxima[0];
  for (int token = 0; token < kBlock; ++token) {
    if (token < count) {
      const std::int64_t index = base + (start + token) * HEAD_DIM + dim;
      q8[index] = static_cast<std::int8_t>(max(-127, min(127,
          __float2int_rn(__half2float(q_cache[token * HEAD_DIM + dim]) /
                         q_maxima[token / 16][0]))));
      k8[index] = static_cast<std::int8_t>(max(-127, min(127,
          __float2int_rn(__half2float(k_cache[token * HEAD_DIM + dim]) /
                         k_maxima[0]))));
    }
  }
}


template <int HEAD_DIM>
__global__ void reduce_quantized_q_local(
    const half *q, half *qc, std::int8_t *q8, float *qs,
    int heads, int q_length, int q_blocks, int q_padded_blocks,
    InputStrides q_stride) {
  static_assert(HEAD_DIM == 64 || HEAD_DIM == 128,
                "Sol local-Q preprocessing supports D64/D128");
  constexpr int HEAD_WARPS = HEAD_DIM / 32;
  __shared__ float q_maxima[4][4];
  __shared__ half q_cache[kBlock * HEAD_DIM];
  const int bh = blockIdx.x / q_padded_blocks;
  const int block = blockIdx.x % q_padded_blocks;
  const int dim = threadIdx.x;
  const std::int64_t summary =
      (static_cast<std::int64_t>(bh) * q_padded_blocks + block) * HEAD_DIM + dim;
  if (block >= q_blocks) {
    qc[summary] = __float2half(0.0f);
    return;
  }
  const int start = block * kBlock;
  const int count = min(kBlock, q_length - start);
  const int batch = bh / heads, head = bh % heads;
  const std::int64_t q_base =
      batch * q_stride.batch + head * q_stride.head + dim;
  float q_sum = 0.0f;
  float q_max[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  for (int token = 0; token < kBlock; ++token) {
    if (token < count) {
      const half value_half = q[q_base + (start + token) * q_stride.token];
      q_cache[token * HEAD_DIM + dim] = value_half;
      const float value = __half2float(value_half);
      q_sum += value;
      q_max[token / 16] = fmaxf(q_max[token / 16], fabsf(value));
    }
  }
  qc[summary] = __float2half_rn(q_sum / static_cast<float>(count));
  const int lane = dim & 31, warp = dim >> 5;
#pragma unroll
  for (int group = 0; group < 4; ++group) {
    const float maximum = warp_max(q_max[group]);
    if (lane == 0) q_maxima[group][warp] = maximum;
  }
  __syncthreads();
  if (warp == 0) {
#pragma unroll
    for (int group = 0; group < 4; ++group) {
      const float maximum =
          warp_max(lane < HEAD_WARPS ? q_maxima[group][lane] : 0.0f);
      if (lane == 0)
        q_maxima[group][0] = fmaxf(maximum / 127.0f, 1.0e-8f);
    }
  }
  __syncthreads();
  if (dim < 4) {
    const float scale = start + dim * 16 < q_length
        ? q_maxima[dim][0] : 1.0f;
    qs[(static_cast<std::int64_t>(bh) * q_blocks + block) * 4 + dim] = scale;
  }
  const std::int64_t out_base =
      static_cast<std::int64_t>(bh) * q_length * HEAD_DIM;
  for (int token = 0; token < kBlock; ++token) {
    if (token < count) {
      const std::int64_t index =
          out_base + static_cast<std::int64_t>(start + token) * HEAD_DIM + dim;
      q8[index] = static_cast<std::int8_t>(max(-127, min(127,
          __float2int_rn(__half2float(q_cache[token * HEAD_DIM + dim]) /
                         q_maxima[token / 16][0]))));
    }
  }
}

template <int HEAD_DIM>
__global__ void reduce_quantized_kv_full(
    const half *k, const half *v, half *kc, half *vc,
    std::int8_t *k8, float *ks, int heads, int length,
    int blocks, int padded_blocks, InputStrides k_stride,
    InputStrides v_stride) {
  static_assert(HEAD_DIM == 64 || HEAD_DIM == 128,
                "Sol KV preprocessing supports D64/D128");
  constexpr int HEAD_WARPS = HEAD_DIM / 32;
  __shared__ float k_maxima[4];
  __shared__ half k_cache[kBlock * HEAD_DIM];
  const int bh = blockIdx.x / padded_blocks;
  const int block = blockIdx.x % padded_blocks;
  const int dim = threadIdx.x;
  const std::int64_t summary =
      (static_cast<std::int64_t>(bh) * padded_blocks + block) * HEAD_DIM + dim;
  if (block >= blocks) {
    kc[summary] = vc[summary] = __float2half(0.0f);
    return;
  }
  const int start = block * kBlock;
  const int count = min(kBlock, length - start);
  const int batch = bh / heads, head = bh % heads;
  const std::int64_t k_base =
      batch * k_stride.batch + head * k_stride.head + dim;
  const std::int64_t v_base =
      batch * v_stride.batch + head * v_stride.head + dim;
  float k_sum = 0.0f, v_sum = 0.0f, k_max = 0.0f;
  for (int token = 0; token < kBlock; ++token) {
    if (token < count) {
      const half k_half = k[k_base + (start + token) * k_stride.token];
      k_cache[token * HEAD_DIM + dim] = k_half;
      const float k_value = __half2float(k_half);
      k_sum += k_value;
      v_sum += __half2float(v[v_base + (start + token) * v_stride.token]);
      k_max = fmaxf(k_max, fabsf(k_value));
    }
  }
  const float inverse = 1.0f / static_cast<float>(count);
  kc[summary] = __float2half_rn(k_sum * inverse);
  vc[summary] = __float2half_rn(v_sum * inverse);
  const int lane = dim & 31, warp = dim >> 5;
  k_max = warp_max(k_max);
  if (lane == 0) k_maxima[warp] = k_max;
  __syncthreads();
  if (warp == 0) {
    const float maximum =
        warp_max(lane < HEAD_WARPS ? k_maxima[lane] : 0.0f);
    if (lane == 0) k_maxima[0] = fmaxf(maximum / 127.0f, 1.0e-8f);
  }
  __syncthreads();
  if (dim == 0)
    ks[static_cast<std::int64_t>(bh) * blocks + block] = k_maxima[0];
  const std::int64_t out_base =
      static_cast<std::int64_t>(bh) * length * HEAD_DIM;
  for (int token = 0; token < kBlock; ++token) {
    if (token < count) {
      const std::int64_t index =
          out_base + static_cast<std::int64_t>(start + token) * HEAD_DIM + dim;
      k8[index] = static_cast<std::int8_t>(max(-127, min(127,
          __float2int_rn(__half2float(k_cache[token * HEAD_DIM + dim]) /
                         k_maxima[0]))));
    }
  }
}


__global__ void route_statistics(const std::int32_t *counts,
                                 std::int64_t *stats, int rows) {
  __shared__ std::int64_t totals[256];
  __shared__ int minima[256], maxima[256];
  const int lane = threadIdx.x;
  std::int64_t total = 0;
  int low = INT_MAX, high = 0;
  for (int row = lane; row < rows; row += blockDim.x) {
    const int value = counts[row];
    total += value;
    low = min(low, value);
    high = max(high, value);
  }
  totals[lane] = total; minima[lane] = low; maxima[lane] = high;
  __syncthreads();
  for (int stride = 128; stride > 0; stride >>= 1) {
    if (lane < stride) {
      totals[lane] += totals[lane + stride];
      minima[lane] = min(minima[lane], minima[lane + stride]);
      maxima[lane] = max(maxima[lane], maxima[lane + stride]);
    }
    __syncthreads();
  }
  if (lane == 0) {
    stats[0] = minima[0]; stats[1] = maxima[0]; stats[2] = totals[0];
  }
}

template <int HEAD_DIM>
__global__ void compute_diag_threshold(
    const half *__restrict__ qc, const float *__restrict__ kc_mean,
    const float *__restrict__ kc_variance, float *__restrict__ threshold,
    int q_blocks, int padded_blocks, float tau, float scale_log2) {
  __shared__ float mean_parts[HEAD_DIM];
  __shared__ float variance_parts[HEAD_DIM];
  const int route = static_cast<int>(blockIdx.x);
  const int bh = route / q_blocks;
  const int q_block = route - bh * q_blocks;
  const int dim = static_cast<int>(threadIdx.x);
  const std::int64_t centroid_index =
      (static_cast<std::int64_t>(bh) * padded_blocks + q_block) * HEAD_DIM + dim;
  const std::int64_t stat_index =
      static_cast<std::int64_t>(bh) * HEAD_DIM + dim;
  const float q_value = __half2float(qc[centroid_index]);
  mean_parts[dim] = q_value * kc_mean[stat_index];
  variance_parts[dim] = q_value * q_value * kc_variance[stat_index];
  __syncthreads();
  for (int stride = HEAD_DIM / 2; stride > 0; stride >>= 1) {
    if (dim < stride) {
      mean_parts[dim] += mean_parts[dim + stride];
      variance_parts[dim] += variance_parts[dim + stride];
    }
    __syncthreads();
  }
  if (dim == 0) {
    const float mean = mean_parts[0] * scale_log2;
    const float variance = variance_parts[0] * scale_log2 * scale_log2;
    threshold[route] = mean + tau * sqrtf(fmaxf(variance, 0.0f) + 1.0e-6f);
  }
}

template <int HEAD_DIM>
__global__ void route_centroid_tiles(
    const half *__restrict__ qc, const half *__restrict__ kc,
    const float *__restrict__ threshold, std::uint8_t *__restrict__ exact_mask,
    std::int32_t *__restrict__ row_count, int heads, int q_blocks,
    int k_blocks, int padded_blocks, float scale_log2,
    int sink_start_block, int sink_end_block,
    int topology_start_token, int topology_end_token,
    int frame_stride_tokens, int temporal_radius, int local_radius_tokens,
    int protect_frame_boundaries, int accumulate_row_count) {
#if __CUDA_ARCH__ >= 700
  using namespace nvcuda;
  __shared__ float score_tile[16 * 16];
  const int k_tile = static_cast<int>(blockIdx.x);
  const int q_tile = static_cast<int>(blockIdx.y);
  const int bh = static_cast<int>(blockIdx.z);
  const half *a = qc +
      (static_cast<std::int64_t>(bh) * padded_blocks + q_tile * 16) * HEAD_DIM;
  // Row-major [16,HEAD_DIM] is the same byte layout as col-major [HEAD_DIM,16].
  const half *b = kc +
      (static_cast<std::int64_t>(bh) * padded_blocks + k_tile * 16) * HEAD_DIM;
  wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> af;
  wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> bf;
  wmma::fragment<wmma::accumulator, 16, 16, 16, float> cf;
  wmma::fill_fragment(cf, 0.0f);
#pragma unroll
  for (int dim = 0; dim < HEAD_DIM; dim += 16) {
    wmma::load_matrix_sync(af, a + dim, HEAD_DIM);
    wmma::load_matrix_sync(bf, b + dim, HEAD_DIM);
    wmma::mma_sync(cf, af, bf, cf);
  }
  wmma::store_matrix_sync(score_tile, cf, 16, wmma::mem_row_major);
  __syncthreads();
  for (int item = static_cast<int>(threadIdx.x); item < 256; item += 32) {
    const int local_q = item / 16;
    const int local_k = item - local_q * 16;
    const int q_block = q_tile * 16 + local_q;
    const int k_block = k_tile * 16 + local_k;
    bool selected = false;
    if (q_block < q_blocks && k_block < k_blocks) {
      const std::int64_t row = static_cast<std::int64_t>(bh) * q_blocks + q_block;
      const bool neighbor = abs(q_block - k_block) <= 1;
      const bool protected_key =
          k_block >= sink_start_block && k_block < sink_end_block;
      const bool protected_query =
          q_block >= sink_start_block && q_block < sink_end_block;

      const int q_start = q_block * kBlock;
      const int q_end = min(q_start + kBlock, topology_end_token);
      const int k_start = k_block * kBlock;
      const int k_end = min(k_start + kBlock, topology_end_token);
      const bool topology_enabled =
          topology_start_token >= 0 && topology_end_token > topology_start_token &&
          frame_stride_tokens > 0;
      const bool q_video = topology_enabled &&
          q_start < topology_end_token && q_start + kBlock > topology_start_token;
      const bool k_video = topology_enabled &&
          k_start < topology_end_token && k_start + kBlock > topology_start_token;

      bool topology_neighbor = false;
      if (q_video && k_video) {
        const int q0 = max(q_start, topology_start_token);
        const int q1 = min(q_start + kBlock, topology_end_token);
        const int k0 = max(k_start, topology_start_token);
        const int k1 = min(k_start + kBlock, topology_end_token);
        topology_neighbor =
            token_intervals_near(k0, k1, q0, q1, local_radius_tokens);
        for (int dt = 1; dt <= temporal_radius && !topology_neighbor; ++dt) {
          const int shift = dt * frame_stride_tokens;
          const int prev0 = q0 - shift, prev1 = q1 - shift;
          const int next0 = q0 + shift, next1 = q1 + shift;
          if ((prev1 > topology_start_token &&
               token_intervals_near(
                   k0, k1, max(prev0, topology_start_token),
                   min(prev1, topology_end_token), local_radius_tokens)) ||
              (next0 < topology_end_token &&
               token_intervals_near(
                   k0, k1, max(next0, topology_start_token),
                   min(next1, topology_end_token), local_radius_tokens)))
            topology_neighbor = true;
        }
        if (protect_frame_boundaries) {
          const int rel0 = k0 - topology_start_token;
          const int rel1 = max(rel0, k1 - 1 - topology_start_token);
          const bool boundary_key =
              rel0 / frame_stride_tokens != rel1 / frame_stride_tokens;
          topology_neighbor = topology_neighbor || boundary_key;
        }
      }

      selected = score_tile[item] * scale_log2 > threshold[row] ||
          neighbor || protected_key || protected_query || topology_neighbor;
      exact_mask[row * k_blocks + k_block] = selected ? 1 : 0;
    }
    const unsigned votes = __ballot_sync(0xffffffff, selected);
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int count = __popc(votes & (lane < 16 ? 0xffffu : 0xffff0000u));
    if (accumulate_row_count && (lane & 15) == 0 && q_block < q_blocks && count)
      atomicAdd(row_count + static_cast<std::int64_t>(bh) * q_blocks + q_block, count);
  }
#endif
}


template <int HEAD_DIM>
__global__ void route_centroid_tiles_partition(
    const half *__restrict__ qc, const half *__restrict__ kc,
    const float *__restrict__ threshold, std::uint8_t *__restrict__ exact_mask,
    std::int32_t *__restrict__ row_count, int heads, int q_blocks,
    int k_blocks, int q_padded_blocks, int k_padded_blocks,
    int q_block_base, float scale_log2,
    int sink_start_block, int sink_end_block,
    int topology_start_token, int topology_end_token,
    int frame_stride_tokens, int temporal_radius, int local_radius_tokens,
    int protect_frame_boundaries, int accumulate_row_count) {
#if __CUDA_ARCH__ >= 700
  using namespace nvcuda;
  __shared__ float score_tile[16 * 16];
  const int k_tile = static_cast<int>(blockIdx.x);
  const int q_tile = static_cast<int>(blockIdx.y);
  const int bh = static_cast<int>(blockIdx.z);
  const half *a = qc +
      (static_cast<std::int64_t>(bh) * q_padded_blocks + q_tile * 16) * HEAD_DIM;
  const half *b = kc +
      (static_cast<std::int64_t>(bh) * k_padded_blocks + k_tile * 16) * HEAD_DIM;
  wmma::fragment<wmma::matrix_a,16,16,16,half,wmma::row_major> af;
  wmma::fragment<wmma::matrix_b,16,16,16,half,wmma::col_major> bf;
  wmma::fragment<wmma::accumulator,16,16,16,float> cf;
  wmma::fill_fragment(cf,0.0f);
#pragma unroll
  for (int dim=0;dim<HEAD_DIM;dim+=16) {
    wmma::load_matrix_sync(af,a+dim,HEAD_DIM);
    wmma::load_matrix_sync(bf,b+dim,HEAD_DIM);
    wmma::mma_sync(cf,af,bf,cf);
  }
  wmma::store_matrix_sync(score_tile,cf,16,wmma::mem_row_major);
  __syncthreads();
  for (int item=static_cast<int>(threadIdx.x);item<256;item+=32) {
    const int local_q=item/16;
    const int local_k=item-local_q*16;
    const int q_block=q_tile*16+local_q;
    const int global_q_block=q_block_base+q_block;
    const int k_block=k_tile*16+local_k;
    bool selected=false;
    if (q_block<q_blocks && k_block<k_blocks) {
      const std::int64_t row=static_cast<std::int64_t>(bh)*q_blocks+q_block;
      const bool neighbor=abs(global_q_block-k_block)<=1;
      const bool protected_key=k_block>=sink_start_block&&k_block<sink_end_block;
      const bool protected_query=
          global_q_block>=sink_start_block&&global_q_block<sink_end_block;
      const int q_start=global_q_block*kBlock;
      const int k_start=k_block*kBlock;
      const bool topology_enabled=topology_start_token>=0 &&
          topology_end_token>topology_start_token && frame_stride_tokens>0;
      const bool q_video=topology_enabled &&
          q_start<topology_end_token && q_start+kBlock>topology_start_token;
      const bool k_video=topology_enabled &&
          k_start<topology_end_token && k_start+kBlock>topology_start_token;
      bool topology_neighbor=false;
      if (q_video&&k_video) {
        const int q0=max(q_start,topology_start_token);
        const int q1=min(q_start+kBlock,topology_end_token);
        const int k0=max(k_start,topology_start_token);
        const int k1=min(k_start+kBlock,topology_end_token);
        topology_neighbor=
            token_intervals_near(k0,k1,q0,q1,local_radius_tokens);
        for (int dt=1;dt<=temporal_radius&&!topology_neighbor;++dt) {
          const int shift=dt*frame_stride_tokens;
          const int prev0=q0-shift,prev1=q1-shift;
          const int next0=q0+shift,next1=q1+shift;
          if ((prev1>topology_start_token &&
               token_intervals_near(k0,k1,max(prev0,topology_start_token),
                   min(prev1,topology_end_token),local_radius_tokens)) ||
              (next0<topology_end_token &&
               token_intervals_near(k0,k1,max(next0,topology_start_token),
                   min(next1,topology_end_token),local_radius_tokens)))
            topology_neighbor=true;
        }
        if (protect_frame_boundaries) {
          const int rel0=k0-topology_start_token;
          const int rel1=max(rel0,k1-1-topology_start_token);
          topology_neighbor=topology_neighbor ||
              (rel0/frame_stride_tokens!=rel1/frame_stride_tokens);
        }
      }
      selected=score_tile[item]*scale_log2>threshold[row] ||
          neighbor||protected_key||protected_query||topology_neighbor;
      exact_mask[row*k_blocks+k_block]=selected?1:0;
    }
    const unsigned votes=__ballot_sync(0xffffffff,selected);
    const int lane=static_cast<int>(threadIdx.x)&31;
    const int count=__popc(votes&(lane<16?0xffffu:0xffff0000u));
    if (accumulate_row_count&&(lane&15)==0&&q_block<q_blocks&&count)
      atomicAdd(row_count+static_cast<std::int64_t>(bh)*q_blocks+q_block,count);
  }
#endif
}


__global__ void pack_mask_lut(
    const std::uint8_t *__restrict__ exact_mask,
    const std::int32_t *__restrict__ expected_count,
    std::int32_t *__restrict__ lut, int rows, int k_blocks, int lut_stride,
    bool complement) {
  __shared__ int cursor;
  if (threadIdx.x == 0) cursor = 0;
  __syncthreads();
  const int row = static_cast<int>(blockIdx.x);
  if (row >= rows) return;
  for (int base = 0; base < k_blocks; base += static_cast<int>(blockDim.x)) {
    const int key = base + static_cast<int>(threadIdx.x);
    const bool exact = key < k_blocks && exact_mask[
        static_cast<std::int64_t>(row) * k_blocks + key] != 0;
    const bool selected = key < k_blocks && (complement ? !exact : exact);
    const unsigned ballot = __ballot_sync(0xffffffff, selected);
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int count = __popc(ballot);
    int offset = 0;
    if (lane == 0) offset = atomicAdd(&cursor, count);
    offset = __shfl_sync(0xffffffff, offset, 0);
    if (selected) {
      const int rank = __popc(ballot & ((1u << lane) - 1u));
      const int slot = offset + rank;
      if (slot < lut_stride)
        lut[static_cast<std::int64_t>(row) * lut_stride + slot] = key;
    }
  }
  __syncthreads();
  if (threadIdx.x == 0 && !complement && cursor != expected_count[row]) {
    // A mismatch indicates an internal packing error.  Poison the count so the
    // Python ABI validation stops before launching attention.
    const_cast<std::int32_t *>(expected_count)[row] = -1;
  }
}

__global__ void pack_ordered_lut(
    const std::uint8_t *mask, std::int32_t *lut, std::int32_t *row_count,
    int rows, int blocks, int stride) {
  constexpr int kWarpsPerCta = 4;
  const int warp = static_cast<int>(threadIdx.x) >> 5;
  const int lane = static_cast<int>(threadIdx.x) & 31;
  const int row = static_cast<int>(blockIdx.x) * kWarpsPerCta + warp;
  if (row >= rows) return;
  int cursor = 0;
  const std::int64_t mask_base = static_cast<std::int64_t>(row) * blocks;
  const std::int64_t lut_base = static_cast<std::int64_t>(row) * stride;
  for (int base = 0; base < blocks; base += 32) {
    const int key = base + lane;
    const bool selected = key < blocks && mask[mask_base + key] != 0;
    const unsigned ballot = __ballot_sync(0xffffffff, selected);
    const int count = __popc(ballot);
    if (selected) {
      const int slot = cursor + __popc(ballot & ((1u << lane) - 1u));
      if (slot < stride) lut[lut_base + slot] = key;
    }
    cursor += count;
  }
  if (lane == 0) row_count[row] = cursor;
}

// Fast route bitmap packer: one warp packs one row, one uint32 per 32
// candidate key blocks. Count and word content come from the same ballot.
__global__ void pack_bitmap_lut(
    const std::uint8_t *__restrict__ mask,
    std::uint32_t *__restrict__ bitmap,
    std::int32_t *__restrict__ row_count,
    int rows, int blocks, int words) {
  constexpr int kWarpsPerCta = 4;
  const int warp = static_cast<int>(threadIdx.x) >> 5;
  const int lane = static_cast<int>(threadIdx.x) & 31;
  const int row = static_cast<int>(blockIdx.x) * kWarpsPerCta + warp;
  if (row >= rows) return;
  int cursor = 0;
  const std::int64_t mask_base = static_cast<std::int64_t>(row) * blocks;
  const std::int64_t bitmap_base = static_cast<std::int64_t>(row) * words;
  for (int word = 0; word < words; ++word) {
    const int key = word * 32 + lane;
    const bool exact = key < blocks && mask[mask_base + key] != 0;
    const std::uint32_t bits = __ballot_sync(0xffffffff, exact);
    if (lane == 0) {
      bitmap[bitmap_base + word] = bits;
      cursor += __popc(bits);
    }
  }
  if (lane == 0) row_count[row] = cursor;
}


template <int HEAD_DIM, int TOKENS>
__global__ void quantize_blocks(
    const half *__restrict__ input, std::int8_t *__restrict__ output,
    float *__restrict__ scales, int length, int groups) {
  __shared__ float warp_maxima[8];
  const int linear = static_cast<int>(blockIdx.x);
  const int bh = linear / groups;
  const int group = linear - bh * groups;
  const int start = group * TOKENS;
  const std::int64_t base = static_cast<std::int64_t>(bh) * length * HEAD_DIM;
  float maximum = 0.0f;
  for (int item = static_cast<int>(threadIdx.x); item < TOKENS * HEAD_DIM;
       item += static_cast<int>(blockDim.x)) {
    const int token = item / HEAD_DIM;
    const int dim = item - token * HEAD_DIM;
    if (start + token < length)
      maximum = fmaxf(maximum, fabsf(__half2float(
          input[base + static_cast<std::int64_t>(start + token) * HEAD_DIM + dim])));
  }
  maximum = warp_max(maximum);
  const int warp = static_cast<int>(threadIdx.x) >> 5;
  const int lane = static_cast<int>(threadIdx.x) & 31;
  if (lane == 0) warp_maxima[warp] = maximum;
  __syncthreads();
  if (warp == 0) {
    maximum = lane < 8 ? warp_maxima[lane] : 0.0f;
    maximum = warp_max(maximum);
    if (lane == 0) warp_maxima[0] = maximum;
  }
  __syncthreads();
  const float scale = fmaxf(warp_maxima[0] / 127.0f, 1.0e-8f);
  if (threadIdx.x == 0)
    scales[static_cast<std::int64_t>(bh) * groups + group] = scale;
  for (int item = static_cast<int>(threadIdx.x); item < TOKENS * HEAD_DIM;
       item += static_cast<int>(blockDim.x)) {
    const int token = item / HEAD_DIM;
    const int dim = item - token * HEAD_DIM;
    if (start + token < length) {
      const std::int64_t index = base +
          static_cast<std::int64_t>(start + token) * HEAD_DIM + dim;
      const int value = __float2int_rn(__half2float(input[index]) / scale);
      output[index] = static_cast<std::int8_t>(max(-127, min(127, value)));
    }
  }
}

template <int HEAD_DIM>
__global__ void quantize_v_with_scale_kernel(
    const half *__restrict__ input, const float *__restrict__ scale,
    std::int8_t *__restrict__ output, int batch, int heads, int length,
    int padded_length) {
  const std::int64_t total = static_cast<std::int64_t>(batch) * heads *
      HEAD_DIM * padded_length;
  for (std::int64_t linear = static_cast<std::int64_t>(blockIdx.x) * blockDim.x +
           threadIdx.x;
       linear < total; linear += static_cast<std::int64_t>(gridDim.x) * blockDim.x) {
    std::int64_t value = linear;
    const int token = static_cast<int>(value % padded_length);
    value /= padded_length;
    const int dim = static_cast<int>(value % HEAD_DIM);
    const int bh = static_cast<int>(value / HEAD_DIM);
    std::int8_t quantized = 0;
    if (token < length) {
      const float step = fmaxf(
          scale[static_cast<std::int64_t>(bh) * HEAD_DIM + dim], 1.0e-8f);
      const float source = __half2float(input[
          (static_cast<std::int64_t>(bh) * length + token) * HEAD_DIM + dim]);
      const int rounded = __float2int_rn(source / step);
      quantized = static_cast<std::int8_t>(max(-127, min(127, rounded)));
    }
    output[(static_cast<std::int64_t>(bh) * HEAD_DIM + dim) * padded_length + token] =
        quantized;
  }
}

template <int HEAD_DIM>
inline cudaError_t prepare_routes(
    const half *q, const half *k, const half *v, half *qc, half *kc, half *vc,
    float *kc_mean, float *kc_variance, float *threshold,
    std::uint8_t *exact_mask, std::int32_t *row_count, int batch, int heads,
    int length, int padded_blocks, float tau, float attention_scale,
    int sink_start_block, int sink_end_block,
    int topology_start_token, int topology_end_token,
    int frame_stride_tokens, int temporal_radius, int local_radius_tokens,
    int protect_frame_boundaries, int accumulate_row_count, cudaStream_t stream,
    std::int8_t *q8 = nullptr, std::int8_t *k8 = nullptr,
    float *qs = nullptr, float *ks = nullptr,
    InputStrides q_stride = {}, InputStrides k_stride = {}, InputStrides v_stride = {}) {
  static_assert(HEAD_DIM == 64 || HEAD_DIM == 128, "Sol preprocessing supports D64/D128");
  const int blocks = (length + kBlock - 1) / kBlock;
  const int bh = batch * heads;
  cudaError_t error = cudaSuccess;
  if (accumulate_row_count) {
    error = cudaMemsetAsync(
        row_count, 0, static_cast<std::size_t>(bh) * blocks * sizeof(std::int32_t),
        stream);
    if (error != cudaSuccess) return error;
  }
  if (q8) {
    reduce_quantized_qkv<HEAD_DIM><<<bh * padded_blocks, HEAD_DIM, 0, stream>>>(
        q, k, v, qc, kc, vc, q8, k8, qs, ks, heads, length, blocks, padded_blocks,
        q_stride, k_stride, v_stride);
  } else {
    reduce_qkv_centroids<HEAD_DIM><<<bh * padded_blocks, HEAD_DIM, 0, stream>>>(
        q, k, v, qc, kc, vc, heads, length, blocks, padded_blocks);
  }
  const float scale_log2 = attention_scale * kLog2E;
  reduce_kc_stats<HEAD_DIM><<<bh,HEAD_DIM,0,stream>>>(
      kc,kc_mean,kc_variance,blocks,padded_blocks);
  compute_diag_threshold<HEAD_DIM><<<bh*blocks,HEAD_DIM,0,stream>>>(
      qc,kc_mean,kc_variance,threshold,blocks,padded_blocks,tau,scale_log2);
  dim3 route_grid((padded_blocks + 15) / 16, (padded_blocks + 15) / 16, bh);
  route_centroid_tiles<HEAD_DIM><<<route_grid, 32, 0, stream>>>(
      qc, kc, threshold, exact_mask, row_count, heads, blocks, blocks,
      padded_blocks, scale_log2, sink_start_block, sink_end_block,
      topology_start_token, topology_end_token, frame_stride_tokens,
      temporal_radius, local_radius_tokens, protect_frame_boundaries,
      accumulate_row_count);
  return cudaPeekAtLastError();
}


template <int HEAD_DIM>
inline cudaError_t prepare_partition_routes(
    const half *q, const half *k, const half *v,
    half *qc, half *kc, half *vc, float *kc_mean, float *kc_variance,
    float *threshold, std::uint8_t *exact_mask, std::int32_t *row_count,
    std::int8_t *q8, std::int8_t *k8, float *qs, float *ks,
    int batch, int heads, int q_length, int kv_length,
    int q_padded_blocks, int k_padded_blocks, int q_block_base,
    float tau, float attention_scale, int sink_start_block, int sink_end_block,
    int topology_start_token, int topology_end_token, int frame_stride_tokens,
    int temporal_radius, int local_radius_tokens, int protect_frame_boundaries,
    int accumulate_row_count, cudaStream_t stream,
    InputStrides q_stride, InputStrides k_stride, InputStrides v_stride) {
  const int q_blocks=(q_length+kBlock-1)/kBlock;
  const int k_blocks=(kv_length+kBlock-1)/kBlock;
  const int bh=batch*heads;
  cudaError_t error=cudaSuccess;
  if (accumulate_row_count) {
    error=cudaMemsetAsync(row_count,0,
        static_cast<std::size_t>(bh)*q_blocks*sizeof(std::int32_t),stream);
    if (error!=cudaSuccess) return error;
  }
  reduce_quantized_q_local<HEAD_DIM><<<bh*q_padded_blocks,HEAD_DIM,0,stream>>>(
      q,qc,q8,qs,heads,q_length,q_blocks,q_padded_blocks,q_stride);
  reduce_quantized_kv_full<HEAD_DIM><<<bh*k_padded_blocks,HEAD_DIM,0,stream>>>(
      k,v,kc,vc,k8,ks,heads,kv_length,k_blocks,k_padded_blocks,k_stride,v_stride);
  const float scale_log2=attention_scale*kLog2E;
  reduce_kc_stats<HEAD_DIM><<<bh,HEAD_DIM,0,stream>>>(
      kc,kc_mean,kc_variance,k_blocks,k_padded_blocks);
  compute_diag_threshold<HEAD_DIM><<<bh*q_blocks,HEAD_DIM,0,stream>>>(
      qc,kc_mean,kc_variance,threshold,q_blocks,q_padded_blocks,tau,scale_log2);
  dim3 route_grid((k_padded_blocks+15)/16,(q_padded_blocks+15)/16,bh);
  route_centroid_tiles_partition<HEAD_DIM><<<route_grid,32,0,stream>>>(
      qc,kc,threshold,exact_mask,row_count,heads,q_blocks,k_blocks,
      q_padded_blocks,k_padded_blocks,q_block_base,scale_log2,
      sink_start_block,sink_end_block,topology_start_token,topology_end_token,
      frame_stride_tokens,temporal_radius,local_radius_tokens,
      protect_frame_boundaries,accumulate_row_count);
  return cudaPeekAtLastError();
}


inline cudaError_t pack_lut(
    const std::uint8_t *exact_mask, const std::int32_t *row_count,
    std::int32_t *lut, int rows, int k_blocks, int lut_stride,
    bool complement, cudaStream_t stream) {
  pack_mask_lut<<<rows, 256, 0, stream>>>(
      exact_mask, row_count, lut, rows, k_blocks, lut_stride, complement);
  return cudaPeekAtLastError();
}

template <int HEAD_DIM>
inline cudaError_t quantize(
    const half *input, std::int8_t *output, float *scale, int batch,
    int heads, int length, int block, cudaStream_t stream) {
  const int groups = (length + block - 1) / block;
  if (block == 16) {
    quantize_blocks<HEAD_DIM, 16><<<batch * heads * groups, 256, 0, stream>>>(
        input, output, scale, length, groups);
  } else if (block == 64) {
    quantize_blocks<HEAD_DIM, 64><<<batch * heads * groups, 256, 0, stream>>>(
        input, output, scale, length, groups);
  } else {
    return cudaErrorInvalidValue;
  }
  return cudaPeekAtLastError();
}

template <int HEAD_DIM>
inline cudaError_t quantize_v_with_scale(
    const half *input, const float *scale, std::int8_t *output, int batch,
    int heads, int length, int padded_length, cudaStream_t stream) {
  const std::int64_t total = static_cast<std::int64_t>(batch) * heads *
      HEAD_DIM * padded_length;
  const std::int64_t requested = (total + 255) / 256;
  const int blocks = static_cast<int>(requested < 65535 ? requested : 65535);
  quantize_v_with_scale_kernel<HEAD_DIM><<<blocks, 256, 0, stream>>>(
      input, scale, output, batch, heads, length, padded_length);
  return cudaPeekAtLastError();
}

}  // namespace star7_sol_preprocess
