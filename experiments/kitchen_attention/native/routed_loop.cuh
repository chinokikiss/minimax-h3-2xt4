// Inside the pinned attention body. Compact sorted leaf64 traversal, original
// Kitchen QK/PV primitives and scales. No TileLang or NxN token mask.
    static_assert(CTA_Q == 64 && CTA_K == 64);
    const uint64_t route_row =
        (uint64_t(batch_id) * num_qo_heads + head_id) * gridDim.x + bx;
    const uint32_t entries = RouteCounts ? min(max(RouteCounts[route_row],0),route_capacity) : num_iterations;
    const float original_sm_scale = sm_scale;
    const float q_scale = Q_scale[q_scale_idx];
    auto load_key = [&](uint32_t tile) {
      int8_t *pointer = K_lane_base_ptr + uint64_t(tile) * CTA_K * stride_seq_k;
      uint32_t offset = K_smem_offset_load;
      load_global_to_share<global_to_shared_line_lanes_QK,
          global_to_shared_copy_lines_per_warp_QK, QK_smem_iters_row,
          K_smem_iters_col, swizzle_mode_QK, QK_SMEM_STRIDE / PACK_SIZE_QK, CTA_K>(
              &pointer, offset, stride_seq_k, smem_K,
              K_load_idx_lane_base + tile * CTA_K, kv_len);
      cp_async::commit_group();
    };
    auto load_value = [&](uint32_t tile) {
      int8_t *pointer = V_lane_base_ptr + tile * CTA_K;
      uint32_t offset = V_smem_offset_load;
      load_int8_V_global_to_share<global_to_shared_line_lanes_V,
          global_to_shared_copy_lines_per_warp_V, V_smem_iters_row,
          V_smem_iters_col, swizzle_mode_V, V_SMEM_STRIDE / PACK_SIZE_V, CTA_K>(
              &pointer, offset, stride_d_v, smem_V);
      cp_async::commit_group();
    };
    for (uint32_t entry = 0; entry < entries; ++entry) {
      const uint32_t tile = RouteIds ? RouteIds[route_row * route_capacity + entry] : entry;
      // Sorted IDs permit causal early exit. Arbitrary query offsets are allowed.
      if (tile >= num_iterations) break;
      load_key(tile);
      load_value(tile);
      cp_async::wait_group<0>();
      __syncthreads();
      K_idx_lane_base = get_warp_idx_k<num_warps_q, num_warps_k>() * WARP_K +
                        2 * (lane_id % 4) + tile * CTA_K;
      sm_scale = original_sm_scale *
          (q_scale * K_scale[k_scale_idx + tile * k_scale_advance_offset]);
      compute_int_qk<num_warps_q, num_warps_k, num_tiles_q, num_tiles_k,
          num_tiles_qk_inner, swizzle_mode_QK, QK_SMEM_STRIDE / PACK_SIZE_QK, DTypeQK>(
              smem_Q, smem_K, RS, Q_smem_offset_mma, K_smem_offset_mma);
      uint32_t RS_u8[num_tiles_q][num_tiles_k / 2][4];
      const bool boundary = (tile + 1) * CTA_K > kv_len ||
          (H3_CAUSAL && (tile + 1) * CTA_K > uint32_t(query_offset) + bx * CTA_Q + 1);
      if (!boundary) {
        update_mdo_i32_u8<num_tiles_q, num_tiles_k, num_tiles_v>(
            RS, RO, m, d, sm_scale, S_U8_OFFSET, RS_u8);
        for (uint32_t fq = 0; fq < num_tiles_q; ++fq)
          valid[fq][0] = valid[fq][1] = true;
      } else {
        auto &scores = reinterpret_cast<float(&)[num_tiles_q][num_tiles_k][8]>(RS);
        for (uint32_t fq = 0; fq < num_tiles_q; ++fq) {
          for (uint32_t fk = 0; fk < num_tiles_k; ++fk) {
            for (uint32_t z = 0; z < 8; ++z) {
              const uint32_t row = Q_idx_lane_base + fq * 16 + 8 * ((z % 4) / 2);
              const uint32_t key = K_idx_lane_base + fk * 16 + 8 * (z / 4) + z % 2;
              const bool allowed = key < kv_len && (!H3_CAUSAL || key <= row + query_offset);
              const float value = __int2float_rz(RS[fq][fk][z]) * sm_scale;
              scores[fq][fk][z] = allowed ? value : -50000.0f;
              valid[fq][(z % 4) / 2] |= allowed;
            }
          }
        }
        update_mdo_f32_u8<num_tiles_q, num_tiles_k, num_tiles_v>(
            scores, RO, m, d, S_U8_OFFSET, RS_u8);
      }
      __syncthreads();
      compute_int8_sv<num_warps_q, num_warps_k, num_tiles_q, num_tiles_k,
          num_tiles_v, swizzle_mode_V, V_SMEM_STRIDE / PACK_SIZE_V, false>(
              smem_V, RS, RS_u8, RO);
      __syncthreads();
    }
