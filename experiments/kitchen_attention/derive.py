"""Derive an experimental header from hash-verified Kitchen 0.2.37 sources."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def verify_vendor():
    manifest = json.loads((ROOT / 'native/vendor/manifest.json').read_text())
    for name, digest in manifest['files'].items():
        actual = hashlib.sha256((ROOT / 'native/vendor' / name).read_bytes()).hexdigest()
        if actual != digest:
            raise RuntimeError(f'Frozen Kitchen source identity mismatch: {name}')
    return manifest


def replace_once(text, before, after):
    if text.count(before) != 1:
        raise RuntimeError(f'Unexpected frozen source structure: {before[:100]}')
    return text.replace(before, after, 1)


def derive_header(destination):
    verify_vendor()
    text = (ROOT / 'native/vendor/qk_int_sv_i8_cuda.cuh').read_text()
    text=replace_once(text,'#include "attn_utils.cuh"',
        '#include "attn_utils.cuh"\n#include "h3_dense_utils.cuh"')
    text = replace_once(text,
        'bool fuse_fp32_probabilities = true, bool skip_masked_copies = false,\n          typename Offset = uint32_t>',
        'bool fuse_fp32_probabilities = true, bool skip_masked_copies = false,\n'
        '          typename Offset = uint32_t, bool H3_ROUTE = false,\n'
        '          bool H3_CAUSAL = false, bool H3_CACHE_Q = false, bool H3_PV_N8 = false>')
    signature = 'const float *__restrict__ MaskTileBias) {'
    # The body and upstream global wrapper share this suffix; change the body only.
    first = text.index(signature)
    text = text[:first] + text[first:].replace(signature,
        'const float *__restrict__ MaskTileBias,\n'
        '    const int *__restrict__ RouteIds = nullptr,\n'
        '    const int *__restrict__ RouteCounts = nullptr,\n'
        '    const int route_capacity = 0, const int query_offset = 0,\n'
        '    float *__restrict__ H3Lse = nullptr) {', 1)
    text = replace_once(text, '  constexpr bool cache_query = false;',
        '  constexpr bool cache_query = H3_CACHE_Q && CTA_Q == 128 &&\n'
        '      head_dim == 128 && CTA_K == 128 && mask_mode == MaskMode::kNone;')
    text = replace_once(text,
        'mask_mode == MaskMode::kCausal ? min(kv_len, (bx + 1) * CTA_Q) : kv_len,',
        '(H3_CAUSAL ? min(kv_len, uint32_t(query_offset) + (bx + 1) * CTA_Q) :\n'
        '       (mask_mode == MaskMode::kCausal ? min(kv_len, (bx + 1) * CTA_Q) : kv_len)),')
    loop = (ROOT / 'native/routed_loop.cuh').read_text()
    text = replace_once(text, '  if constexpr (skip_masked_copies) {',
        '  if constexpr (H3_ROUTE) {\n' + loop + '\n  } else if constexpr (skip_masked_copies) {')
    text = replace_once(text,
        'if constexpr (custom_mask && mask_mode != MaskMode::kPreparedKey) {',
        'if constexpr (H3_ROUTE || (custom_mask && mask_mode != MaskMode::kPreparedKey)) {')
    # Generic LSE mapping: upstream explicitly supports only num_tiles_q == 2.
    marker = '  // ! here we just implement the case for fp32 acumulation'
    lse = '''  if constexpr (H3_ROUTE) {
    if (lane_id % 4 == 0 && H3Lse) {
      for (uint32_t fq = 0; fq < num_tiles_q; ++fq) {
        for (uint32_t r = 0; r < 2; ++r) {
          const uint32_t row = bx * CTA_Q +
              get_warp_idx_q<num_warps_q, num_warps_k>() * WARP_Q +
              fq * 16 + lane_id / 4 + r * 8;
          if (row < qo_len)
            H3Lse[(uint64_t(batch_id) * num_qo_heads + head_id) * qo_len + row] =
                valid[fq][r] ? math::ptx_log2(d[fq][r]) + m[fq][r] : -CUDART_INF_F;
        }
      }
    }
  }
'''
    text = replace_once(text, marker, lse + '\n' + marker)
    text=text.replace('compute_int8_sv<','h3_compute_int8_sv<H3_PV_N8, ')
    destination = Path(destination)
    destination.write_text('// H3 experimental derivative; upstream vendor files are unchanged.\n' + text)
    return hashlib.sha256(destination.read_bytes()).hexdigest()
