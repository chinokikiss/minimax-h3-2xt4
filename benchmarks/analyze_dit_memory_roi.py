"""Reproducible TP2 buffer arithmetic; no GPU execution or peak predictions."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def estimate(tokens=59134,chunk_rows=4096,video_tokens=54432):
    if tokens<1 or chunk_rows<1 or not 0<video_tokens<=tokens:
        raise ValueError('Positive rows and a bounded video segment required')
    hidden,local_attention,local_ffn=5376,3584,7168
    def row(width,size,rows=tokens): return rows*width*size
    sizes=dict(residual_fp16=row(hidden,2),residual_fp32=row(hidden,4),
        fc1_input_codes=row(hidden,1),fc1_accumulator_int32=row(2*local_ffn,4),
        fc1_output_fp16=row(2*local_ffn,2),fc2_rotated_fp32=row(local_ffn,4),
        fc2_input_codes=row(local_ffn,1),row_partial_int32=row(hidden,4),
        row_payload_fp16=row(hidden,2),row_epilogue_fp32=row(hidden,4),
        packed_qkv_fp16=row(3*local_attention,2),qk_outputs_fp16=row(2*local_attention,2),
        attention_output_fp16=row(local_attention,2),
        kitchen_int8_qkv_unpadded=row(3*local_attention,1),
        fc1_chunk_accumulator_int32=row(2*local_ffn,4,rows=min(tokens,chunk_rows)),
        fc1_chunk_output_fp16=row(2*local_ffn,2,rows=min(tokens,chunk_rows)),
        largest_segment_fp32=row(hidden,4,rows=video_tokens))
    candidates=dict(
        release_fc1_input_after_quant=sizes['residual_fp16'],
        release_fc1_output_before_fc2_gemm=sizes['fc1_output_fp16'],
        release_float_qkv_before_attention=sizes['packed_qkv_fp16']+sizes['qk_outputs_fp16'],
        attention_output_allocation_bound=sizes['attention_output_fp16'],
        inplace_rms_rope_output_buffers=sizes['qk_outputs_fp16'],
        fc1_accumulator_reduction=sizes['fc1_accumulator_int32']-sizes['fc1_chunk_accumulator_int32'],
        norm_two_fp32_buffers_vs_fp16_destination=2*sizes['residual_fp32']-sizes['residual_fp16'],
        largest_segment_gated_add_three_fp32_temporaries=3*sizes['largest_segment_fp32'],
        direct_fp16_row_epilogue_removes_fp32_destination=sizes['row_epilogue_fp32'],
        recompute_removes_rotated_storage=sizes['fc2_rotated_fp32'])
    return dict(kind='Analytical buffer sizes, not measured memory savings',tokens=tokens,
        chunk_rows=chunk_rows,largest_video_segment_tokens=video_tokens,tp=2,bytes_per_rank=sizes,
        gib_per_rank={k:v/2**30 for k,v in sizes.items()},
        candidate_buffer_bytes=candidates,candidate_buffer_gib={k:v/2**30 for k,v in candidates.items()},
        exclusions=['Resident/prefetched weights','cuBLAS/Kitchen/NCCL workspaces',
                    'Padding/scales/allocator rounding','CUDA context/external allocations'],
        warning='Candidates overlap, move phase maxima and are not additive global-peak savings.')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();result=estimate()
    sources=['patches/tp_dit.py','patches/dit_column_linear.py','patches/h3_tensor_flow.py',
        'patches/tilelang_swiglu_quant.py','patches/tilelang_column_epilogue.py',
        'patches/tilelang_fp32_rms_rope.py','patches/tilelang_transport_p2.py',
        'patches/weight_residency.py','patches/ptd/runtime.py','patches/ptd/queue.py']
    result['source_sha256']={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in sources}
    result.update(gpu_used=False,kaggle_interactions=0,dependencies_changed=False)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result['candidate_buffer_gib'],indent=2))


if __name__=='__main__':main()
