"""CPU-only shape/source/saved-T4 trace audit. Does not allocate CUDA or submit jobs."""
import argparse
from collections import Counter,defaultdict
import hashlib
import json
from pathlib import Path
import sqlite3
import struct

ROOT=Path(__file__).resolve().parents[1]
PREVIOUS=Path('D:/attention_eval/quality_constrained_20261015')
CAPTURE=Path('D:/attention_eval/tokyo_ref2va_20261013')
COMFY=Path('C:/Users/Administrator/Documents/project/ComfyUI-master')


def sha(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def header(path):
    with path.open('rb') as stream:
        size=struct.unpack('<Q',stream.read(8))[0]
        if size>32*2**20:raise ValueError('Unexpected checkpoint header size')
        result=json.loads(stream.read(size))
    return result,8+size


def quant_config(path,h,offset,prefix):
    name=prefix+'.comfy_quant'
    if name not in h:return None
    a,b=h[name]['data_offsets']
    if b-a>4096:raise ValueError('Unexpected quantization metadata size')
    with path.open('rb') as stream:stream.seek(offset+a);return json.loads(stream.read(b-a))


def trace_summary(path):
    db=sqlite3.connect(path);db.row_factory=sqlite3.Row
    names=dict(db.execute('select id,value from StringIds'))
    ranges=[dict(r) for r in db.execute('select * from NVTX_EVENTS where end>start')]
    for r in ranges:r['label']=r['text'] or names.get(r['textId'],'')
    calls={}
    tables={r[0] for r in db.execute("select name from sqlite_master where type='table'")}
    for table in ['CUPTI_ACTIVITY_KIND_RUNTIME','CUPTI_ACTIVITY_KIND_DRIVER']:
        if table in tables:
            for r in db.execute('select * from '+table):calls.setdefault(r['correlationId'],dict(r))
    categories=defaultdict(lambda:{'kernel_count':0,'summed_kernel_ms':0.})
    top=defaultdict(lambda:{'kernel_count':0,'summed_kernel_ms':0.})
    kernels=[dict(r) for r in db.execute('select * from CUPTI_ACTIVITY_KIND_KERNEL')]
    for r in kernels:
        name=names.get(r['demangledName'],names.get(r['shortName'],''));launch=calls.get(r['correlationId'])
        parent=[]
        if launch:parent=[v for v in ranges if v['globalTid']==launch['globalTid'] and v['start']<=launch['start']<v['end']]
        if any('Attention_API' in v['label'] for v in parent):
            labels=[v['label'] for v in parent]
            if any('QK_softmax_PV' in label for label in labels):category='attention_QK_softmax_PV_scope'
            elif any('QK_quantization' in label for label in labels):category='attention_quantization_and_mean'
            elif 'input_cast' in labels:category='attention_input_cast'
            elif any('routing' in label for label in labels):category='attention_routing'
            elif any('preparation' in label for label in labels):category='attention_legacy_preparation'
            elif '_native_' in path.name:category='native_attention_API_scope'
            else:category='attention_other_layout_output'
        elif 'cutlass' in name and 'signed char' in name and 'Visitor' in name:category='linear_INT8_GEMM_fused_epilogue'
        elif 'cutlass' in name and ('i8816gemm' in name or 's8' in name):category='linear_INT8_GEMM_other'
        elif 'sgemm' in name:category='FP32_GEMM_outside_attention_unassigned'
        elif 'dequantize_int8_linear' in name:category='linear_separate_dequantization'
        elif 'quantize_int8' in name:category='linear_quantization'
        elif any(s in name.lower() for s in ['rms_rope','apply_rope','rope_kernel']):category='RoPE_or_RMS_RoPE'
        elif any(s in name.lower() for s in ['layer_norm','rms_norm','rmsnorm']):category='normalization'
        elif 'addcmul' in name:category='gated_residual_addcmul'
        elif 'silu' in name:category='SiLU'
        elif 'copy' in name.lower():category='copy_cast_or_packing_kernel'
        else:category='other_kernel'
        duration=(r['end']-r['start'])/1e6
        for row in [categories[category],top[name]]:row['kernel_count']+=1;row['summed_kernel_ms']+=duration
    total=sum(r['summed_kernel_ms'] for r in categories.values())
    for r in categories.values():r['fraction_of_summed_kernel_time']=r['summed_kernel_ms']/total
    memcpy=[]
    if 'CUPTI_ACTIVITY_KIND_MEMCPY' in tables:
        memcpy=[dict(r) for r in db.execute('select copyKind,count(*) as count,sum(bytes) as bytes,sum(end-start)/1e6 as summed_ms from CUPTI_ACTIVITY_KIND_MEMCPY group by copyKind')]
    db.close()
    return {'file':str(path),'sha256':sha(path),'scope':'CPU analysis of saved physical-T4 DiT blocks0–49, denoiser call9; serial two-device placement, not TP2',
        'kernel_count':len(kernels),'summed_kernel_ms':total,'categories':dict(categories),
        'top_kernels':[{'name':k,**v} for k,v in sorted(top.items(),key=lambda item:-item[1]['summed_kernel_ms'])[:20]],
        'memcpy':memcpy,'limits':['Summed GPU kernel durations are not an exclusive wall-time critical-path decomposition.',
            'The QK/softmax/PV phase can include validation/LSE auxiliaries in the legacy sparse API; its kernel count is preserved.',
            'FP32 SGEMM source-call attribution is not recorded; the ConvRot rotation explanation is a source/count-based inference.',
            'No measured occupancy, stalls, spills or Tensor Core utilization; Nsight Compute permission failure remains applicable.']}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,default=Path('D:/attention_eval/operator_optimization_20261016'));args=ap.parse_args()
    out=args.output;out.mkdir(parents=True,exist_ok=True)
    policy={'stage':'analysis_only','target_architecture':'SM75 physical T4','new_GPU_jobs_submitted':0,
        'full_pipeline_runs_allowed':False,'scalar_pooling_retained':True,
        'attention_modes':{'TE_text':['group32_dense_causal_GQA'],'DiT':['group32_dense','group32_sparse_drop','group32_sparse_scalar_pool'],
                           'Video_VAE_decoder':['group32_dense']},
        'precision_contract':'Per-token contiguous channel groups32 for Q/K INT8, FP32 scales/softmax/accumulation, FP16 P/V; GEMM retains checkpoint ConvRot256 and native activation row scales.',
        'coverage_gaps':['Current fused Group32 dense requires equal Q/K head counts; TE native GQA must be implemented without repeat_interleave.',
            'Existing Sol drop/scalar kernels use per-token D-wide scales, not Group32; retained-leaf grouped QK paths must be added.',
            'TE visual front-end uses source-derived D72 (1152/16), outside existing D64/D128 Group32 kernels; tail-group support and real shape coverage are required before claiming all TE attention.',
            'VAE encoder/FP32 islands are not silently covered by the decoder attention replacement.'],
        'proposed_first_allocation_wall_minutes_cap':20,'budget_status':'Proposed screening cap only; no allocation launched or quota consumption assumed.'}
    (out/'operator_policy.json').write_text(json.dumps(policy,indent=2))
    capture_path=CAPTURE/'capture_manifest.json';capture=json.loads(capture_path.read_text())['records']
    counts=Counter((r['component'],tuple(r['q_shape']),tuple(r['k_shape']),r['dtype']) for r in capture)
    attn=[{'component':c,'q_shape':q,'k_shape':k,'dtype':d,'captured_record_count':v,
           'count_scope':'Stored diagnostic records, not total pipeline invocation frequency'} for (c,q,k,d),v in counts.items()]
    newer=PREVIOUS/'quality_challenge_v1/generation_screen'
    observed={}
    for name in ['fixed_group32_dit_seed20261013','challenge_group32_dit_seed20261013']:
        rows=json.loads((newer/name/'attention_profile.json').read_text())
        observed[name]=[{'component':c,'q_shape':q,'k_shape':k,'dtype':d,'event_count':n} for (c,q,k,d),n in Counter(
            (r['component'],tuple(r['q_shape']),tuple(r['k_shape']),r['dtype']) for r in rows if r['component'] in ['dit','video'] and r['layer']>=0).items()]
    inventory={'source_capture_manifest':str(capture_path),'source_sha256':sha(capture_path),'captured_shapes':attn,
        'recent_observed_shapes':observed,'notes':['Native6814 real QKV captures are available.7895 is observed in a saved generation profile; no full7895 QKV dump was created.',
        'GEMM shapes below are derived from real checkpoint weight headers and corresponding captured token counts; per-GEMM runtime activation hooks have not been captured.']}
    (out/'operator_shape_inventory.json').write_text(json.dumps(inventory,indent=2))
    specs={
        'te':('qwen3vl_32b_minimax_h3_int8_convrot.safetensors',353,'model.layers.0.',
            ['self_attn.q_proj','self_attn.k_proj','self_attn.v_proj','self_attn.o_proj','mlp.gate_proj','mlp.up_proj','mlp.down_proj']),
        'dit':('MiniMax_H3_Ref2VA_pruned_int8_convrot.safetensors',6814,'blocks.0.',
            ['attn.qkv_proj','attn.out_proj','mlp.fc1','mlp.fc2']),
        'video':('minimax_h3_video_vae_int8_convrot.safetensors',1797,'decoder.transformer_blocks.0.',
            ['attn.to_qkv','attn.to_out','ff.w1','ff.w2'])}
    gemms=[];headers={}
    for component,(filename,m,prefix,modules) in specs.items():
        path=Path('D:/minimaxH3')/filename;h,offset=header(path);headers[component]={'file':str(path),'file_bytes':path.stat().st_size,'header_sha256':hashlib.sha256(json.dumps(h,sort_keys=True).encode()).hexdigest()}
        for module in modules:
            key=prefix+module;weight=h[key+'.weight'];n,k=weight['shape'];assert weight['dtype']=='I8'
            scales={name:entry['shape'] for name,entry in h.items() if name.startswith(key+'.') and 'scale' in name}
            gemms.append({'component':component,'module':module,'weight_key':key+'.weight','M':m,'N':n,'K':k,
                'weight_dtype':weight['dtype'],'bias_dtype':h.get(key+'.bias',{}).get('dtype'),'scales':scales,
                'quant_config':quant_config(path,h,offset,key),'integer_GEMM_operations':2*m*n*k,'INT8_weight_bytes':n*k,
                'int32_output_bytes_if_materialized':4*m*n,
                'shape_provenance':'Actual weight header + native captured token count; not a runtime GEMM operand capture'})
    (out/'gemm_target_shapes.json').write_text(json.dumps({'checkpoints':headers,'cases':gemms,
        'unique_MNK_count':len({(r['M'],r['N'],r['K']) for r in gemms}),
        'activation_coverage':'No proven real inputs to these GEMMs are present in the QKV-only captures. Seeded synthetic timing/code-equivalence cases must be labeled; real-distribution quantization claims require actual operator input fixtures.'},indent=2))
    traces=[trace_summary(p) for p in sorted(newer.glob('*step09.sqlite'))]
    (out/'existing_t4_operator_attribution.json').write_text(json.dumps(traces,indent=2))
    files=[ROOT/'patches/tilelang_grouped_qk.py',ROOT/'patches/tilelang_grouped_fused.py',ROOT/'patches/tilelang_sol_drop_fused.py',
        ROOT/'patches/tilelang_scalar_pool.py',COMFY/'comfy/ldm/minimax/model.py',COMFY/'comfy/ldm/minimax/vae.py',COMFY/'comfy/text_encoders/llama.py',
        COMFY/'comfy/text_encoders/qwen3vl.py',out/'kitchen_0_2_37/comfy_kitchen/backends/cuda/__init__.py',Path(__file__)]
    (out/'audit_source_hashes.json').write_text(json.dumps({str(p):sha(p) for p in files},indent=2))
    print('CPU-only audit complete:',len(attn),'captured attention shapes,',len(gemms),'GEMM modules,',len(traces),'saved T4 traces;0new GPU jobs')


if __name__=='__main__':main()
