"""Derive per-rank targets from actual TP module schemas; CPU/meta only."""
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
OUT=Path('D:/attention_eval/operator_optimization_tp2_20261016');OUT.mkdir(parents=True,exist_ok=True)
from patches.rank_local_int8 import shell
from patches.parallel_vae import aligned_ranges


def main():
    cases=[]
    for component,rows in [('te',353),('dit',6814)]:
        states=[shell(component,rank).state_dict() for rank in range(2)]
        assert {k:list(v.shape) for k,v in states[0].items()}=={k:list(v.shape) for k,v in states[1].items()}
        columns={'te':['q_proj','k_proj','v_proj','gate_proj','up_proj'], 'dit':['qkv_proj','fc1']}[component]
        for name,value in states[0].items():
            module=name.removesuffix('.weight')
            if not name.endswith('.weight') or value.ndim!=2 or str(value.dtype)!='torch.int8':continue
            n,k=value.shape;column=module in columns
            cases.append({'component':component,'module':module,'M':rows,'N':n,'K':k,
                'partition':'column' if column else 'row','input_dtype':'float32',
                'quantized_row_width':k if column else 2*k,
                'input_act':'swiglu' if component=='dit' and module=='fc2' else None,
                'collectives':[] if column else ['FP32 row maximum AllReduce(MAX) on fallback rotation','INT32 [M,N] AllReduce(SUM) before dequantization'],
                'quantization':'Column: full input row; row: current scalemax route, full-width FP32 rotation geometry preserved'})
    ranges=aligned_ranges(1797);assert ranges==((0,1024),(1024,1797))
    for rank,(a,b) in enumerate(ranges):
        for module,n,k in [('qkv',6144,2048),('out',2048,2048),('w1',16384,2048),('w2',2048,8192)]:
            cases.append({'component':'video','module':module,'rank':rank,'M':b-a,'N':n,'K':k,
                'partition':'sequence/query','input_dtype':'float16','quantized_row_width':k,
                'input_act':'swiglu' if module=='w2' else 'rms_norm' if module in ['qkv','w1'] else None,
                'collectives':[],'quantization':'Native complete local token rows; replicated decoder weights'})
    manifest={'scope':'Current PhysicalTPService: optimized_tp.enable(scalemax); Video AlignedVideoDecode, not legacy head-sharded tp_video_vae',
        'attention':[{'component':'te','rank':rank,'q':[1,32,353,128],'k':[1,4,353,128],'causal':True,'head_offset':rank*32,'kv_head_offset':rank*4} for rank in [0,1]]+
            [{'component':'dit','rank':rank,'q':[1,28,6814,128],'k':[1,28,6814,128],'causal':False,'head_offset':rank*28,'prefix_tokens':874} for rank in [0,1]]+
            [{'component':'video','rank':rank,'q':[1,32,b-a,64],'k':[1,32,1797,64],'query_offset':a,'causal':False} for rank,(a,b) in enumerate(ranges)],
        'GEMMs':cases,'unique_MNK_count':len({(x['M'],x['N'],x['K']) for x in cases}),
        'fusion_constraints':['Keep global activation scale; local-only row maxima change the operator.',
            'Keep INT32 sum before row-output dequant/bias/residual; a fused per-rank floating epilogue before SUM is not equivalent.',
            'Current scalemax FP32 rotation pads to the global width to preserve GEMM rounding geometry.',
            'Current optimized native_int_mm already removes transposed-weight copies and M padding.',
            'Video queries are unequal and KV exchange precedes attention; key centering uses global keys.'],
        'source_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROOT/'patches'/f for f in
            ['tp_dit.py','tp_qwen3vl.py','parallel_int8.py','optimized_tp.py','parallel_vae.py','rank_local_int8.py','comfy_tp_runtime.py']]}}
    (OUT/'tp2_operator_manifest.json').write_text(json.dumps(manifest,indent=2))
    print('Actual meta schemas verified:',len(cases),'rank-shape GEMM entries,',manifest['unique_MNK_count'],'unique MNK; video ranges',ranges)


if __name__=='__main__':main()
