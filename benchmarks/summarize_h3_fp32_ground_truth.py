"""Preserve out-of-scope forced-dense-FP32 results, not Native FP32-flow evidence."""
import argparse,hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
FILES=('dit_7895.json','dit_layer24_stress.json','dit_layer49_stress.json','te_layer0.json',
       'te_layer49_stress.json','dit_lora_513.json','frontend.json','vision.json','video_source_direct.json')


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def collect(root):
    result=[];environment=None
    for name in FILES:
        path=root/'final'/name;data=json.loads(path.read_text(encoding='utf-8'))
        if data.get('schema')!='fp32-ground-truth-v1' or data['status']!='complete':raise ValueError('Invalid record: '+name)
        observed={key:data[key] for key in ('hardware','capability','torch','runtime_lock_sha256')}
        if environment is None:environment=observed
        elif environment!=observed:raise ValueError('Mixed environments')
        for source,expected in data['source_sha256'].items():
            paths=[ROOT/source,*root.glob('sources/*/'+source)]
            if not any(p.is_file() and digest(p)==expected for p in paths):raise ValueError('Unpreserved source: '+source)
        gold={key:value for key,value in data.items() if key.startswith('fp32_ground_truth')}
        if not gold:raise ValueError('Missing FP32 ground truth')
        for reference in gold.values():
            contract=reference['contract']
            if reference['role']!='ground_truth' or reference['activation_dtype']!='torch.float32':raise ValueError('Wrong ground-truth role')
            if contract['int8_linear_calls'] or contract['tf32_matmul'] or contract['tf32_cudnn']:raise ValueError('Ground-truth precision leak')
            if any(v['nonfinite'] or v['dtype']!='torch.float32' for v in reference['output']):raise ValueError('Invalid FP32 output')
        cases=[]
        for case in data['cases']:
            if case['role']!='candidate' or 'vs_fp32_ground_truth' not in case:raise ValueError('Candidate has no FP32 comparison')
            cases.append({k:v for k,v in case.items() if k!='operators'})
        result.append(dict(file=name,sha256=digest(path),configuration=data['configuration'],distribution=data.get('distribution'),
            fp32_ground_truth=gold,cases=cases))
    selected=[c for r in result for c in r['cases'] if c['name']=='tp_fp16_fold_c64']
    if len(selected)!=6 or any(c['output']['nonfinite'] or not c['rank_equal'] or not c['repeat_equal'] for c in selected):
        raise ValueError('Selected bounded FP16 range/repeat gate failed')
    return dict(status='out_of_scope_forced_dense_fp32; not_native_fp32_tensor_flow',environment=environment,
        ground_truth='Dense FP32 execution on Native-dequantized fixed checkpoint weights, TF32 disabled; no INT8 activation GEMMs or attention',
        bf16_role='candidate, never ground truth',fp16_role='candidate, never ground truth',
        limitations=['Original pre-quantization FP32 H3 weights are unavailable; fixed INT8 checkpoint quantization error is outside this ground truth.',
                     'Local SM120 accuracy only. No full H3, denoising trajectory, perceptual acceptance, T4/NCCL or performance qualification.',
                     'TE inputs are synthetic and half-rounded; late layers reuse other-layer inputs as stress probes.'],
        case_count=sum(len(r['cases']) for r in result),records=result)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=Path('D:/attention_eval/h3_fp32_ground_truth_20261021'))
    parser.add_argument('--output',type=Path,default=ROOT/'docs/data/h3_fp32_ground_truth_20261021.json')
    args=parser.parse_args();result=collect(args.root)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    for record in result['records']:
        selected=[c for c in record['cases'] if c['name'] in ('tp_fp32_w8a8_control','tp_bf16_w8a8','tp_fp16_fold_c64')]
        if selected:print(record['file'],[(c['name'],c['vs_fp32_ground_truth']) for c in selected],flush=True)
    print('FP32_GROUND_TRUTH_VERIFIED',result['case_count'],flush=True)


if __name__=='__main__':main()
