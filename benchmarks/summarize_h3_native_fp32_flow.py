"""Validate the corrected Native FP32-flow study; reject substituted operators."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILES = ('dit_513.json', 'dit_7895.json', 'dit_layer24_stress.json',
         'dit_layer49_stress.json', 'te_layer0.json', 'te_layer49_stress.json',
         'dit_lora_513.json', 'frontend_native_components.json')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def collect(root):
    records = []
    environment = None
    for name in FILES:
        path = root/'final'/name
        data = json.loads(path.read_text(encoding='utf-8'))
        if data['schema'] != 'native-fp32-flow-v1' or data['status'] != 'complete':
            raise ValueError('Invalid record: '+name)
        current = {k: data[k] for k in ('hardware', 'capability', 'torch', 'runtime_lock_sha256', 'tf32_matmul', 'tf32_cudnn')}
        if environment is None:
            environment = current
        elif current != environment:
            raise ValueError('Mixed execution environments')
        for source, expected in data['source_sha256'].items():
            paths = [ROOT/source, *root.glob('sources/*/'+source)]
            if not any(p.is_file() and digest(p) == expected for p in paths):
                raise ValueError('Unpreserved source: '+source)
        reference, *candidates = data['cases']
        if reference['name'] != 'native_fp32_flow' or reference['role'] != 'reference':
            raise ValueError('The existing FP32 flow must be the reference')
        errors = reference['vs_native_fp32_flow']
        if isinstance(errors, dict):
            errors = [errors]
        if any(c['relative_l2'] != 0 or c['nonfinite'] for c in errors):
            raise ValueError('Invalid reference self-check')
        for case in data['cases']:
            if name == 'frontend_native_components.json':
                if not case['native_frontend_dense_weights'] or data['frontend_checkpoint_dtypes'] != ['BF16', 'F32']:
                    raise ValueError('Native frontend checkpoint contract changed')
            else:
                if not case['int8_linear_calls']:
                    raise ValueError('INT8 GEMM was bypassed')
                if not case['int8_weight_bytes_equal']:
                    raise ValueError('INT8 weights changed')
                if case['name'] in ('native_fp32_flow', 'bf16_flow', 'fp16_fold_c64'):
                    if not case['rank_equal'] or not case['repeat_equal'] or case['output']['nonfinite']:
                        raise ValueError('Reference/selected finite repeat gate failed')
        if any(case['role'] != 'candidate' for case in candidates):
            raise ValueError('BF16/FP16 must remain candidates')
        records.append(dict(file=name, sha256=digest(path), configuration=data['configuration'],
            distribution=data['distribution'], operator_policy=data.get('operator_policy'),
            cases=[{k: v for k, v in case.items() if k != 'operators'} for case in data['cases']]))
    return dict(schema='native-fp32-flow-summary-v1',
        reference='Existing legacy FP32 activation flow with current INT8 operators and mixed-precision islands retained',
        bf16_role='candidate', fp16_role='candidate',
        excluded='Forced dense-FP32 study; BF16-as-truth historical comparisons',
        environment=environment, case_count=sum(len(r['cases']) for r in records), records=records,
        limitations=['Local SM120 with logical TP2, not physical T4/NCCL or kernel performance qualification.',
            'SM75-only TE Group32, packed rows and fused SwiGLU use existing native fallbacks locally.',
            'DiT input reconstructs retained first block; late layers reuse it as stress probes. TE inputs are synthetic.',
            'Vision/Video retain Native FP16; Audio retains Native FP32. No all-FP32 conversion.',
            'No full H3, denoising trajectory, video/audio quality or production acceptance evidence.'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('D:/attention_eval/h3_native_fp32_flow_20261021'))
    parser.add_argument('--output', type=Path, default=ROOT/'docs/data/h3_native_fp32_flow_20261021.json')
    args = parser.parse_args()
    data = collect(args.root)
    args.output.write_text(json.dumps(data, indent=2, allow_nan=False), encoding='utf-8')
    print('NATIVE_FP32_FLOW_VERIFIED', data['case_count'])


if __name__ == '__main__':
    main()
