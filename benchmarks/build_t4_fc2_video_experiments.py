"""One bounded TP2 batch: FP32 preparation/fusion/transport, independent Video localization."""
import argparse
import ast
import json
from pathlib import Path
import sys
import build_kaggle_native_torch_jobs as source
import build_t4_component_followup as previous


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--preflight',type=Path,required=True);parser.add_argument('--assets-receipt',type=Path,required=True)
    parser.add_argument('--prior-result',type=Path,required=True);args=parser.parse_args()
    source.BENCHMARKS+=['h3_fp16_fc2_experiment.py','run_t4_fc2_fusion.py','run_t4_video_split_localize.py']
    sys.argv=['build_t4_component_followup','--diagnosis-only','--output',str(args.output),
        '--preflight',str(args.preflight),'--assets-receipt',str(args.assets_receipt),'--prior-result',str(args.prior_result)]
    previous.main()
    path=args.output/'experiment.ipynb';notebook=json.loads(path.read_text());code=''.join(notebook['cells'][0]['source'])
    code=code.replace('BUDGET=240\n','BUDGET=480\n')
    start=code.index("    run([*launcher,str(SRC/'benchmarks/run_t4_fp16_lora_diagnosis.py')")
    stop=code.index('    required=',start)
    code=code[:start]+'''    run([*launcher,str(SRC/'benchmarks/run_t4_fc2_fusion.py'),'--assets',str(ASSETS),
        '--repeats','7','--output',str(OUT/'fc2_fusion.json')],'fp16_fp32prep_fusion_transport',270)
    run([*launcher,str(SRC/'benchmarks/run_t4_video_split_localize.py'),'--assets',str(ASSETS),
        '--output',str(OUT/'video_split_localization.json')],'video_261_localization',150)
'''+code[stop:]
    ast.parse(code);notebook['cells'][0]['source']=code.splitlines(keepends=True);path.write_text(json.dumps(notebook))
    meta_path=args.output/'kernel-metadata.json';meta=json.loads(meta_path.read_text())
    meta.update(id='pamcks/h3-fp16-fc2-video-localization',title='H3 FP16 FC2 Video Localization')
    meta_path.write_text(json.dumps(meta,indent=2))
    binding_path=args.output/'source_binding.json';binding=json.loads(binding_path.read_text())
    binding['scope']='FP16 residual + FP32 SwiGLU/regular ConvRot/quant; original INT8 GEMM, INT32 or scaled FP16 SUM; independent 261-token Video split localization'
    binding_path.write_text(json.dumps(binding,indent=2))


if __name__=='__main__':main()
