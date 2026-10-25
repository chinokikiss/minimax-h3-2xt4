"""Bind a narrow follow-up to reviewed v1 evidence, reusing its frozen setup."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys
import build_kaggle_native_torch_jobs as source
import build_t4_native_validation_batch as batch


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--preflight',type=Path,required=True);parser.add_argument('--assets-receipt',type=Path,required=True)
    parser.add_argument('--prior-result',type=Path,required=True)
    parser.add_argument('--diagnosis-only',action='store_true');args=parser.parse_args()
    prior=json.loads(args.prior_result.read_text())
    assert prior['operator_gates']=={'attention':True,'rows':True}
    assert len(prior['hardware']['devices'])==2 and all(d['capability']==[7,5] for d in prior['hardware']['devices'])
    source.BENCHMARKS.append('run_t4_fp16_lora_diagnosis.py')
    sys.argv=['build_t4_native_validation_batch','--output',str(args.output),'--preflight',str(args.preflight),
        '--assets-receipt',str(args.assets_receipt),'--budget-seconds','240' if args.diagnosis_only else '480']
    batch.main()
    path=args.output/'experiment.ipynb';notebook=json.loads(path.read_text());code=''.join(notebook['cells'][0]['source'])
    # Named template boundaries preserve the exact reviewed offline setup and
    # device gate while replacing the broad experiment schedule only.
    start=code.index('    launcher=');stop=code.index('    components=')
    digest=hashlib.sha256(args.prior_result.read_bytes()).hexdigest()
    followup=f'''    prior_paths=[p for p in pathlib.Path('/kaggle/input').rglob('batch_results.json') if hashlib.sha256(p.read_bytes()).hexdigest()=={digest!r}]
    assert len(prior_paths)==1,'Reviewed v1 result missing or changed'
    prior=json.loads(prior_paths[0].read_text())
    assert state['hardware']['torch']==prior['hardware']['torch'],'Native GPU Torch changed between runs'
    state['qualification_reference']=dict(kernel='pamcks/h3-native-torch-t4-validation',version=1,receipt_sha256={digest!r},operator_gates=prior['operator_gates'])
    launcher=[sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=2']
    run([*launcher,str(SRC/'benchmarks/run_t4_fp16_lora_diagnosis.py'),'--assets',str(ASSETS),
        '--output',str(OUT/'fp16_lora_diagnosis.json')],'fp16_lora_diagnosis',150)
'''
    code=code[:start]+followup+code[stop:]
    start=code.index("    run([sys.executable,str(SRC/'benchmarks/run_physical_layer_bake.py')")
    stop=code.index("    run([*components,'vae'",start)
    code=code[:start]+code[stop:]
    code=code.replace("'native_refiner',120)","'native_refiner',60)").replace("'vae_idle_and_native_regressions',420)","'vae_idle_and_native_regressions',300)")
    start=code.index('    if nsys and attention');stop=code.index('    required=',start)
    code=code[:start]+'''    reports=list(prior_paths[0].parent.glob('*.nsys-rep'))
    if nsys and len(reports)==1:
        # CPU analysis of the existing Systems capture; no new profile/replay.
        report=OUT/'v1_te.nsys-rep';shutil.copyfile(reports[0],report)
        run([nsys,'stats','--report','cuda_gpu_kern_sum,cuda_api_sum,cuda_gpu_mem_time_sum',
             '--format','csv','--output',str(OUT/'v1_te_stats'),str(report)],'v1_nsight_statistics',45)
    else:state['profiling']['nsys_blocker']='Reviewed Systems capture or executable unavailable'
'''+code[stop:]
    code=code.replace("state['status']='passed' if attention and rows and all", "state['status']='passed' if all")
    if args.diagnosis_only:
        start=code.index('    components=');stop=code.index('    required=',start)
        code=code[:start]+code[stop:]
    ast.parse(code);notebook['cells'][0]['source']=code.splitlines(keepends=True)
    path.write_text(json.dumps(notebook))
    meta_path=args.output/'kernel-metadata.json';meta=json.loads(meta_path.read_text())
    meta.update(id='pamcks/h3-t4-component-followup',title='H3 T4 Component Followup')
    if args.diagnosis_only:meta.update(id='pamcks/h3-t4-fp16-stage-diagnosis',title='H3 T4 FP16 Stage Diagnosis')
    meta['kernel_sources'].append('pamcks/h3-native-torch-t4-validation')
    meta_path.write_text(json.dumps(meta,indent=2))
    binding_path=args.output/'source_binding.json';binding=json.loads(binding_path.read_text())
    binding.update(prior_result_sha256=digest,scope='FP16 stage diagnosis only' if args.diagnosis_only else
        'Failed components and FP16 stress diagnosis only; broad gates/bake are not repeated')
    binding_path.write_text(json.dumps(binding,indent=2))


if __name__=='__main__':main()
