"""Fresh source-bound CPU reuse/preflight and bounded current-default T4 batch."""
import argparse
import ast
import base64
import hashlib
import json
from pathlib import Path
import sys
import build_kaggle_native_torch_jobs as source


def main():
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['cpu','gpu'])
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--preflight',type=Path,required=True)
    parser.add_argument('--assets-receipt',type=Path);parser.add_argument('--prior-result',type=Path)
    args=parser.parse_args()
    if args.phase=='cpu':
        if args.output.exists():raise FileExistsError('Preserve submitted sources')
        source.BENCHMARKS+=['run_h3_current_cpu_preflight.py','test_h3_row_partial_combined.py',
            'test_h3_reference_loading.py','test_h3_loading_stages.py']
        data,hashes=source.capsule();args.output.mkdir(parents=True)
        digest=hashlib.sha256(args.preflight.read_bytes()).hexdigest()
        code='import base64,hashlib,io,json,pathlib,subprocess,sys,tarfile\n'
        code+="OUT=pathlib.Path('/kaggle/working/native_profile');OUT.mkdir(exist_ok=True)\nSRC=pathlib.Path('/kaggle/temp/h3_revalidated_source');SRC.mkdir(parents=True,exist_ok=True)\n"
        code+='payload=base64.b64decode('+repr(base64.b64encode(data).decode())+')\n'
        code+=f"assert hashlib.sha256(payload).hexdigest()=={hashlib.sha256(data).hexdigest()!r}\n"
        code+="with tarfile.open(fileobj=io.BytesIO(payload),mode='r:gz') as tar:tar.extractall(SRC,filter='data')\n"
        code+=f"(OUT/'source_hashes.json').write_text(json.dumps({hashes!r},indent=2))\n"
        code+=f"subprocess.run([sys.executable,str(SRC/'benchmarks/run_h3_current_cpu_preflight.py'),'--source',str(SRC),'--prior-sha256',{digest!r},'--output',str(OUT)],check=True,timeout=360)\n"
        metadata=dict(id='pamcks/h3-rowpartial-reference-cpu-preflight',title='H3 RowPartial Reference CPU Preflight',
            code_file='experiment.ipynb',language='python',kernel_type='notebook',is_private=True,enable_gpu=False,
            enable_tpu=False,enable_internet=False,docker_image_pinning_type='original',
            dataset_sources=['pamcks/h3-frozen-runtime-20261021'],kernel_sources=['pamcks/h3-native-torch-cpu-preflight'],
            competition_sources=[],model_sources=[])
        ast.parse(code)
        notebook=dict(nbformat=4,nbformat_minor=5,metadata=dict(kernelspec=dict(name='python3',language='python',display_name='Python 3')),
            cells=[dict(id='current-core-cpu',cell_type='code',execution_count=None,metadata={},outputs=[],source=code.splitlines(keepends=True))])
        (args.output/'experiment.ipynb').write_text(json.dumps(notebook))
        (args.output/'kernel-metadata.json').write_text(json.dumps(metadata,indent=2))
        (args.output/'source_binding.json').write_text(json.dumps(dict(files_sha256=hashes,reused_cpu_receipt_sha256=digest),indent=2))
    else:
        import build_t4_component_followup as previous
        source.BENCHMARKS+=['run_t4_row_partial_combined.py','run_t4_reference_demand_vae.py']
        sys.argv=['builder','--diagnosis-only','--output',str(args.output),'--preflight',str(args.preflight),
            '--assets-receipt',str(args.assets_receipt),'--prior-result',str(args.prior_result)]
        previous.main()
        path=args.output/'experiment.ipynb';notebook=json.loads(path.read_text());code=''.join(notebook['cells'][0]['source'])
        code=code.replace('BUDGET=240\n','BUDGET=600\n')
        start=code.index("    run([*launcher,str(SRC/'benchmarks/run_t4_fp16_lora_diagnosis.py')")
        stop=code.index('    required=',start)
        code=code[:start]+'''    run([*launcher,str(SRC/'benchmarks/run_t4_row_partial_combined.py'),'--assets',str(ASSETS),
        '--output',str(OUT/'combined_rows.json')],'combined_row_partial',300)
    run([sys.executable,str(SRC/'benchmarks/run_t4_reference_demand_vae.py'),'--assets',str(ASSETS),
        '--output',str(OUT/'reference_demand_vae.json')],'reference_demand_vae',150)
    nsys=shutil.which('nsys')
    if not nsys and pathlib.Path('/opt/nvidia').exists():
        nsys=next((str(p) for p in pathlib.Path('/opt/nvidia').rglob('nsys') if p.is_file()),None)
    state['profiling']=dict(nsys=nsys,ncu='Not collected; unchanged historical GPU-counter permission blocker')
    if nsys and deadline-time.monotonic()>75:
        ok=run([nsys,'profile','--trace=cuda,nvtx,osrt','--sample=none','--cpuctxsw=none','--force-overwrite=true',
            '-o',str(OUT/'combined_rows_nsys'),*launcher,str(SRC/'benchmarks/run_t4_row_partial_combined.py'),
            '--assets',str(ASSETS),'--trace-only','--output',str(OUT/'systems_rows.json')],'combined_nsight_systems',75)
        if ok:run([nsys,'stats','--report','cuda_gpu_kern_sum,cuda_api_sum,cuda_gpu_mem_time_sum',
            '--format','csv','--output',str(OUT/'combined_rows_stats'),str(OUT/'combined_rows_nsys.nsys-rep')],'nsight_statistics',20)
    else:state['profiling']['nsys_blocker']='Executable unavailable or bounded run budget insufficient'
'''+code[stop:]
        from t4_notebook_watchdog import wrap_notebook
        code=wrap_notebook(code,budget=600)
        ast.parse(code);notebook['cells'][0]['source']=code.splitlines(keepends=True);path.write_text(json.dumps(notebook))
        metadata=json.loads((args.output/'kernel-metadata.json').read_text())
        metadata.update(id='pamcks/h3-rowpartial-reference-t4',title='H3 RowPartial Reference T4',machine_shape='NvidiaTeslaT4')
        metadata['kernel_sources']=[s.replace('pamcks/h3-native-torch-cpu-preflight','pamcks/h3-rowpartial-reference-cpu-preflight') for s in metadata['kernel_sources']]
        (args.output/'kernel-metadata.json').write_text(json.dumps(metadata,indent=2))
        binding=json.loads((args.output/'source_binding.json').read_text());binding['scope']='Current production FP16-fix/fused-stored/scaled-FP16 rows, native-gather vs FP32 RowPartial, reference-demand VAE; no H3 sampler'
        binding['notebook_watchdog']=dict(deadline_seconds=600,includes='Source preparation, offline extraction/install, device gate and all experiment subprocesses',
            excludes='Provider startup and input mounting before notebook execution',
            generator_sha256=hashlib.sha256(Path(__file__).with_name('t4_notebook_watchdog.py').read_bytes()).hexdigest())
        (args.output/'source_binding.json').write_text(json.dumps(binding,indent=2))
    print(args.output)


if __name__=='__main__':main()
