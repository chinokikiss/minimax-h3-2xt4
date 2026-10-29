"""Private zero-mount CPU gate / single PTD E2E T4 job; never poll Kaggle."""
import argparse
import ast
import datetime as dt
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
from private_reference_delivery import private_reference_prefix,mirror_reference_prefix
from t4_notebook_watchdog import supervise_worker


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--mode',choices=('cpu','gpu'),required=True)
    parser.add_argument('--publication',type=Path,required=True); parser.add_argument('--evidence',type=Path,required=True)
    parser.add_argument('--plan',type=Path,required=True); parser.add_argument('--reference-directory',type=Path,required=True)
    parser.add_argument('--reference-mirror',type=Path,required=True)
    parser.add_argument('--qualification',type=Path); parser.add_argument('--ptd-qualification',type=Path)
    parser.add_argument('--submit',action='store_true'); args=parser.parse_args()
    args.evidence.mkdir(parents=True,exist_ok=True); job=args.evidence/(args.mode+'_job')
    publication=json.loads(args.publication.read_text(encoding='utf-8'))
    plan=json.loads(args.plan.read_text(encoding='utf-8'))
    from ptd_e2e_contract import validate
    validate(plan)
    command=[sys.executable,str(Path(__file__).with_name('build_kaggle_light_job.py')),
        '--mode','cpu' if args.mode=='cpu' else 'e2e','--publication',str(args.publication),
        '--output',str(job),'--setup-budget','600']
    if args.mode=='cpu': command+=['--review-suite']
    else:
        if not args.qualification or not args.ptd_qualification:
            parser.error('GPU requires this immutable source\'s successful CPU setup and PTD E2E gate')
        setup=json.loads(args.qualification.read_text(encoding='utf-8'))
        gate=json.loads(args.ptd_qualification.read_text(encoding='utf-8'))
        assert setup['status']=='cpu_ready' and setup['review_suite']['passed']
        assert gate['status']=='ptd_e2e_cpu_ready' and gate['no_weights_fetched'] and not gate['gpu_used']
        assert gate['native_sampler_integration'] and gate['auxiliary_layout']=='monolithic'
        assert setup['capsule_sha256']==gate['capsule_sha256']==publication['capsule_sha256']
        command+=['--qualification',str(args.qualification)]
    subprocess.run(command,check=True)
    notebook_path=job/'experiment.ipynb'; notebook=json.loads(notebook_path.read_text(encoding='utf-8'))
    code=''.join(notebook['cells'][0]['source'])
    if args.mode=='gpu':
        marker="assert receipt['status']=='ready_for_device_gate'\n"
        assert code.count(marker)==1; code=code[:code.index(marker)+len(marker)]
    destination='/kaggle/temp/h3_light_source/bench_assets/l1_current'
    mirrors=json.loads(args.reference_mirror.read_text(encoding='utf-8'))['files']
    rows=plan['reference_images']; mirrored=[r for r in rows if r['file'] in mirrors]
    private=[r for r in rows if r['file'] not in mirrors]
    # The original publicly sourced character JPEG stays inside this private
    # notebook; no token or new public redistribution of its pixels is needed.
    code+=private_reference_prefix(private,args.reference_directory,destination)
    code+=mirror_reference_prefix(mirrored,{r['file']:mirrors[r['file']]['url'] for r in mirrored},destination)
    script='qualify_ptd_e2e.py' if args.mode=='cpu' else 'run_ptd_e2e_batch.py'
    result='ptd_e2e_cpu' if args.mode=='cpu' else 'ptd_e2e'
    execution=f"""import json,os,pathlib,sys
env=dict(os.environ);env.update(json.loads(pathlib.Path('/kaggle/working/light_setup/execution_env.json').read_text()))
source=pathlib.Path('/kaggle/temp/h3_light_source')
argv=[sys.executable,'-u',str(source/'benchmarks/{script}'),'--source',str(source),
 '--output','/kaggle/working/{result}']
os.execve(sys.executable,argv,env)
"""
    code+=supervise_worker(execution,budget=300 if args.mode=='cpu' else 10800,
                           output='/kaggle/working/ptd_e2e_supervision')
    expected='ptd_e2e_cpu_ready' if args.mode=='cpu' else 'complete'
    filename='result.json' if args.mode=='cpu' else 'batch_results.json'
    code+=f"assert json.loads(pathlib.Path('/kaggle/working/{result}/{filename}').read_text())['status']=={expected!r}\n"
    ast.parse(code)
    assert not re.search(r'(?:hf_[A-Za-z0-9]{25,}|gh[pousr]_[A-Za-z0-9]{30,})',code)
    notebook['cells'][0]['source']=code.splitlines(keepends=True)
    notebook_path.write_text(json.dumps(notebook),encoding='utf-8')
    assert notebook_path.stat().st_size<9_900_000
    metadata=json.loads((job/'kernel-metadata.json').read_text(encoding='utf-8'))
    label='cpu' if args.mode=='cpu' else 't4'
    metadata.update(id=f'pamcks/h3-ptd-long-e2e-{label}-20261029',title=f'H3 PTD Long E2E {label.upper()} 20261029')
    assert metadata['is_private'] and metadata['dataset_sources']==metadata['kernel_sources']==[]
    (job/'kernel-metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    binding=json.loads((job/'source_binding.json').read_text(encoding='utf-8'))
    binding.update(full_pipeline=args.mode=='gpu',full_pipeline_authorized=True,production_layers=50,expert_hops=8,
        auxiliary_layout='monolithic',online_lora=False,private_references=True,credentials=False,
        prefetch_refiner=plan.get('prefetch_refiner',True),
        request_contract=validate(plan),plan_sha256=hashlib.sha256(args.plan.read_bytes()).hexdigest(),
        execution_budget=300 if args.mode=='cpu' else 10800,
        ptd_qualification_sha256=hashlib.sha256(args.ptd_qualification.read_bytes()).hexdigest() if args.ptd_qualification else None)
    (job/'source_binding.json').write_text(json.dumps(binding,indent=2),encoding='utf-8')
    if not args.submit: return
    timeout=1100 if args.mode=='cpu' else 11700
    state=dict(kernel=metadata['id'],status='submitting',submitted_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        source_binding=binding,run_timeout_seconds=timeout,earliest_final_collection_seconds=timeout,
        final_collection_count=0,
        result_relative=f'{result}/{filename}',notebook_sha256=hashlib.sha256(notebook_path.read_bytes()).hexdigest())
    receipt=args.evidence/(args.mode+'_submission.json')
    with receipt.open('x',encoding='utf-8') as stream: json.dump(state,stream,indent=2)
    argv=[sys.executable,'-m','kaggle','kernels','push','-p',str(job),'-t',str(timeout)]
    if args.mode=='gpu': argv+=['--accelerator','NvidiaTeslaT4']
    try:
        process=subprocess.run(argv,capture_output=True,text=True,timeout=180)
        text=process.stdout+process.stderr
        state.update(status='accepted' if process.returncode==0 else 'acceptance_uncertain',returncode=process.returncode)
        version=re.search(r'Kernel version (\d+)',text)
        if version: state['version']=int(version.group(1))
        url=re.search(r'https://www\.kaggle\.com/code/[\w-]+/[\w-]+',text)
        if url: state['url']=url.group(0)
    except subprocess.TimeoutExpired:
        text='Acknowledgement timeout; preserve submission claim, do not duplicate automatically.'
        state.update(status='acceptance_uncertain',returncode=-1)
    receipt.write_text(json.dumps(state,indent=2),encoding='utf-8')
    (args.evidence/(args.mode+'_submission_cli.log')).write_text(text,encoding='utf-8')
    print(text[-1500:])


if __name__=='__main__': main()
