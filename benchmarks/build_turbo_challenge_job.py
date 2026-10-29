"""Fresh CPU gate and one authorized regular Turbo E2E; zero mounts, no polling."""
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
from turbo_challenge_contract import validate
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=('cpu','gpu'),required=True)
    p.add_argument('--publication',type=Path,required=True);p.add_argument('--evidence',type=Path,required=True)
    p.add_argument('--plan',type=Path,required=True);p.add_argument('--reference-directory',type=Path,required=True)
    p.add_argument('--reference-mirror',type=Path,required=True)
    p.add_argument('--qualification',type=Path);p.add_argument('--input-qualification',type=Path)
    p.add_argument('--submit',action='store_true');args=p.parse_args()
    args.evidence.mkdir(parents=True,exist_ok=True);job=args.evidence/(args.mode+'_job')
    pub=json.loads(args.publication.read_text());plan=json.loads(args.plan.read_text());validate(plan)
    argv=[sys.executable,str(Path(__file__).with_name('build_kaggle_light_job.py')),
        '--mode','cpu' if args.mode=='cpu' else 'e2e','--publication',str(args.publication),
        '--output',str(job),'--setup-budget','600']
    if args.mode=='cpu':argv+=['--review-suite']
    else:
        if not args.qualification or not args.input_qualification:p.error('Fresh matching CPU setup and workload gates required')
        setup=json.loads(args.qualification.read_text());gate=json.loads(args.input_qualification.read_text())
        assert setup['status']=='cpu_ready' and setup['review_suite']['passed'] and setup['disk_bake_cpu']['passed']
        assert gate['status']=='turbo_challenge_cpu_ready' and gate['no_weights_fetched'] and not gate['gpu_used']
        assert setup['capsule_sha256']==gate['capsule_sha256']==pub['capsule_sha256']
        assert gate['workload']==validate(plan) and gate['lora_targets']==208
        argv+=['--qualification',str(args.qualification)]
    subprocess.run(argv,check=True)
    nbpath=job/'experiment.ipynb';nb=json.loads(nbpath.read_text());code=''.join(nb['cells'][0]['source'])
    if args.mode=='gpu':
        marker="assert receipt['status']=='ready_for_device_gate'\n";assert code.count(marker)==1
        code=code[:code.index(marker)+len(marker)]
    destination='/kaggle/temp/h3_light_source/bench_assets/l1_current'
    mirrors=json.loads(args.reference_mirror.read_text())['files'];refs=plan['reference_images']
    mirrored=[r for r in refs if r['file'] in mirrors];private=[r for r in refs if r['file'] not in mirrors]
    for r in mirrored:
        assert (mirrors[r['file']]['sha256'],mirrors[r['file']]['bytes'])==(r['sha256'],r['bytes'])
    code+=private_reference_prefix(private,args.reference_directory,destination)
    code+=mirror_reference_prefix(mirrored,{r['file']:mirrors[r['file']]['url'] for r in mirrored},destination)
    script='qualify_turbo_challenge.py' if args.mode=='cpu' else 'run_h3_l1_turbo_batch.py'
    out='turbo_challenge_cpu' if args.mode=='cpu' else 'turbo_1mp_e2e'
    execution=f"""import json,os,pathlib,sys
env=dict(os.environ);env.update(json.loads(pathlib.Path('/kaggle/working/light_setup/execution_env.json').read_text()))
source=pathlib.Path('/kaggle/temp/h3_light_source')
argv=[sys.executable,'-u',str(source/'benchmarks/{script}'),'--source',str(source),'--output','/kaggle/working/{out}']
if {args.mode=='gpu'!r}:argv+=['--plan',str(source/'turbo_challenge_plan.json'),'--scratch','/tmp/h3_turbo_1mp_10s_20261029','--e2e-timeout','5400']
os.execve(sys.executable,argv,env)
"""
    code+=supervise_worker(execution,budget=240 if args.mode=='cpu' else 7200,
                           output='/kaggle/working/turbo_challenge_supervision')
    expected='turbo_challenge_cpu_ready' if args.mode=='cpu' else 'complete'
    filename='result.json' if args.mode=='cpu' else 'batch_results.json'
    code+=f"assert json.loads(pathlib.Path('/kaggle/working/{out}/{filename}').read_text())['status']=={expected!r}\n"
    ast.parse(code);assert not re.search(r'(?:hf_[A-Za-z0-9]{25,}|gh[pousr]_[A-Za-z0-9]{30,})',code)
    nb['cells'][0]['source']=code.splitlines(keepends=True);nbpath.write_text(json.dumps(nb),encoding='utf-8')
    assert nbpath.stat().st_size<9_900_000
    meta=json.loads((job/'kernel-metadata.json').read_text());label='cpu' if args.mode=='cpu' else 't4'
    meta.update(id=f'pamcks/h3-turbo-1mp-10s-{label}-20261029',title=f'H3 Turbo 1MP 10s {label.upper()} 20261029')
    assert meta['is_private'] and meta['dataset_sources']==meta['kernel_sources']==[]
    (job/'kernel-metadata.json').write_text(json.dumps(meta,indent=2))
    binding=json.loads((job/'source_binding.json').read_text())
    binding.update(full_pipeline=args.mode=='gpu',full_pipeline_authorized=True,ptd=False,online_lora=False,
        credentials=False,request_contract=validate(plan),plan_sha256=hashlib.sha256(args.plan.read_bytes()).hexdigest(),
        input_qualification_sha256=hashlib.sha256(args.input_qualification.read_bytes()).hexdigest() if args.input_qualification else None,
        reference_sha256=[r['sha256'] for r in refs],trusted_bake_default=True)
    (job/'source_binding.json').write_text(json.dumps(binding,indent=2))
    if not args.submit:return
    timeout=1100 if args.mode=='cpu' else 8100
    state=dict(kernel=meta['id'],status='submitting',submitted_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        source_binding=binding,run_timeout_seconds=timeout,earliest_final_collection_seconds=timeout,
        final_collection_count=0,result_relative=f'{out}/{filename}',
        notebook_sha256=hashlib.sha256(nbpath.read_bytes()).hexdigest())
    receipt=args.evidence/(args.mode+'_submission.json')
    with receipt.open('x') as f:json.dump(state,f,indent=2)
    argv=[sys.executable,'-m','kaggle','kernels','push','-p',str(job),'-t',str(timeout)]
    if args.mode=='gpu':argv+=['--accelerator','NvidiaTeslaT4']
    try:
        process=subprocess.run(argv,capture_output=True,text=True,timeout=180);output=process.stdout+process.stderr
        state.update(status='accepted' if process.returncode==0 else 'acceptance_uncertain',returncode=process.returncode)
        match=re.search(r'Kernel version (\d+)',output)
        if match:state['version']=int(match.group(1))
        match=re.search(r'https://www\.kaggle\.com/code/[\w-]+/[\w-]+',output)
        if match:state['url']=match.group(0)
    except subprocess.TimeoutExpired:
        output='Acknowledgement timeout; preserve claim, do not resubmit automatically.'
        state.update(status='acceptance_uncertain',returncode=-1)
    receipt.write_text(json.dumps(state,indent=2));(args.evidence/(args.mode+'_submission_cli.log')).write_text(output,encoding='utf-8')
    print(output[-1500:])


if __name__=='__main__':main()
