"""Build one CPU API gate or one bounded two-T4 PTD fixture job; no polling."""
import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

from build_cu128_dispatch_job import assert_operator_only
from t4_notebook_watchdog import supervise_worker


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--mode',choices=('cpu','gpu'),required=True)
    parser.add_argument('--publication',type=Path,required=True)
    parser.add_argument('--evidence',type=Path,required=True)
    parser.add_argument('--qualification',type=Path)
    parser.add_argument('--ptd-qualification',type=Path)
    parser.add_argument('--submit',action='store_true')
    parser.add_argument('--cpu-scheduling',action='store_true')
    args=parser.parse_args(); args.evidence.mkdir(parents=True,exist_ok=True)
    job=args.evidence/(args.mode+'_job'); publication=json.loads(args.publication.read_text())
    command=[sys.executable,str(Path(__file__).with_name('build_kaggle_light_job.py')),
        '--mode','cpu' if args.mode=='cpu' else 'e2e','--publication',str(args.publication),
        '--output',str(job),'--setup-budget','600']
    if args.mode=='cpu': command+=['--review-suite']
    else:
        if not args.qualification or not args.ptd_qualification:
            parser.error('GPU submission requires matching frozen setup and PTD CPU API receipts')
        setup=json.loads(args.qualification.read_text()); gate=json.loads(args.ptd_qualification.read_text())
        assert setup['status']=='cpu_ready' and setup['review_suite']['passed']
        assert gate['status']=='ptd_cpu_ready' and gate['no_weights_fetched']
        if args.cpu_scheduling: assert gate.get('cpu_scheduling') is True and gate['scheduling_contract_tests'] >= 8
        assert setup['capsule_sha256']==gate['capsule_sha256']==publication['capsule_sha256']
        command+=['--qualification',str(args.qualification)]
    subprocess.run(command,check=True)
    notebook_path=job/'experiment.ipynb'; notebook=json.loads(notebook_path.read_text())
    code=''.join(notebook['cells'][0]['source'])
    if args.mode=='gpu':
        marker="assert receipt['status']=='ready_for_device_gate'\n"
        assert code.count(marker)==1
        code=code[:code.index(marker)+len(marker)]
    execution="""import json,os,pathlib,sys
env=dict(os.environ);env.update(json.loads(pathlib.Path('/kaggle/working/light_setup/execution_env.json').read_text()))
source=pathlib.Path('/kaggle/temp/h3_light_source')
argv=[sys.executable,'-u',str(source/'benchmarks/run_ptd_tp2_t4.py'),'--source',str(source),
 '--output','/kaggle/working/ptd_qualification']
EXTRA
os.execve(sys.executable,argv,env)
""".replace('EXTRA',"argv+=['--cpu-preflight']" if args.mode=='cpu' else '')
    if args.cpu_scheduling:execution=execution.replace('os.execve(',"argv+=['--cpu-scheduling']\nos.execve(")
    code+=supervise_worker(execution,budget=180 if args.mode=='cpu' else 1800,
                           output='/kaggle/working/ptd_supervision')
    code+="assert json.loads(pathlib.Path('/kaggle/working/ptd_qualification/result.json').read_text())['status'] in ('ptd_cpu_ready','complete')\n"
    assert_operator_only(code)
    notebook['cells'][0]['source']=code.splitlines(keepends=True)
    notebook_path.write_text(json.dumps(notebook),encoding='utf-8')
    metadata=json.loads((job/'kernel-metadata.json').read_text())
    label='cpu' if args.mode=='cpu' else 't4'
    metadata.update(id=f'pamcks/h3-ptd-tp2-{label}-20261028',title=f'H3 PTD TP2 {label.upper()} 20261028')
    if args.cpu_scheduling:
        metadata.update(id=f'pamcks/h3-ptd-cpu-schedule-{label}-20261029',
            title=f'H3 PTD CPU Schedule {label.upper()} 20261029')
    assert metadata['dataset_sources']==metadata['kernel_sources']==[]
    (job/'kernel-metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    binding=json.loads((job/'source_binding.json').read_text())
    binding.update(full_pipeline=False,private_inputs=False,credentials=False,
        fixture_layers=2,fixture_hops=8,production_bank=False,
        handoff_order=['rebuild','queue','queue','rebuild'] if args.mode=='gpu' else None,
        ptd_qualification_sha256=hashlib.sha256(args.ptd_qualification.read_bytes()).hexdigest()
            if args.ptd_qualification else None)
    if args.cpu_scheduling:
        from run_ptd_tp2_t4 import SCHEDULING_CASES
        binding.update(cpu_scheduling=True,scheduling_cases=list(SCHEDULING_CASES),handoff_order=None)
    (job/'source_binding.json').write_text(json.dumps(binding,indent=2),encoding='utf-8')
    if not args.submit:return
    timeout=1000 if args.mode=='cpu' else 2500
    state=dict(kernel=metadata['id'],status='submitting',submitted_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        source_binding=binding,final_collection_count=0,run_timeout_seconds=timeout,
        earliest_final_collection_seconds=timeout,
        result_relative='ptd_qualification/result.json',
        notebook_sha256=hashlib.sha256(notebook_path.read_bytes()).hexdigest())
    receipt=args.evidence/(args.mode+'_submission.json')
    with receipt.open('x',encoding='utf-8') as stream:json.dump(state,stream,indent=2)
    argv=[sys.executable,'-m','kaggle','kernels','push','-p',str(job),'-t',str(timeout)]
    if args.mode=='gpu':argv+=['--accelerator','NvidiaTeslaT4']
    try:
        process=subprocess.run(argv,capture_output=True,text=True,timeout=180)
        output=process.stdout+process.stderr
        state.update(status='accepted' if process.returncode==0 else 'acceptance_uncertain',returncode=process.returncode)
        match=re.search(r'Kernel version (\d+)',output)
        if match:state['version']=int(match.group(1))
        match=re.search(r'https://www\.kaggle\.com/code/[\w-]+/[\w-]+',output)
        if match:state['url']=match.group(0)
    except subprocess.TimeoutExpired:
        output='Acknowledgement timeout; preserve claim and do not retry automatically.'
        state.update(status='acceptance_uncertain',returncode=-1)
    receipt.write_text(json.dumps(state,indent=2),encoding='utf-8')
    (args.evidence/(args.mode+'_submission_cli.log')).write_text(output,encoding='utf-8')
    print(output[-1500:])


if __name__=='__main__':main()
