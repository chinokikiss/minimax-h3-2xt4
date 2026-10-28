"""One frozen CPU qualification, then one bounded operator/bake GPU batch."""
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
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=('cpu','gpu'),required=True)
    p.add_argument('--publication',type=Path,required=True);p.add_argument('--evidence',type=Path,required=True)
    p.add_argument('--qualification',type=Path);p.add_argument('--operator-qualification',type=Path)
    p.add_argument('--submit',action='store_true');args=p.parse_args();args.evidence.mkdir(parents=True,exist_ok=True)
    job=args.evidence/(args.mode+'_job');pub=json.loads(args.publication.read_text())
    command=[sys.executable,str(Path(__file__).with_name('build_kaggle_light_job.py')),'--mode',
        'cpu' if args.mode=='cpu' else 'e2e','--publication',str(args.publication),'--output',str(job),'--setup-budget','600']
    if args.mode=='cpu':command+=['--review-suite']
    else:
        if not args.qualification or not args.operator_qualification:p.error('Matching CPU setup and optimization receipts required')
        gate=json.loads(args.operator_qualification.read_text());setup=json.loads(args.qualification.read_text())
        assert gate['status']=='optimization_cpu_ready' and gate['capsule_sha256']==pub['capsule_sha256']
        assert setup['capsule_sha256']==pub['capsule_sha256'] and setup['review_suite']['passed']
        command+=['--qualification',str(args.qualification)]
    subprocess.run(command,check=True)
    path=job/'experiment.ipynb';notebook=json.loads(path.read_text());code=''.join(notebook['cells'][0]['source'])
    if args.mode=='gpu':
        marker="assert receipt['status']=='ready_for_device_gate'\n";assert code.count(marker)==1
        code=code[:code.index(marker)+len(marker)]
    execution="""import json,os,pathlib,sys
env=dict(os.environ);env.update(json.loads(pathlib.Path('/kaggle/working/light_setup/execution_env.json').read_text()))
source=pathlib.Path('/kaggle/temp/h3_light_source')
argv=[sys.executable,'-u',str(source/'benchmarks/run_h3_optimization_batch.py'),'--source',str(source),
 '--output','/kaggle/working/operator_batch']
EXTRA
os.execve(sys.executable,argv,env)
""".replace('EXTRA',"argv+=['--cpu-preflight']" if args.mode=='cpu' else '')
    code+=supervise_worker(execution,budget=180 if args.mode=='cpu' else 2700,output='/kaggle/working/operator_batch_supervision')
    code+="assert json.loads(pathlib.Path('/kaggle/working/operator_batch/result.json').read_text())['status'] in ('optimization_cpu_ready','complete','operator_blockers_found')\n"
    assert_operator_only(code);notebook['cells'][0]['source']=code.splitlines(keepends=True)
    path.write_text(json.dumps(notebook));label='cpu' if args.mode=='cpu' else 't4'
    meta=json.loads((job/'kernel-metadata.json').read_text())
    meta.update(id=f'pamcks/h3-operators-aot-bake-{label}-20261028',title=f'H3 Operators AOT Bake {label.upper()} 20261028')
    assert meta['dataset_sources']==meta['kernel_sources']==[]
    (job/'kernel-metadata.json').write_text(json.dumps(meta,indent=2))
    binding=json.loads((job/'source_binding.json').read_text());binding.update(full_pipeline=False,
        experiment='Scoped defaults, corrected conv, all-selected custom AOT, FP32 RMS-RoPE, bake AB/BA',
        private_inputs=False,operator_qualification_sha256=hashlib.sha256(args.operator_qualification.read_bytes()).hexdigest() if args.operator_qualification else None)
    (job/'source_binding.json').write_text(json.dumps(binding,indent=2))
    if not args.submit:return
    timeout=900 if args.mode=='cpu' else 3300
    state=dict(kernel=meta['id'],status='submitting',submitted_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        source_binding=binding,final_collection_count=0,earliest_final_collection_seconds=timeout,run_timeout_seconds=timeout,
        result_relative='operator_batch/result.json',notebook_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    receipt=args.evidence/(args.mode+'_submission.json')
    with receipt.open('x') as stream:json.dump(state,stream,indent=2)
    argv=[sys.executable,'-m','kaggle','kernels','push','-p',str(job),'-t',str(timeout)]
    if args.mode=='gpu':argv+=['--accelerator','NvidiaTeslaT4']
    try:
        proc=subprocess.run(argv,capture_output=True,text=True,timeout=180);output=proc.stdout+proc.stderr
        state.update(status='accepted' if proc.returncode==0 else 'acceptance_uncertain',returncode=proc.returncode)
        version=re.search(r'Kernel version (\d+)',output)
        if version:state['version']=int(version.group(1))
        url=re.search(r'https://www\.kaggle\.com/code/[\w-]+/[\w-]+',output)
        if url:state.update(url=url.group(0),kernel=url.group(0).split('/code/',1)[1])
    except subprocess.TimeoutExpired:
        output='Acknowledgement timeout; preserve submission claim and do not retry automatically.'
        state.update(status='acceptance_uncertain',returncode=-1)
    receipt.write_text(json.dumps(state,indent=2));(args.evidence/(args.mode+'_submission_cli.log')).write_text(output)
    print(output[-1500:])


if __name__=='__main__':main()
