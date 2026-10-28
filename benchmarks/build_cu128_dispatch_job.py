"""One CPU contract gate or one bounded T4 probe; no polling or full H3 code."""
import argparse
import ast
import datetime as dt
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
from t4_notebook_watchdog import supervise_worker


def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=('cpu','gpu'),required=True)
    p.add_argument('--publication',type=Path,required=True);p.add_argument('--evidence',type=Path,required=True)
    p.add_argument('--qualification',type=Path);p.add_argument('--operator-qualification',type=Path)
    p.add_argument('--submit',action='store_true');args=p.parse_args()
    args.evidence.mkdir(parents=True,exist_ok=True);job=args.evidence/(args.mode+'_job')
    argv=[sys.executable,str(Path(__file__).with_name('build_kaggle_light_job.py')),
        '--mode','cpu' if args.mode=='cpu' else 'e2e','--publication',str(args.publication),'--output',str(job),'--setup-budget','600']
    if args.mode=='cpu':argv+=['--review-suite']
    else:
        if not args.qualification or not args.operator_qualification:p.error('Matching native/core and operator CPU receipts required')
        q=json.loads(args.operator_qualification.read_text());pub=json.loads(args.publication.read_text())
        assert q['status']=='cu128_dispatch_cpu_ready' and q['capsule_sha256']==pub['capsule_sha256']
        assert json.loads(args.qualification.read_text())['review_suite']['passed']
        argv+=['--qualification',str(args.qualification)]
    subprocess.run(argv,check=True)
    notebook_path=job/'experiment.ipynb';notebook=json.loads(notebook_path.read_text());code=''.join(notebook['cells'][0]['source'])
    if args.mode=='gpu':
        marker="assert receipt['status']=='ready_for_device_gate'\n";assert code.count(marker)==1
        code=code[:code.index(marker)+len(marker)]
    execution="""import json,os,pathlib,sys
env=dict(os.environ);env.update(json.loads(pathlib.Path('/kaggle/working/light_setup/execution_env.json').read_text()))
source=pathlib.Path('/kaggle/temp/h3_light_source')
argv=[sys.executable,'-u',str(source/'benchmarks/run_cu128_dispatch.py'),'--source',str(source),
 '--output','/kaggle/working/cu128_dispatch']
EXTRA
os.execve(sys.executable,argv,env)
""".replace('EXTRA',"argv+=['--cpu-preflight']" if args.mode=='cpu' else '')
    code+=supervise_worker(execution,budget=120 if args.mode=='cpu' else 1200,output='/kaggle/working/cu128_dispatch_supervision')
    code+="assert json.loads(pathlib.Path('/kaggle/working/cu128_dispatch/result.json').read_text())['status'] in ('cu128_dispatch_cpu_ready','complete','operator_blockers_found')\n"
    ast.parse(code)
    assert 'launch_kaggle_light.py' not in code and 'run_h3_l1_current_batch.py' not in code
    notebook['cells'][0]['source']=code.splitlines(keepends=True);notebook_path.write_text(json.dumps(notebook))
    meta=json.loads((job/'kernel-metadata.json').read_text());label='cpu' if args.mode=='cpu' else 't4'
    meta.update(id='pamcks/h3-cu128-dispatch-'+label+'-20261028',title='H3 cu128 Dispatch '+label.upper()+' 20261028')
    assert meta['dataset_sources']==meta['kernel_sources']==[]
    (job/'kernel-metadata.json').write_text(json.dumps(meta,indent=2))
    binding=json.loads((job/'source_binding.json').read_text());binding.update(experiment='Explicit cu128 dispatch candidates',
        full_pipeline=False,model_downloads=0,operator_qualification_sha256=hashlib.sha256(args.operator_qualification.read_bytes()).hexdigest() if args.operator_qualification else None)
    (job/'source_binding.json').write_text(json.dumps(binding,indent=2))
    if not args.submit:return
    timeout=900 if args.mode=='cpu' else 2100
    state=dict(kernel=meta['id'],status='submitting',submitted_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        source_binding=binding,final_collection_count=0,earliest_final_collection_seconds=timeout,run_timeout_seconds=timeout,
        result_relative='cu128_dispatch/result.json',notebook_sha256=hashlib.sha256(notebook_path.read_bytes()).hexdigest())
    receipt=args.evidence/(args.mode+'_submission.json')
    with receipt.open('x') as stream:json.dump(state,stream,indent=2)
    command=[sys.executable,'-m','kaggle','kernels','push','-p',str(job),'-t',str(timeout)]
    if args.mode=='gpu':command+=['--accelerator','NvidiaTeslaT4']
    try:
        run=subprocess.run(command,capture_output=True,text=True,timeout=180);output=run.stdout+run.stderr
        state.update(returncode=run.returncode,status='accepted' if run.returncode==0 else 'acceptance_uncertain')
        version=re.search(r'Kernel version (\d+)',output)
        if version:state['version']=int(version.group(1))
        url=re.search(r'https://www\.kaggle\.com/code/[\w-]+/[\w-]+',output)
        if url:state.update(url=url.group(0),kernel=url.group(0).split('/code/',1)[1])
    except subprocess.TimeoutExpired:
        output='Submission acknowledgement timed out; no automatic retry.';state.update(status='acceptance_uncertain',returncode=-1)
    (args.evidence/(args.mode+'_submission_cli.log')).write_text(output);receipt.write_text(json.dumps(state,indent=2));print(output[-1500:])


if __name__=='__main__':main()
