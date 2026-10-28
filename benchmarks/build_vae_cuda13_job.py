"""One-shot CPU qualification or GPU ablation; never execute a full H3 pipeline."""
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
    p.add_argument('--qualification',type=Path);p.add_argument('--cuda13-qualification',type=Path)
    p.add_argument('--submit',action='store_true');args=p.parse_args()
    args.evidence.mkdir(parents=True,exist_ok=True);job=args.evidence/(args.mode+'_job')
    command=[sys.executable,str(Path(__file__).with_name('build_kaggle_light_job.py')),
             '--mode','cpu' if args.mode=='cpu' else 'e2e','--publication',str(args.publication),'--output',str(job),'--setup-budget','600']
    q=None
    if args.mode=='cpu':command+=['--review-suite']
    else:
        if not args.qualification or not args.cuda13_qualification:p.error('Both CPU receipts required')
        command+=['--qualification',str(args.qualification)]
        q=json.loads(args.cuda13_qualification.read_text())
        publication=json.loads(args.publication.read_text())
        assert q['status']=='cuda13_cpu_ready' and q['capsule_sha256']==publication['capsule_sha256']
        assert q['probe']['input_sha256']=='08a10be3efdda85ea6415c832eb2654101da5d5fabe54b6e34002071ee2dde25'
    subprocess.run(command,check=True)
    path=job/'experiment.ipynb';notebook=json.loads(path.read_text());code=''.join(notebook['cells'][0]['source'])
    if args.mode=='gpu':
        marker="assert receipt['status']=='ready_for_device_gate'\n"
        assert code.count(marker)==1
        code=code[:code.index(marker)+len(marker)]
    if q:code+="pathlib.Path('/kaggle/working/cuda13_cpu_qualification.json').write_text("+repr(json.dumps(q))+')\n'
    body="""import json,os,pathlib,sys
source=pathlib.Path('/kaggle/temp/h3_light_source')
argv=[sys.executable,'-u',str(source/'benchmarks/prepare_vae_cuda13.py'),'--source',str(source),
 '--output','/kaggle/working/cuda13_setup']
EXTRA
os.execve(sys.executable,argv,dict(os.environ))
""".replace('EXTRA',"argv+=['--cpu']" if args.mode=='cpu' else "argv+=['--qualification','/kaggle/working/cuda13_cpu_qualification.json']")
    code+=supervise_worker(body,budget=1200,output='/kaggle/working/cuda13_supervision')
    code+="""guard=json.loads(pathlib.Path('/kaggle/working/cuda13_supervision/supervision.json').read_text())
if guard['status']!='worker_exited' or guard['exit_code']!=0:raise RuntimeError('CUDA13 CPU/API qualification failed; inspect artifacts')
"""
    if args.mode=='gpu':
        execution="""import json,os,pathlib,sys
env=dict(os.environ);env.update(json.loads(pathlib.Path('/kaggle/working/cuda13_setup/execution_env.json').read_text()))
source=pathlib.Path('/kaggle/temp/h3_light_source')
argv=[sys.executable,'-u',str(source/'benchmarks/run_vae_cuda13_batch.py'),'--source',str(source),
 '--output','/kaggle/working/vae_cuda13']
os.execve(sys.executable,argv,env)
"""
        code+=supervise_worker(execution,budget=2100,output='/kaggle/working/vae_cuda13_supervision')
        code+="assert json.loads(pathlib.Path('/kaggle/working/vae_cuda13/result.json').read_text())['status']=='complete'\n"
    ast.parse(code);notebook['cells'][0]['source']=code.splitlines(keepends=True);path.write_text(json.dumps(notebook))
    metadata=json.loads((job/'kernel-metadata.json').read_text())
    metadata.update(id='pamcks/h3-vae-cuda13-'+('cpu' if args.mode=='cpu' else 'ablation-t4')+'-20261028',
        title='H3 VAE CUDA13 '+('CPU' if args.mode=='cpu' else 'Ablation T4')+' 20261028',
        dataset_sources=['pamcks/h3-vae-ab-latents-20261028'])
    (job/'kernel-metadata.json').write_text(json.dumps(metadata,indent=2))
    binding=json.loads((job/'source_binding.json').read_text());binding.update(
        experiment='VAE CUDA13 only',full_pipeline=False,cuda13_qualification_sha256=hashlib.sha256(args.cuda13_qualification.read_bytes()).hexdigest() if q else None,
        kaggle_dataset_sources=metadata['dataset_sources'],execution_budget=2100 if args.mode=='gpu' else 0,platform_setup_budget=1200)
    (job/'source_binding.json').write_text(json.dumps(binding,indent=2))
    if not args.submit:return
    timeout=2400 if args.mode=='cpu' else 4200
    state=dict(kernel=metadata['id'],status='submitting',submitted_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        source_binding=binding,final_collection_count=0,earliest_final_collection_seconds=timeout,run_timeout_seconds=timeout,
        result_relative='cuda13_setup/setup_receipt.json' if args.mode=='cpu' else 'vae_cuda13/result.json',
        notebook_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    receipt=args.evidence/(args.mode+'_submission.json')
    with receipt.open('x') as stream:json.dump(state,stream,indent=2)
    argv=[sys.executable,'-m','kaggle','kernels','push','-p',str(job),'-t',str(timeout)]
    if args.mode=='gpu':argv+=['--accelerator','NvidiaTeslaT4']
    try:
        result=subprocess.run(argv,capture_output=True,text=True,timeout=180)
        output=result.stdout+result.stderr;state.update(returncode=result.returncode,status='accepted' if result.returncode==0 else 'acceptance_uncertain')
        match=re.search(r'Kernel version (\d+)',output)
        if match:state['version']=int(match.group(1))
        url=re.search(r'https://www\.kaggle\.com/code/[\w-]+/[\w-]+',output)
        if url:
            state['url']=url.group(0)
            state['kernel']=url.group(0).split('/code/',1)[1]
    except subprocess.TimeoutExpired:
        output='Submission acknowledgement timed out; do not retry automatically.';state.update(status='acceptance_uncertain',returncode=-1)
    (args.evidence/(args.mode+'_submission_cli.log')).write_text(output)
    receipt.write_text(json.dumps(state,indent=2));print(output[-1500:],flush=True)


if __name__=='__main__':main()
