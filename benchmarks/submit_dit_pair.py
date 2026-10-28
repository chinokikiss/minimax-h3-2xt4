"""Submit one CPU gate or one two-call T4 experiment, never poll or retry."""
import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys


def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=('cpu','gpu'),required=True)
    p.add_argument('--publication',type=Path,required=True);p.add_argument('--evidence',type=Path,required=True)
    p.add_argument('--qualification',type=Path);p.add_argument('--pair-qualification',type=Path);p.add_argument('--submit',action='store_true');args=p.parse_args()
    args.evidence.mkdir(parents=True,exist_ok=True);job=args.evidence/(args.mode+'_job')
    argv=[sys.executable,str(Path(__file__).with_name('build_kaggle_light_job.py')),'--mode','cpu' if args.mode=='cpu' else 'e2e',
          '--dit-pair','--publication',str(args.publication),'--output',str(job),'--setup-budget','600']
    if args.mode=='cpu':argv+=['--review-suite']
    else:
        if not args.qualification or not args.pair_qualification:p.error('Both matching CPU receipts required')
        argv+=['--qualification',str(args.qualification),'--pair-qualification',str(args.pair_qualification),'--execution-budget','4800']
    subprocess.run(argv,check=True)
    if not args.submit:return
    metadata=json.loads((job/'kernel-metadata.json').read_text());binding=json.loads((job/'source_binding.json').read_text())
    receipt=args.evidence/(args.mode+'_submission.json')
    state=dict(kernel=metadata['id'],submitted_utc=dt.datetime.now(dt.timezone.utc).isoformat(),status='submitting',returncode=None,
        source_binding=binding,final_collection_count=0,earliest_final_collection_seconds=900 if args.mode=='cpu' else 5400,
        run_timeout_seconds=900 if args.mode=='cpu' else 5400,experiment_kind='dit_pair_'+args.mode,
        result_relative='dit_pair_preflight/batch_results.json' if args.mode=='cpu' else 'dit_pair/batch_results.json',
        notebook_sha256=hashlib.sha256((job/'experiment.ipynb').read_bytes()).hexdigest())
    with receipt.open('x') as stream:json.dump(state,stream,indent=2)
    argv=[sys.executable,'-m','kaggle','kernels','push','-p',str(job),'-t',str(state['run_timeout_seconds'])]
    if args.mode=='gpu':argv+=['--accelerator','NvidiaTeslaT4']
    try:
        result=subprocess.run(argv,capture_output=True,text=True,timeout=180);output=result.stdout+result.stderr
        state.update(returncode=result.returncode,status='accepted' if result.returncode==0 else 'acceptance_uncertain')
        version=re.search(r'Kernel version (\d+)',output)
        if version:state['version']=int(version.group(1))
        url=re.search(r'https://www\.kaggle\.com/code/([\w-]+/[\w-]+)',output)
        if url:state.update(url=url.group(0),kernel=url.group(1))
    except subprocess.TimeoutExpired:
        output='Submission acknowledgement timed out; no automatic retry.';state.update(returncode=-1,status='acceptance_uncertain')
    (args.evidence/(args.mode+'_submission_cli.log')).write_text(output,encoding='utf-8');receipt.write_text(json.dumps(state,indent=2))
    print(output[-1500:],flush=True)


if __name__=='__main__':main()
