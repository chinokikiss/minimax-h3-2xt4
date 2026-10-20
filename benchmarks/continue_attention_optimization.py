"""Bounded one-shot P0 collection -> Group32 hardware execution -> collection."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import time
from kaggle.api.kaggle_api_extended import KaggleApi

ROOT=Path(__file__).resolve().parents[1];OUT=Path('D:/attention_eval/quality_constrained_20261015')
RUNNER=ROOT/'benchmarks/kaggle_tls12_runner.py';GROUP=ROOT/'kaggle_h3-attention-grouped-study'
STATE=OUT/'execution_state.json';state={'status':'waiting_p0','started':time.time(),'deadline_seconds':10800,
    'group_notebook_sha256':hashlib.sha256((GROUP/'experiment.ipynb').read_bytes()).hexdigest()}


def save():
    state['updated']=time.time();tmp=STATE.with_suffix('.tmp');tmp.write_text(json.dumps(state,indent=2));tmp.replace(STATE)


def run(args,name,timeout=1800):
    with (OUT/(name+'.log')).open('w') as log:p=subprocess.run([sys.executable,str(RUNNER),*map(str,args)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=timeout)
    return p.returncode,(OUT/(name+'.log')).read_text(errors='replace')


def wait(api,kernel):
    errors=0
    while time.time()<state['started']+state['deadline_seconds']:
        try:s=str(api.kernels_status(kernel).status);errors=0
        except Exception as error:
            errors+=1;state.update(network_error=type(error).__name__,consecutive_errors=errors);save()
            if errors>=10:raise RuntimeError('Kaggle status unavailable after bounded retries') from None
            time.sleep(45);continue
        state['provider_status']=s;save()
        if s.endswith('.COMPLETE') or s.endswith('.ERROR') or s.endswith('.CANCEL_ACKNOWLEDGED'):return s
        time.sleep(45)
    raise RuntimeError('Bounded experiment continuation expired')


def collect(kernel,version,folder,prefix):
    code,_=run([ROOT/'benchmarks/download_kaggle_evidence.py',kernel,'--version',version,'--output',OUT/folder,
        '--pattern',prefix+r'/.*\.(json|log|csv|sqlite|nsys-rep|prof|ptx|cu)$'],'collect_'+folder)
    if code:raise RuntimeError('Evidence collection failed; inspect '+folder+' collector log')


def report():
    with (OUT/'report_update.log').open('w') as log:subprocess.run([sys.executable,str(ROOT/'benchmarks/report_quality_constrained.py')],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=120)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--group-version');parser.add_argument('--full-version');args=parser.parse_args()
    save();api=KaggleApi();api.authenticate()
    if args.full_version:
        version=args.full_version;state.update(status='waiting_native_full',full_version=version,deadline_seconds=25200);save()
        s=wait(api,'chinokiki9/h3-attention-native-full-shapes')
        state['status']='collecting_native_full';save();collect('chinokiki9/h3-attention-native-full-shapes',version,'native_full_v'+version,'native_full');report()
        state.update(status='native_full_complete_evidence_collected' if s.endswith('.COMPLETE') else 'native_full_failed_evidence_collected',quality_acceptance='informal V4 review only; Group32 generation pending');save();return
    if args.group_version:
        version=args.group_version;state.update(status='waiting_group_hardware',group_version=version);save()
        s=wait(api,'chinokiki9/h3-attention-grouped-study')
        state['status']='collecting_group';save();collect('chinokiki9/h3-attention-grouped-study',version,'group_v'+version,'grouped');report()
        state.update(status='group_complete_evidence_collected' if s.endswith('.COMPLETE') else 'group_failed_evidence_collected',quality_acceptance='informal V4 review only; Group32 generation pending');save();return
    p0='chinokiki9/h3-attention-p0-matched-matrix';s=wait(api,p0)
    state['status']='collecting_p0';save();collect(p0,'1','p0_v1','p0');report()
    if not s.endswith('.COMPLETE'):state['status']='p0_failed_evidence_collected';save();return
    while time.time()<state['started']+state['deadline_seconds']:
        if hashlib.sha256((GROUP/'experiment.ipynb').read_bytes()).hexdigest()!=state['group_notebook_sha256']:raise RuntimeError('Frozen Group32 source changed; do not auto-submit')
        code,body=run(['--cli','kernels','push','-p',GROUP,'--accelerator','NvidiaTeslaT4','-t','14400'],'group_push',180)
        match=re.search(r'Kernel version (\d+) successfully pushed',body)
        if match:version=match.group(1);break
        if 'Maximum batch GPU session count' not in body:raise RuntimeError('Group hardware submission failed; inspect group_push.log')
        state['status']='waiting_gpu_capacity';save();time.sleep(45)
    else:raise RuntimeError('GPU capacity unavailable within bounded continuation')
    state.update(status='waiting_group_hardware',group_version=version);save()
    s=wait(api,'chinokiki9/h3-attention-grouped-study')
    state['status']='collecting_group';save();collect('chinokiki9/h3-attention-grouped-study',version,'group_v'+version,'grouped');report()
    state.update(status='group_complete_evidence_collected' if s.endswith('.COMPLETE') else 'group_failed_evidence_collected',quality_acceptance='pending blinded ratings and Group32 generation');save()


if __name__=='__main__':
    try:main()
    except Exception as error:
        state.update(status='continuation_error',error=re.sub(r'https?://\S+','[URL]',str(error)));save()
