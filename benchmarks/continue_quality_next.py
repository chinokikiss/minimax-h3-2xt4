"""Bounded collection and queued T4 Sol ablation; never cancel other jobs."""
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import time
from kaggle.api.kaggle_api_extended import KaggleApi

ROOT=Path(__file__).resolve().parents[1];OUT=Path('D:/attention_eval/quality_constrained_20261015')
RUNNER=ROOT/'benchmarks/kaggle_tls12_runner.py'
STATE=OUT/'next_execution_state.json'
SOL=ROOT/'kaggle_h3-attention-sol-fused-study'
state={'started':time.time(),'deadline_seconds':25200,'status':'active',
       'quality':{'kernel':'chinokiki9/h3-attention-quality-challenge','version':'1','status':'submitted'},
       'sol':{'kernel':'chinokiki9/h3-attention-sol-fused-study','status':'waiting_gpu_capacity',
              'notebook_sha256':hashlib.sha256((SOL/'experiment.ipynb').read_bytes()).hexdigest()}}
if STATE.exists():
    previous=json.loads(STATE.read_text())
    if previous['quality']['kernel']==state['quality']['kernel'] and previous['quality']['version']==state['quality']['version']:
        state=previous


def save():
    state['updated']=time.time();tmp=STATE.with_suffix('.tmp');tmp.write_text(json.dumps(state,indent=2));tmp.replace(STATE)


def command(args,name,timeout=1800):
    path=OUT/(name+'.log')
    with path.open('w') as log:
        p=subprocess.run([sys.executable,str(RUNNER),*map(str,args)],stdout=log,stderr=subprocess.STDOUT,cwd=ROOT,timeout=timeout)
    return p.returncode,path.read_text(errors='replace')


def collect(key):
    job=state[key];prefix='generation_screen' if key=='quality' else 'sol_fused'
    folder=OUT/(('quality_challenge_v' if key=='quality' else 'sol_fused_v')+job['version'])
    pattern=prefix+r'/.*\.(json|log|csv|mp4|wav|sqlite|nsys-rep|prof|txt)$'
    if key=='quality':pattern+='|generation_screen/challenge_inputs_seed[0-9]+/(conditioning|initial_noise|sigmas)\.pt$'
    job['status']='collecting';save()
    for attempt in range(3):
        code,_=command([ROOT/'benchmarks/download_kaggle_evidence.py',job['kernel'],'--version',job['version'],
                        '--output',folder,'--pattern',pattern],'collect_'+key+'_v'+job['version'])
        if code==0:
            job.update(status='evidence_collected',output=str(folder));save();return
        job.update(collection_attempts=attempt+1);save();time.sleep(15)
    job['status']='collection_failed';save()


def main():
    api=KaggleApi();api.authenticate();save();next_submit=0;last_status={};network_errors=0;last_report=None
    while time.time()<state['started']+state['deadline_seconds']:
        for key in ['quality','sol']:
            job=state[key]
            if 'version' not in job or job['status'] in ['evidence_collected','collection_failed']:continue
            try:
                provider=str(api.kernels_status(job['kernel']).status);network_errors=0
            except Exception as error:
                network_errors+=1;job['network_error']=type(error).__name__;save()
                if network_errors>=20:raise RuntimeError('Status unavailable after bounded retries') from None
                continue
            job['provider_status']=provider
            if last_status.get(key)!=provider:
                print(key,provider,flush=True);last_status[key]=provider
            if any(provider.endswith('.'+name) for name in ['COMPLETE','ERROR','CANCEL_ACKNOWLEDGED']):collect(key)
            save()
        if 'version' not in state['sol'] and time.time()>=next_submit:
            if hashlib.sha256((SOL/'experiment.ipynb').read_bytes()).hexdigest()!=state['sol']['notebook_sha256']:
                raise RuntimeError('Frozen Sol notebook changed; do not auto-submit')
            code,body=command(['--cli','kernels','push','-p',SOL,'--accelerator','NvidiaTeslaT4','-t','10800'],'sol_queued_push',180)
            match=re.search(r'Kernel version (\d+) successfully pushed',body)
            if match:
                state['sol'].update(version=match.group(1),status='submitted');print('SOL_SUBMITTED',match.group(1),flush=True)
            elif 'Maximum batch GPU session count' not in body:
                raise RuntimeError('Sol submission failed; inspect sol_queued_push.log')
            next_submit=time.time()+300;save()
        quality=state['quality']
        if quality['status']=='evidence_collected' and not quality.get('local_postprocessing'):
            directory=Path(quality['output'])/'generation_screen'
            if (directory/'status.json').exists():
                code,_=command([ROOT/'benchmarks/prepare_next_quality_artifacts.py','--runs',directory],'prepare_next_quality',300)
                quality['local_postprocessing']='prepared' if code==0 else 'failed';save()
                if code==0:
                    code,_=command([ROOT/'benchmarks/score_quality_first_artifacts.py','--runs',directory,
                        '--embedding-revisions',directory/'embedding_revisions.json'],'score_next_quality_cpu',2400)
                    quality['CPU_proxy_scoring']='finished' if code==0 else 'failed';save()
            else:quality['local_postprocessing']='no_generation_status';save()
        signature=json.dumps({k:{f:state[k].get(f) for f in ['status','provider_status','local_postprocessing','CPU_proxy_scoring']}
            for k in ['quality','sol']},sort_keys=True)
        if signature!=last_report:
            subprocess.run([sys.executable,str(ROOT/'benchmarks/report_quality_constrained.py')],cwd=ROOT,
                stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,timeout=120)
            last_report=signature
        if all(state[k]['status'] in ['evidence_collected','collection_failed'] for k in ['quality','sol']):
            state['status']='finished';save();return
        time.sleep(45)
    state['status']='bounded_wait_expired';save()


if __name__=='__main__':
    try:main()
    except Exception as error:
        state.update(status='continuation_error',error=re.sub(r'https?://\S+','[URL]',str(error)));save();raise
