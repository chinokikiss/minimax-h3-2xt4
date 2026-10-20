"""One-shot bounded collection and one qualified recovery, not a scheduler."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import time
import requests
import shutil
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelRequest

ROOT=Path(__file__).resolve().parents[1]
OUT=Path('D:/attention_eval/quality_first_20261014')
KERNEL='chinokiki9/h3-quality-first-generation'
FOLDER=ROOT/'kaggle_h3-quality-first-generation'
NOTEBOOK=FOLDER/'experiment.ipynb'
KAGGLE=Path(sys.executable).parent/'Scripts/kaggle.exe'
state={'status':'waiting_generation_v3','started':time.time(),'deadline_seconds':43200,
       'queued_recovery_notebook_sha256':hashlib.sha256(NOTEBOOK.read_bytes()).hexdigest()}
STATE=OUT/'execution_state.json'


def save():
    state['updated']=time.time();tmp=STATE.with_suffix('.tmp');tmp.write_text(json.dumps(state,indent=2));tmp.replace(STATE)


def collect(version):
    destination=OUT/f'generation_v{version}'
    args=[sys.executable,str(ROOT/'benchmarks/download_kaggle_evidence.py'),KERNEL,'--version',str(version),
        '--output',str(destination),'--pattern',r'generation_screen/.*\.(json|log|mp4|wav|txt|csv)$']
    for attempt in range(3):
        with (OUT/f'collect_generation_v{version}_{attempt}.log').open('w') as log:
            result=subprocess.run(args,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=1800)
        if result.returncode==0 and (destination/'generation_screen/status.json').exists():return destination/'generation_screen'
        time.sleep(15)
    raise RuntimeError('Could not collect version-bound generation evidence')


def wait(api,version,deadline):
    metadata=ApiGetKernelRequest();metadata.user_name,metadata.kernel_slug=KERNEL.split('/')
    errors=0
    while time.time()<deadline:
        try:
            with api.build_kaggle_client() as client:
                current=client.kernels.kernels_api_client.get_kernel(metadata)
            response=api.kernels_status(KERNEL)
            errors=0
        except requests.RequestException as error:
            errors+=1;state['transient_poll_error']=type(error).__name__;state['consecutive_poll_errors']=errors;save()
            if errors>=10:raise RuntimeError('Provider polling unavailable after10bounded retries') from None
            time.sleep(45);continue
        if current.metadata.current_version_number!=version:
            raise RuntimeError('Notebook version changed outside this continuation; automatic submission stopped')
        status=str(response.status)
        state['provider_status']=status;state['watched_version']=version;save()
        if any(status.endswith('.'+x) for x in ['COMPLETE','ERROR','CANCEL_ACKNOWLEDGED']):return
        time.sleep(45)
    raise RuntimeError('Bounded wait expired; remote job state must be checked')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--version',type=int,default=3)
    args=parser.parse_args()
    state['status']=f'waiting_generation_v{args.version}'
    api=KaggleApi();api.authenticate();save();deadline=state['started']+state['deadline_seconds']
    wait(api,args.version,deadline);directory=collect(args.version);report=json.loads((directory/'status.json').read_text())
    state[f'generation_v{args.version}_experiment_status']=report['status'];save()
    known_export_bug=any('torch.stack' in log.read_text(errors='replace') and
        any(x in log.read_text(errors='replace') for x in ['same device','cuda:0 and cuda:1'])
        for log in directory.glob('*_seed*.log'))
    if report['status']=='blocked' and not known_export_bug:
        state.update(status='generation_blocker_recorded',blocker=report);save();return
    if hashlib.sha256(NOTEBOOK.read_bytes()).hexdigest()!=state['queued_recovery_notebook_sha256']:
        raise RuntimeError('Queued source changed; automatic recovery stopped')
    if known_export_bug and args.version==3:
        # The patched density export and validated fused preparation are frozen.
        # A single rerun is authorized by the existing experiment request.
        state['status']='submitting_export_recovery';save()
        with (OUT/'generation_recovery_push.log').open('w') as log:
            proc=subprocess.run([str(KAGGLE),'kernels','push','-p',str(FOLDER),'--accelerator','NvidiaTeslaT4','-t','21600'],
                                cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=180)
        body=(OUT/'generation_recovery_push.log').read_text()
        match=re.search(r'Kernel version (\d+) successfully pushed',body)
        if proc.returncode or not match:raise RuntimeError('Provider did not confirm recovery submission; inspect generation_recovery_push.log')
        version=int(match.group(1));state['recovery_version']=version;state['status']='waiting_export_recovery';save()
        wait(api,version,deadline);directory=collect(version);report=json.loads((directory/'status.json').read_text())
    if report['status']=='screen_generated':
        state['status']='scoring_cpu_quality_metrics';save()
        reference=ROOT/'bench_assets/tokyo_neon.jpg'
        shutil.copy2(reference,directory/'reference.jpg')
        metric_failures=[]
        for clip in directory.glob('*_seed*/native_video.mp4'):
            config=json.loads((directory/(clip.parent.name+'.json')).read_text())
            command=[sys.executable,str(ROOT/'benchmarks/quality_first_video_metrics.py'),'--video',str(clip),
                '--reference',str(reference),'--prompt',config['prompt'],'--output',str(clip.parent/'embedding_metrics.json'),
                '--device','cpu','--embeddings']
            with (OUT/(clip.parent.name+'_embedding_metrics.log')).open('w') as log:
                try:
                    completed=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=900)
                    if completed.returncode:metric_failures.append(clip.parent.name)
                except subprocess.TimeoutExpired:metric_failures.append(clip.parent.name)
        state['metric_failures']=metric_failures
        with (OUT/'blinded_review_build.log').open('w') as log:
            completed=subprocess.run([sys.executable,str(ROOT/'benchmarks/quality_first_review.py'),'--runs',str(directory),
                '--output',str(OUT/'blinded_review')],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=120)
        state['review_build_exit_code']=completed.returncode
    state.update(status='generation_artifacts_collected' if report['status']=='screen_generated' else 'generation_blocker_recorded',
                 experiment_status=report,quality_acceptance='pending metrics and blinded ratings; no numerical-only acceptance')
    save()


if __name__=='__main__':
    try:main()
    except Exception as error:
        state.update(status='continuation_error',error=re.sub(r'https?://\S+','[URL]',str(error)));save()
