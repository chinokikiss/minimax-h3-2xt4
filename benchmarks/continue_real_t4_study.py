"""One-shot continuation: wait for the authorized upload, run T4 replay, collect.

This is a bounded local execution helper, not a recurring scheduled task.
"""
import json
from pathlib import Path
import re
import subprocess
import sys
import time
import hashlib
from kaggle.api.kaggle_api_extended import KaggleApi

ROOT=Path(__file__).resolve().parents[1]
OUT=Path('D:/attention_eval/t4_redo_20261013')
KAGGLE=Path(sys.executable).parent/'Scripts/kaggle.exe'
DATASET='chinokiki9/h3-native-attention-tokyo-20261013'
KERNEL='chinokiki9/h3-real-t4-attention-root-causes'
state={'status':'waiting_upload','started':time.time(),
       'dataset':DATASET,'kernel':KERNEL,'deadline_seconds':48*3600,
       'counter_blocker':'Confirmed ERR_NVGPUCTRPERM; no counter-based performance conclusions',
       'scope':'Upload then operator replay/accuracy/Nsight Systems. Full-pipeline and targeted-improvement study still requires follow-up implementation.'}
STATE=OUT/'continuation_state.json'


def save():
    state['updated']=time.time()
    temp=STATE.with_suffix('.tmp');temp.write_text(json.dumps(state,indent=2));temp.replace(STATE)


def command(args,name,timeout):
    with (OUT/(name+'.log')).open('w') as log:
        proc=subprocess.run(args,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=timeout)
    if proc.returncode:
        raise RuntimeError(name+' exited '+str(proc.returncode)+'; see log')


def main():
    save();api=KaggleApi();api.authenticate()
    deadline=state['started']+state['deadline_seconds']
    uploader=None;upload_log=None;upload_attempts=0
    while time.time()<deadline:
        upload_path=OUT/'upload_state.json'
        if upload_path.exists():
            upload=json.loads(upload_path.read_text())
            state['upload_files_complete']=sum(r.get('status')=='complete' for r in upload.get('files',{}).values())
            state['upload_total_files']=upload.get('total_files')
            state['upload_acknowledged_bytes']=sum(r.get('bytes',0) if r.get('status')=='complete'
                else r.get('server_acknowledged_bytes',0) for r in upload.get('files',{}).values())
            if upload['status']=='dataset_processing':break
        if uploader is None or uploader.poll() is not None:
            if upload_attempts>=6:raise RuntimeError('Upload retry budget exhausted; resumable metadata retained')
            if upload_log:upload_log.close()
            upload_attempts+=1
            upload_log=(OUT/f'continuation_upload_{upload_attempts}.log').open('w')
            uploader=subprocess.Popen([sys.executable,'-u',str(ROOT/'benchmarks/upload_real_attention_dataset.py')],
                cwd=ROOT,stdout=upload_log,stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NO_WINDOW if sys.platform=='win32' else 0)
            state.update(upload_process_pid=uploader.pid,upload_process_attempt=upload_attempts)
        save();time.sleep(30)
    else:
        if uploader and uploader.poll() is None:uploader.terminate()
        raise TimeoutError('Upload did not finish within bounded continuation deadline')
    if uploader and uploader.poll() is None:uploader.wait(timeout=120)
    if upload_log:upload_log.close()
    state['status']='waiting_dataset_processing';save()
    for _ in range(120):
        try:
            status=api.dataset_status(DATASET)
            state['dataset_status']=str(status.status);save()
            if 'ready' in str(status.status).lower():break
        except Exception as error:
            state['dataset_status_error']=re.sub(r'https?://\S+','[URL]',str(error));save()
        time.sleep(30)
    else:raise TimeoutError('Dataset processing was not ready after one hour')
    # Sources were frozen by build_real_t4_attention_job.py before continuation.
    state['status']='submitting_t4_job';save()
    state['notebook_sha256']=hashlib.sha256((ROOT/'kaggle_real_t4_attention/experiment.ipynb').read_bytes()).hexdigest()
    save()
    command([str(KAGGLE),'kernels','push','-p',str(ROOT/'kaggle_real_t4_attention'),
             '--accelerator','NvidiaTeslaT4','-t','21600'],'real_t4_push',180)
    if 'successfully pushed' not in (OUT/'real_t4_push.log').read_text():
        raise RuntimeError('CLI did not confirm job submission; see real_t4_push.log')
    state['status']='waiting_t4_job';save()
    job_deadline=time.time()+8*3600
    while time.time()<job_deadline:
        response=api.kernels_status(KERNEL)
        status=str(response.status);state['kernel_status']=status;save()
        if status.endswith('.COMPLETE'):break
        if any(status.endswith('.'+x) for x in ['ERROR','CANCEL_ACKNOWLEDGED']):
            raise RuntimeError('Provider job '+status+': '+str(response.failure_message))
        time.sleep(45)
    else:raise TimeoutError('T4 execution exceeded bounded wait')
    state['status']='collecting_t4_artifacts';save()
    collected=OUT/'real_t4_results'
    for attempt in range(3):
        try:
            command([str(KAGGLE),'kernels','output',KERNEL,'-p',str(collected),'--page-size','200'],
                    'real_t4_download_'+str(attempt),3600)
            page=0;log=OUT/('real_t4_download_'+str(attempt)+'.log')
            while match:=re.search(r'Next page token:\s*(\S+)',log.read_text()):
                page+=1
                if page>8:raise RuntimeError('Unexpectedly many artifact pages')
                name='real_t4_download_'+str(attempt)+'_page_'+str(page)
                command([str(KAGGLE),'kernels','output',KERNEL,'-p',str(collected),'--page-size','200',
                         '--page-token',match.group(1)],name,3600)
                log=OUT/(name+'.log')
            break
        except Exception:
            if attempt==2:raise
            time.sleep(30)
    proof=collected/'h3_real_t4_results/status.json'
    if not proof.exists():raise RuntimeError('Output collection incomplete; operator status artifact missing')
    state['remote_study_status']=json.loads(proof.read_text())
    state['status']='operator_artifacts_collected_full_study_pending';save()
    print('COLLECTED',proof,flush=True)


if __name__=='__main__':
    try:main()
    except Exception as error:
        state.update(status='blocked',error=re.sub(r'https?://\S+','[URL]',str(error)));save()
        print('CONTINUATION_BLOCKED',state['error'],flush=True)
        raise
