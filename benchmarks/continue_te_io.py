"""Bounded one-shot experiment continuation; no recurring task or model upload."""
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import time
from kaggle.api.kaggle_api_extended import KaggleApi

ROOT=Path(__file__).resolve().parents[1]
STATE=ROOT/'semantic_results/te_io_continuation.json'
STORAGE='chinokiki9/minimax-h3-te-storage-controls'
IDLE='chinokiki9/minimax-h3-idle-residency-pilot'


def main():
    state=dict(status='waiting_storage',started=time.time(),deadline_seconds=8*3600,
        storage=STORAGE,idle=IDLE,model_upload=False)
    def save():
        state['updated']=time.time();temporary=STATE.with_suffix('.partial')
        temporary.write_text(json.dumps(state,indent=2));temporary.replace(STATE)
    save();api=KaggleApi();api.authenticate();deadline=state['started']+state['deadline_seconds']
    def command(arguments,label,timeout=600):
        path=ROOT/'semantic_results'/(label+'.log')
        with path.open('w') as log:
            result=subprocess.run([sys.executable,*arguments],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=timeout)
        if result.returncode:raise RuntimeError(label+' failed; inspect bounded local log')
    def wait(kernel,phase):
        failures=0
        while time.time()<deadline:
            try:status=str(api.kernels_status(kernel).status);failures=0
            except Exception as error:
                failures+=1;state['transport_error']=type(error).__name__;save()
                if failures>=3:raise RuntimeError('Three consecutive status transport failures') from None
                time.sleep(60);continue
            state.update(status=phase,kaggle_status=status);save()
            if 'COMPLETE' in status:return
            if any(k in status for k in ('ERROR','CANCEL')):raise RuntimeError('Physical job terminal failure: '+kernel)
            time.sleep(60)
        raise TimeoutError('One-shot continuation deadline exceeded')
    try:
        wait(STORAGE,'waiting_storage')
        command(['tools/experiments/pipeline_audit_control.py','--slug',STORAGE,'--output','kaggle_te_io_storage_output',
            '--prefix','te_storage_results/','--download','--resume','--workers','2'], 'te_storage_collect')
        command(['tools/experiments/analyze_te_storage_controls.py','kaggle_te_io_storage_output/te_storage_results'],'te_storage_analysis')
        state.update(status='storage_analyzed',storage_summary='kaggle_te_io_storage_output/te_storage_results/storage_summary.json');save()
        # Launch only the reviewed, qualified frozen source. Never rebuild from
        # a checkout that another task could edit while this helper waits.
        sys.path.insert(0,str(ROOT))
        from tools.experiments.freeze_pipeline_job import read_bundle
        _,files=read_bundle(ROOT/'kaggle_idle_residency/pipeline.ipynb')
        binding=json.loads((ROOT/'kaggle_idle_residency/source_binding.json').read_text())
        if any(hashlib.sha256(files[n]).hexdigest()!=sha for n,sha in binding.items()):
            raise RuntimeError('Frozen idle notebook/source binding changed')
        pushed=False
        for attempt in range(10):
            if time.time()>=deadline:raise TimeoutError('No submission budget remains')
            result=subprocess.run([sys.executable,str(ROOT/'benchmarks/kaggle_tls12_runner.py'),'--cli',
                'kernels','push','-p',str(ROOT/'kaggle_idle_residency')],cwd=ROOT,capture_output=True,text=True,timeout=120)
            message=result.stdout+result.stderr
            version=re.search(r'Kernel version (\d+) successfully pushed',message)
            if version:
                state.update(status='idle_submitted',idle_version=int(version[1]),idle_submitted_at=time.time());save();pushed=True;break
            if 'Maximum batch GPU session count' not in message:raise RuntimeError('Idle submission rejected; no automatic retry of non-capacity errors')
            state.update(status='waiting_gpu_capacity',submission_attempt=attempt+1);save();time.sleep(60)
        if not pushed:raise RuntimeError('GPU capacity unavailable after bounded retries')
        time.sleep(60);wait(IDLE,'waiting_idle_pilot')
        output='kaggle_idle_residency_v2_output'
        command(['tools/experiments/pipeline_audit_control.py','--slug',IDLE,'--output',output,'--prefix','idle_residency_results/',
            '--download','--resume','--workers','2','--exclude','*nccl*'],'te_idle_v2_collect',timeout=1200)
        command(['tools/experiments/analyze_idle_pilot.py',output+'/idle_residency_results'],'te_idle_v2_analysis')
        state.update(status='complete',idle_summary=output+'/idle_residency_results/idle_validated_summary.json',
            remaining='Review physical controls/pilot before new file-cache or combined inference changes; smaller workload and gap matrix still required.');save()
    except BaseException as error:
        state.update(status='failed',error_type=type(error).__name__,
            error=re.sub(r'https?://\S+','[URL]',str(error))[:300]);save();raise


if __name__=='__main__':main()
