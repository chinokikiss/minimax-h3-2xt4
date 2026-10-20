"""Resume validation/collection after an already-running artifact download."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import psutil

ROOT=Path(__file__).resolve().parents[1]
OUT=Path('D:/attention_eval/quality_constrained_20261015')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--wait-process',type=int,required=True);args=ap.parse_args()
    folder=OUT/'quality_challenge_v1';manifest=folder/'download_manifest.json'
    deadline=time.monotonic()+1800
    while not manifest.exists() and time.monotonic()<deadline:
        try:
            process=psutil.Process(args.wait_process)
            if 'download_kaggle_evidence.py' not in ' '.join(process.cmdline()):break
        except psutil.NoSuchProcess:break
        time.sleep(15)
    if not manifest.exists():raise RuntimeError('Artifact collection did not finish; saved GPU outputs must not be regenerated')
    data=json.loads(manifest.read_text())
    if data['kernel']!='chinokiki9/h3-attention-quality-challenge' or data['requested_version']!='1':
        raise RuntimeError('Unexpected collection version')
    for row in data['files']:
        with (folder/row['file']).open('rb') as stream:
            if hashlib.file_digest(stream,'sha256').hexdigest()!=row['sha256']:raise RuntimeError('Collected artifact hash mismatch')
    path=OUT/'next_execution_state.json';state=json.loads(path.read_text())
    state.update(status='active',updated=time.time())
    state['quality'].update(status='evidence_collected',output=str(folder),provider_status='KernelWorkerStatus.COMPLETE')
    state['quality'].pop('network_error',None)
    state['sol'].update(version='1',status='submitted',provider_status='KernelWorkerStatus.RUNNING')
    state.pop('error',None)
    path.write_text(json.dumps(state,indent=2))
    print('Verified',len(data['files']),'quality artifacts; resuming CPU assessment and Sol collection',flush=True)
    subprocess.run([sys.executable,str(ROOT/'benchmarks/kaggle_tls12_runner.py'),
        str(ROOT/'benchmarks/continue_quality_next.py')],cwd=ROOT,check=True)


if __name__=='__main__':main()
