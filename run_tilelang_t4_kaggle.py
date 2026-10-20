"""Kaggle CLI submission/collection for this isolated private T4 experiment."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parent
KERNEL='chinokiki9/tilelang-t4-int8-sparse-attention'


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--resume',action='store_true',help='collect the already submitted run')
    parser.add_argument('--output',type=Path,default=ROOT/'tilelang_t4_results'/'attention_latest')
    args=parser.parse_args()
    cli=shutil.which('kaggle') or str(Path(sys.executable).parent/'Scripts'/'kaggle.exe')
    def call(*items,timeout=120):
        result=subprocess.run([cli,*items],capture_output=True,text=True,timeout=timeout)
        print(result.stdout.strip(),flush=True)
        if result.stderr:print(result.stderr.strip(),flush=True)
        if result.returncode:raise RuntimeError('Kaggle CLI failed')
        return result.stdout
    if not args.resume:
        subprocess.run([sys.executable,str(ROOT/'build_tilelang_t4_job.py')],check=True)
        call('kernels','push','-p',str(ROOT/'kaggle_tilelang_t4_attention'),
             '--accelerator','NvidiaTeslaT4','-t','3000')
    deadline=time.monotonic()+3600
    while True:
        status=call('kernels','status',KERNEL,timeout=30)
        if any(word in status for word in ['COMPLETE','ERROR','CANCEL']):break
        if time.monotonic()>deadline:raise TimeoutError('Collection deadline; remote job was not cancelled')
        time.sleep(30)
    args.output.mkdir(parents=True,exist_ok=True)
    call('kernels','output',KERNEL,'-p',str(args.output),timeout=300)
    result_path=args.output/'tilelang_t4_results'/'results.json'
    if result_path.exists():
        result=json.loads(result_path.read_text())
        print(json.dumps({'completed':result['completed'],'passed':result['passed'],
                          'records':len(result['cases']),
                          'failed':[(r['case']['name'],r['mode'],r['status']) for r in result['cases'] if not r.get('passed')]},indent=2),flush=True)
    else:raise RuntimeError('No experiment results; inspect the downloaded kernel log')


if __name__=='__main__':main()
