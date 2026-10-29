"""Submit the already inspected notebook once, without rebuilding its directory."""
import datetime as dt
import hashlib
import json
from pathlib import Path
import re
import runpy
import subprocess
import sys

root=Path(__file__).resolve().parent
runpy.run_path(str(root/'verify_submission.py'),run_name='__main__')
evidence=root/'kaggle';job=evidence/'gpu_job'
binding=json.loads((job/'source_binding.json').read_text(encoding='utf-8'))
meta=json.loads((job/'kernel-metadata.json').read_text(encoding='utf-8'))
state=dict(kernel=meta['id'],status='submitting',submitted_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
    source_binding=binding,run_timeout_seconds=8100,earliest_final_collection_seconds=8100,
    final_collection_count=0,result_relative='turbo_085mp_e2e/batch_results.json',
    notebook_sha256=hashlib.sha256((job/'experiment.ipynb').read_bytes()).hexdigest())
receipt=evidence/'gpu_submission.json'
with receipt.open('x',encoding='utf-8') as f:json.dump(state,f,indent=2)
try:
    process=subprocess.run([sys.executable,'-m','kaggle','kernels','push','-p',str(job),'-t','8100',
        '--accelerator','NvidiaTeslaT4'],capture_output=True,text=True,timeout=180)
    output=process.stdout+process.stderr
    state.update(status='accepted' if process.returncode==0 else 'acceptance_uncertain',returncode=process.returncode)
    match=re.search(r'Kernel version (\d+)',output)
    if match:state['version']=int(match.group(1))
    match=re.search(r'https://www\.kaggle\.com/code/([\w-]+/[\w-]+)',output)
    if match:state.update(requested_kernel=state['kernel'],kernel=match.group(1),url=match.group(0))
except subprocess.TimeoutExpired:
    output='Acknowledgement timeout; preserve claim, do not retry automatically.'
    state.update(status='acceptance_uncertain',returncode=-1)
receipt.write_text(json.dumps(state,indent=2),encoding='utf-8')
(evidence/'gpu_submission_cli.log').write_text(output,encoding='utf-8')
print(output[-1500:])
