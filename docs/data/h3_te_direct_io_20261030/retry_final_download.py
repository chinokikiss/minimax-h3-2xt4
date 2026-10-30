"""One retry of failed final-output transfer, never job status polling."""
import datetime as dt
import hashlib
import json
from pathlib import Path
import traceback

import requests
from kaggle.api.kaggle_api_extended import KaggleApi

root=Path(__file__).resolve().parent;out=root/'final_v1'
claim=root/'artifact_retry1.json'
state=dict(kind='final_artifact_transfer_retry',status='collecting',
    started_utc=dt.datetime.now(dt.timezone.utc).isoformat(),status_polling=False)
with claim.open('x') as stream:json.dump(state,stream,indent=2)
old=out/'download_manifest.json'
if old.exists():(root/'download_manifest_attempt1.json').write_bytes(old.read_bytes())
request=requests.sessions.Session.request
def bounded(self,*args,**kwargs):
    if kwargs.get('timeout') is None:kwargs['timeout']=(20,90)
    return request(self,*args,**kwargs)
requests.sessions.Session.request=bounded
try:
    api=KaggleApi();api.authenticate()
    files,_=api.kernels_output('pamcks/h3-te-direct-io-t4-20261030',str(out),quiet=True,page_size=100)
    state.update(status='downloaded' if files else 'final_outputs_unavailable',returned_files=len(files))
except Exception as error:
    (root/'artifact_retry1_private_error.txt').write_text(traceback.format_exc(),encoding='utf-8')
    state.update(status='transfer_failed',error_type=type(error).__name__)
finally:
    manifest={p.relative_to(out).as_posix():dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
        for p in out.rglob('*') if p.is_file() and p.name!='download_manifest.json'}
    old.write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    state.update(completed_utc=dt.datetime.now(dt.timezone.utc).isoformat(),local_files=len(manifest))
    claim.write_text(json.dumps(state,indent=2),encoding='utf-8')
print(json.dumps(state),flush=True)
