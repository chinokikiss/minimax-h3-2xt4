"""Collect one final self-contained audit JSON, without status or log polling."""
import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys


def collect(evidence):
    evidence=Path(evidence);path=evidence/'gpu_submission.json'
    state=json.loads(path.read_text())
    earliest=dt.datetime.fromisoformat(state['submitted_utc'])+dt.timedelta(seconds=630)
    if dt.datetime.now(dt.timezone.utc)<earliest:raise RuntimeError('Fixed final collection window has not arrived')
    assert state['returncode']==0
    if state.get('final_collection_count'):return dict(status='already_claimed',provider_requests=0)
    claim=evidence/'gpu_submission.collection-claimed'
    with claim.open('x') as stream:stream.write(dt.datetime.now(dt.timezone.utc).isoformat())
    state['final_collection_count']=1;path.write_text(json.dumps(state,indent=2)+'\n')
    output=evidence/'gpu_outputs';output.mkdir()
    argv=[sys.executable,str(evidence/'final_cli.py'),'kernels','output',state['kernel'],
          '-p',str(output),'--file-pattern',r'^000_te_causal_precision\.json$','--page-size','200']
    process=subprocess.run(argv,capture_output=True,text=True,timeout=300)
    log=process.stdout+process.stderr
    log=re.sub(r'/kf/[^\s\x27\x22\)]+','/kf/<redacted>',log)
    log=re.sub(r'https?://[^\s]+','<redacted-url>',log)
    (evidence/'gpu_collection_cli.log').write_text(log,encoding='utf8')
    receipt=dict(status='final_output_unavailable',final_collection_operations=1,status_or_live_log_polling=False,
        returncode=process.returncode,completed_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        requested_file='000_te_causal_precision.json',self_contained_source_binding=True)
    raw=output/'000_te_causal_precision.json'
    if raw.exists():
        data=json.loads(raw.read_text())
        assert data['delivery_binding']['capsule_sha256']==state['source_binding']['publication']['capsule_sha256']
        assert data['supplemental_sources']==state['source_binding']['supplemental_sources']
        receipt.update(status='collected',audit_status=data['status'],raw_sha256=hashlib.sha256(raw.read_bytes()).hexdigest(),
            source_binding_verified=True,ranks=len(data['ranks']))
        if data['status']=='completed':
            assert len(data['ranks'])==2 and len({r['uuid'] for r in data['ranks']})==2
            assert all(r['status']=='completed' and 'T4' in r['gpu'] for r in data['ranks'])
    (evidence/'gpu_collection.json').write_text(json.dumps(receipt,indent=2)+'\n')
    return receipt


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--evidence',type=Path,required=True)
    print(json.dumps(collect(parser.parse_args().evidence),indent=2))
