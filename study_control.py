"""Read status/logs or retrieve only study evidence, without model caches."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import gzip
import json
from pathlib import Path
import threading
import time
from kaggle.api.kaggle_api_extended import KaggleApi

SLUG='chinokiki9/minimax-h3-te-dit-parallel-study'
LOCAL=threading.local()


def fetch(name,output):
    path=output/name
    if path.exists():
        data=path.read_bytes()
        if name.endswith('.json'):json.loads(data)
        elif name.endswith('.gz'):gzip.decompress(data)
        return {'file':name,'bytes':len(data),'cached':True}
    if not hasattr(LOCAL,'session'):
        api=KaggleApi();api.authenticate();client=api.build_kaggle_client();client._http_client._init_session()
        LOCAL.client=client;LOCAL.session=client._http_client._session
    for attempt in range(4):
        response=LOCAL.session.get('https://www.kaggle.com/api/v1/kernels/output/download/'+SLUG+'/'+name,timeout=(8,30))
        if response.status_code!=429:break
        time.sleep(2**attempt)
    if response.status_code!=200:raise RuntimeError(f'{name}: HTTP {response.status_code}')
    data=response.content
    if name.endswith('.json'):json.loads(data)
    elif name.endswith('.gz'):gzip.decompress(data)
    path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
    return {'file':name,'bytes':len(data)}


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['status','download']);p.add_argument('--output',type=Path,default=Path('kaggle_study_output/baseline'));a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True)
    api=KaggleApi();api.authenticate()
    status=api.kernels_status(SLUG);print('STATUS',status.status,status.failure_message,flush=True)
    if a.action=='status':
        rows=[]
        def read():
            try:
                for e in api.kernels_logs_stream(SLUG):rows.append(e)
            except Exception:pass
        t=threading.Thread(target=read,daemon=True);t.start();t.join(8)
        (a.output/'live_logs.json').write_text(json.dumps(rows))
        seen=set();text=[]
        for e in rows:
            k=(str(e.get('time')),e.get('stream_name'),e.get('data'))
            if k not in seen:text.append(e.get('data',''));seen.add(k)
        joined=''.join(text)
        meaningful=[line for line in joined.splitlines() if line.startswith(('VERIFIED','START','FINISHED','MEASURED','FULL','THROUGHPUT','LOADED','Traceback','RuntimeError','TimeoutError','ValueError','Error','torch.OutOfMemoryError'))]
        print('\n'.join(meaningful[-40:]) if meaningful else 'No benchmark result lines yet; setup/download stage.')
        return
    print(fetch('study_results/manifest.json',a.output),flush=True)
    manifest=json.loads((a.output/'study_results/manifest.json').read_text())
    names=sorted(set(manifest['files']+['study_source.tar.gz']))
    with ThreadPoolExecutor(max_workers=1) as pool:
        for row in pool.map(lambda name:fetch(name,a.output),names):print(row,flush=True)


if __name__=='__main__':main()
