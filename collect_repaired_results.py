"""Retrieve small evidence files only; never fetch cached model outputs."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import threading
from kaggle.api.kaggle_api_extended import KaggleApi

ROOT = Path(__file__).resolve().parent / 'kaggle_repaired_output'
LOCAL = threading.local()
SLUG = 'chinokiki9/minimax-h3-repaired-dual-t4'


def fetch(name):
    if not hasattr(LOCAL, 'session'):
        api = KaggleApi(); api.authenticate()
        client = api.build_kaggle_client()
        client._http_client._init_session()
        LOCAL.client, LOCAL.session = client, client._http_client._session
    response = LOCAL.session.get('https://www.kaggle.com/api/v1/kernels/output/download/' + SLUG + '/' + name,
                                 timeout=(8, 20))
    if response.status_code != 200:
        return {'file': name, 'downloaded': False, 'status': response.status_code}
    data = response.content
    if name.endswith('.json'):
        json.loads(data)
    elif not data.startswith(b'\x1f\x8b'):
        raise ValueError('Source archive response is not a gzip archive')
    path = ROOT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return {'file': name, 'downloaded': True, 'bytes': len(data)}


if __name__ == '__main__':
    names = ['repaired_manifest.json', 'repaired_source.tar.gz']
    for suite in ['te', 'dit', 'video', 'slabs']:
        names += [f'repaired_results/{suite}/{name}' for name in ['results.json', 'environment.json']]
    with ThreadPoolExecutor(max_workers=3) as pool:
        rows = list(pool.map(fetch, names))
    ROOT.mkdir(exist_ok=True)
    (ROOT / 'download_manifest.json').write_text(json.dumps(rows, indent=2))
    for row in rows:
        print(row, flush=True)
