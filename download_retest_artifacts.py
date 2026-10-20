"""Retrieve only small benchmark evidence files; never start a notebook."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import threading
from kaggle.api.kaggle_api_extended import KaggleApi

ROOT = Path(__file__).resolve().parent / 'kaggle_retest_output'
LOCAL = threading.local()
RUNS = {
    'cu130': ('minimax-h3-cuda13-retest', ['audio','video','video_tiling','te']),
    'fp32_pp': ('minimax-h3-fp32-and-pp-retest', ['dit','te_stack','dit_stack']),
}

def download(task):
    folder,slug,name = task
    path = ROOT/folder/name
    if path.exists():
        try:
            if name.endswith('.json'):
                json.loads(path.read_bytes())
                return True
        except (ValueError,UnicodeError):
            pass
    if not hasattr(LOCAL,'session'):
        api = KaggleApi()
        api.authenticate()
        client = api.build_kaggle_client()
        client._http_client._init_session()
        LOCAL.client = client
        LOCAL.session = client._http_client._session
    url = f'https://www.kaggle.com/api/v1/kernels/output/download/chinokiki9/{slug}/{name}'
    for attempt in range(4):
        try:
            response = LOCAL.session.get(url,timeout=(8,15))
            response.raise_for_status()
            if name.endswith('.json'):
                json.loads(response.content)
            elif not response.content.startswith(b'\x1f\x8b'):
                raise ValueError('Invalid source archive')
            path.parent.mkdir(parents=True,exist_ok=True)
            path.write_bytes(response.content)
            return True
        except Exception as exc:
            if attempt == 3:
                print('FAILED',folder,name,type(exc).__name__,flush=True)
    return False

if __name__ == '__main__':
    tasks = []
    for folder,(slug,suites) in RUNS.items():
        tasks.extend((folder,slug,name) for name in ['retest_manifest.json','retest_source.tar.gz'])
        for round_index in range(3):
            for suite in suites:
                names = ['results.json','environment.json']
                if suite not in ['audio','video_tiling']:
                    names.append('input_manifest.json')
                tasks.extend((folder,slug,f'retest_results/round_{round_index}/{suite}/{name}') for name in names)
    main_slug = RUNS['cu130'][0]
    tasks.extend(('cu130',main_slug,f'retest_results/round_0/video_profile/{name}.json')
                 for name in ['accuracy','native_profile','tp_rank0_profile','tp_rank1_profile','sp_rank0_profile','sp_rank1_profile'])
    tasks.extend(('cu130',main_slug,f'retest_results/round_0/{suite}/results.json')
                 for suite in ['te_stack','dit_stack'])
    with ThreadPoolExecutor(max_workers=4) as pool:
        success = list(pool.map(download,tasks))
    print('Evidence files available:',sum(success),'/',len(tasks),flush=True)
