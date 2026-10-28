"""Immutable public DiT bundle staging for bake-only experiments."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import shutil
import time


def transient(error):
    """Retry transport/service failures only; never retry an identity failure."""
    response = getattr(error, 'response', None)
    code = getattr(response, 'status_code', None)
    if code is not None:
        return code in (408, 429, 500, 502, 503, 504)
    import requests
    import httpx
    return isinstance(error, (requests.ConnectionError, requests.Timeout,
                              httpx.TransportError, TimeoutError, ConnectionError))


def recover(operation, *, label, sleep=time.sleep):
    for attempt in range(3):
        try:
            return operation(attempt)
        except Exception as error:
            if attempt == 2 or not transient(error):
                raise
            # Bounded model-download recovery, never a Kaggle status request.
            print('ASSET_DOWNLOAD_RECOVER', label, attempt + 1,
                  type(error).__name__, getattr(getattr(error, 'response', None), 'status_code', None), flush=True)
            sleep((2, 5)[attempt])


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def download(row, directory):
    from huggingface_hub import hf_hub_download
    path=Path(recover(lambda attempt: hf_hub_download(row['repository'],row['file'],revision=row['revision'],
        token=False,local_dir=str(directory),etag_timeout=30,force_download=attempt>0),label=row['file']))
    if path.stat().st_size!=row['bytes'] or sha(path)!=row['sha256']:
        raise ValueError('Immutable checkpoint mismatch: '+row['file'])
    return path


def stage(source, scratch):
    source,scratch=Path(source),Path(scratch)
    scratch.mkdir(exist_ok=False)
    assets=json.loads((source/'bundle_assets.json').read_text())
    root=scratch/'base_bundle';root.mkdir()
    for path in (source/'bundle_provenance').glob('*.json'):shutil.copy2(path,root/path.name)
    assert sha(root/'bundle.json')==assets['publication']['bundle_sha256']
    objects=[(n,row) for n,row in assets['objects'].items() if n.startswith('dit.')]
    assert len(objects)==100
    needed=2*sum(row['bytes'] for _,row in objects)+assets['auxiliary']['dit']['bytes']+5*2**30
    if shutil.disk_usage(scratch).free<needed:raise OSError('Insufficient disk for immutable base plus persisted bake')
    def one(item):
        name,row=item
        return download(dict(repository=assets['publication']['repo'],revision=assets['publication']['commit'],file=name,**row),root)
    with ThreadPoolExecutor(max_workers=4) as pool:
        for i,job in enumerate(as_completed([pool.submit(one,item) for item in objects]),1):
            job.result()
            if i%10==0:print('BAKE_DOWNLOAD_VERIFIED',i,'/100',flush=True)
    import requests
    from checkpoint_ranges import assemble
    auxiliary=assets['auxiliary']['dit'];original=assets['source_identity']['dit']
    url='https://huggingface.co/'+assets['model_repository']+'/resolve/'+assets['model_revision']+'/'+original['remote_file']
    def read_once(start,end):
        with requests.get(url,params={'download':'true','offset':str(start)},headers={'Range':f'bytes={start}-{end-1}'},stream=True,timeout=(15,90)) as response:
            response.raise_for_status()
            if response.status_code!=206 or response.headers.get('Content-Range')!=f'bytes {start}-{end-1}/{original["bytes"]}':
                raise ValueError('Immutable auxiliary range mismatch')
            data=bytearray()
            for chunk in response.iter_content(4*2**20):
                data.extend(chunk)
                if len(data)>end-start:raise ValueError('Oversized auxiliary range')
            return data
    def read(start,end):
        return recover(lambda attempt: read_once(start,end),label=f'dit_aux:{start}:{end}')
    path=scratch/'dit_aux.safetensors';assemble(auxiliary,path,read)
    from patches.tp2_bundle import DESCRIPTOR
    descriptor=scratch/'dit_bundle.json'
    descriptor.write_text(json.dumps(dict(format=DESCRIPTOR,kind='dit',bundle_root=root.name,
        bundle_sha256=assets['publication']['bundle_sha256'],source={k:original[k] for k in ('bytes','sha256')},
        native_header=auxiliary['native_header'],native_header_sha256=auxiliary['native_header_sha256'],
        auxiliary_file=path.name,auxiliary_sha256=auxiliary['sha256']),indent=2))
    return descriptor
