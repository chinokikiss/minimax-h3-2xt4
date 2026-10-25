"""Kaggle download of immutable base shards and only missing auxiliary ranges."""
from concurrent.futures import ThreadPoolExecutor,as_completed
import hashlib
import json
from pathlib import Path
import shutil
import time
from checkpoint_ranges import assemble
from patches.tp2_bundle import DESCRIPTOR,sha


def prepare(source,scratch,output):
    import requests
    from huggingface_hub import hf_hub_download
    source=Path(source);scratch=Path(scratch);output=Path(output)
    plans=json.loads((source/'bundle_assets.json').read_text());publication=plans['publication']
    root=scratch/'bundle';root.mkdir();state=dict(status='preparing_base_bundle',base_only=True,objects=[],auxiliary={})
    def save():(output/'bundle_assets_receipt.json').write_text(json.dumps(state,indent=2))
    save();start=time.monotonic()
    total=sum(r['bytes'] for r in plans['objects'].values())+sum(r['bytes'] for r in plans['auxiliary'].values())+5*2**30
    if shutil.disk_usage(scratch).free<total:raise OSError('Insufficient scratch for immutable base bundle')
    for path in (source/'bundle_provenance').glob('*.json'):shutil.copyfile(path,root/path.name)
    def one(item):
        name,row=item;started=time.monotonic()
        path=Path(hf_hub_download(publication['repo'],name,revision=publication['commit'],token=False,local_dir=str(root)))
        if path.stat().st_size!=row['bytes'] or sha(path)!=row['sha256']:raise ValueError('Base shard transfer mismatch: '+name)
        return dict(file=name,bytes=row['bytes'],sha256=row['sha256'],seconds=time.monotonic()-started)
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures=[pool.submit(one,item) for item in plans['objects'].items()]
        for future in as_completed(futures):
            state['objects'].append(future.result());save()
            if len(state['objects'])%20==0:print('BASE_BUNDLE_READY',len(state['objects']),'/200',flush=True)
    paths={}
    for kind,plan in plans['auxiliary'].items():
        aux=scratch/(kind+'_aux.safetensors');identity=plans['source_identity'][kind]
        url='https://huggingface.co/'+plans['model_repository']+'/resolve/'+plans['model_revision']+'/'+identity['remote_file']
        def read(a,b):
            with requests.get(url,params={'download':'true','offset':str(a)},headers={'Range':f'bytes={a}-{b-1}'},stream=True,timeout=(15,90)) as response:
                if response.status_code!=206 or response.headers.get('Content-Range')!=f'bytes {a}-{b-1}/{identity["bytes"]}':
                    raise ValueError('Immutable auxiliary range mismatch')
                data=bytearray()
                for part in response.iter_content(4*2**20):
                    data.extend(part)
                    if len(data)>b-a:raise ValueError('Oversized auxiliary range')
                return data
        assemble(plan,aux,read)
        descriptor=scratch/(kind+'_bundle.json')
        descriptor.write_text(json.dumps(dict(format=DESCRIPTOR,kind=kind,bundle_root='bundle',
            bundle_sha256=publication['bundle_sha256'],source={k:identity[k] for k in ('bytes','sha256')},
            native_header=plan['native_header'],native_header_sha256=plan['native_header_sha256'],
            auxiliary_file=aux.name,auxiliary_sha256=plan['sha256']),indent=2))
        paths[kind]=str(descriptor);state['auxiliary'][kind]=dict(bytes=plan['bytes'],sha256=plan['sha256']);save()
    state.update(status='ready',seconds=time.monotonic()-start,persistent_baked_bytes=0,
        repo=publication['repo'],revision=publication['commit']);save()
    return paths,state
