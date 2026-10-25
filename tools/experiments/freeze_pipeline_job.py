"""Shared private source freezing and remote Native-reference restoration."""
import base64
import hashlib
import io
import json
from pathlib import Path
import re
import tarfile
import time


def read_bundle(path):
    notebook=json.loads(path.read_text())
    payload=re.search(r"b64decode\('([^']+)'\)",''.join(notebook['cells'][0]['source'])).group(1)
    with tarfile.open(fileobj=io.BytesIO(base64.b64decode(payload)),mode='r:gz') as archive:
        files={m.name:archive.extractfile(m).read() for m in archive.getmembers()}
    return notebook,files


def refresh_references(files):
    from kaggle.api.kaggle_api_extended import KaggleApi,ApiListKernelSessionOutputRequest
    import requests
    api=KaggleApi();api.authenticate();request=ApiListKernelSessionOutputRequest()
    request.user_name='chinokiki9';request.kernel_slug='minimax-h3-causal-pipeline-audit'
    old=json.loads(files['resume_reference_ticket.json']);objects={}
    with api.build_kaggle_client() as client:
        while True:
            page=client.kernels.kernels_api_client.list_kernel_session_output(request)
            for item in page.files or []:
                name=Path(item.file_name).name
                if not item.file_name.startswith('causal_audit_results/native/native_reference/') or name not in old['objects']:continue
                try:
                    with requests.get(item.url,headers={'Range':'bytes=0-0'},stream=True,timeout=(10,45)) as response:
                        if response.status_code not in (200,206):raise RuntimeError('Native reference HTTP '+str(response.status_code))
                        count=int(response.headers['Content-Range'].split('/')[-1]) if response.status_code==206 else int(response.headers['Content-Length'])
                except requests.RequestException:raise RuntimeError('Native reference transport failed; capability omitted') from None
                objects[name]=dict(url=item.url,bytes=count,shape=old['objects'][name]['shape'])
            if not page.next_page_token:break
            request.page_token=page.next_page_token
    if set(objects)!=set(old['objects']):raise RuntimeError('Incomplete Native reference set')
    files['resume_reference_ticket.json']=json.dumps(dict(created_at=time.time(),objects=objects)).encode()


def write_job(root,folder,notebook,files,slug,title,entry):
    for name,data in files.items():
        if name.endswith('.py'):compile(data,name,'exec')
    buffer=io.BytesIO()
    with tarfile.open(fileobj=buffer,mode='w:gz') as archive:
        for name,data in files.items():
            member=tarfile.TarInfo(name);member.size=len(data);archive.addfile(member,io.BytesIO(data))
    source="""import base64,io,pathlib,runpy,tarfile
root=pathlib.Path('/kaggle/working/causal_audit_source');root.mkdir(exist_ok=True)
with tarfile.open(fileobj=io.BytesIO(base64.b64decode(PAYLOAD)),mode='r:gz') as archive:archive.extractall(root,filter='data')
runpy.run_path(str(root/ENTRY),run_name='__main__')
""".replace('PAYLOAD',repr(base64.b64encode(buffer.getvalue()).decode())).replace('ENTRY',repr(entry))
    if len(source)>1_000_000:raise RuntimeError('Notebook source limit exceeded')
    notebook['cells'][0]['source']=source.splitlines(keepends=True)
    out=root/folder;out.mkdir(exist_ok=True)
    (out/'pipeline.ipynb').write_text(json.dumps(notebook))
    metadata=json.loads((root/'kaggle_residency_p01/kernel-metadata.json').read_text())
    metadata.update(id='chinokiki9/'+slug,title=title,code_file='pipeline.ipynb')
    (out/'kernel-metadata.json').write_text(json.dumps(metadata,indent=2))
    (out/'source_binding.json').write_text(json.dumps({n:hashlib.sha256(d).hexdigest() for n,d in files.items()
        if n!='resume_reference_ticket.json'},indent=2))
    print('FROZEN',slug,'source bytes',len(source),'private capabilities omitted',flush=True)
