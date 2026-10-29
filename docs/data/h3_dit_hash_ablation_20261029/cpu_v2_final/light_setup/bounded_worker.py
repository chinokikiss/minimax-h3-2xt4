import hashlib,io,json,os,pathlib,sys,tarfile,urllib.request
print('LIGHT_SOURCE_DOWNLOAD_START',flush=True)
PUB={'file': 'eadb59c0f8e6d7a2af5030a2d0aa60254443f15fcaba955112892bd18ea3089c.tar.gz', 'sha256': 'eadb59c0f8e6d7a2af5030a2d0aa60254443f15fcaba955112892bd18ea3089c', 'bytes': 18696038, 'capsule_sha256': 'aded869fdade8e3d55aaca455d6a17345e2dc3431514a16e625548ed7ff11316', 'source_files': 894, 'core_pins': 78, 'kaggle_mounts': 0, 'published': True, 'commit': 'e1966aca5323d555cffc891de8bfff7812295987', 'url': 'https://raw.githubusercontent.com/chinokikiss/minimax-h3-2xt4/e1966aca5323d555cffc891de8bfff7812295987/runtime/kaggle_artifacts/eadb59c0f8e6d7a2af5030a2d0aa60254443f15fcaba955112892bd18ea3089c.tar.gz'}
DIST=pathlib.Path('/kaggle/temp/h3_light_distribution');DIST.mkdir(parents=True,exist_ok=True)
SRC=pathlib.Path('/kaggle/temp/h3_light_source');SRC.mkdir(parents=True,exist_ok=True)
with urllib.request.urlopen(PUB['url'],timeout=90) as response:
    raw=response.read(PUB['bytes']+1)
assert len(raw)==PUB['bytes'] and hashlib.sha256(raw).hexdigest()==PUB['sha256'],'Frozen distribution checksum mismatch'
with tarfile.open(fileobj=io.BytesIO(raw),mode='r:gz') as archive:archive.extractall(DIST,filter='data')
payload=(DIST/'capsule.tar.gz.bin').read_bytes()
assert hashlib.sha256(payload).hexdigest()==PUB['capsule_sha256']
with tarfile.open(fileobj=io.BytesIO(payload),mode='r:gz') as archive:archive.extractall(SRC,filter='data')
pathlib.Path('/kaggle/working/light_setup/delivery_binding.json').write_text(json.dumps(PUB,indent=2))
print('LIGHT_SOURCE_READY',flush=True)
argv=[sys.executable,'-u',str(SRC/'benchmarks/prepare_kaggle_direct_pip.py'),
    '--distribution',str(DIST),'--source',str(SRC),'--output','/kaggle/working/light_setup','--mode','cpu']
if 'cpu'!='cpu':argv+=['--qualification','/kaggle/working/light_setup/cpu_qualification.json']
argv+=['--review-suite']
os.execve(sys.executable,argv,dict(os.environ))
