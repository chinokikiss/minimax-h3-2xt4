import hashlib,io,json,os,pathlib,sys,tarfile,urllib.request
print('LIGHT_SOURCE_DOWNLOAD_START',flush=True)
PUB={'file': '6e42d303e0bd8820f090a1ef54a8fb14e284f3db44c2392010ed5c9cd0c5e655.tar.gz', 'sha256': '6e42d303e0bd8820f090a1ef54a8fb14e284f3db44c2392010ed5c9cd0c5e655', 'bytes': 18696119, 'capsule_sha256': '1246b03ba71c886884d39b0cc91d5c18ce0f255efeb7ed63239ee6327b155e09', 'source_files': 894, 'core_pins': 78, 'kaggle_mounts': 0, 'published': True, 'commit': 'e1093f1c3c9aeba12d9d0f42d5b47815d8d7f268', 'url': 'https://raw.githubusercontent.com/chinokikiss/minimax-h3-2xt4/e1093f1c3c9aeba12d9d0f42d5b47815d8d7f268/runtime/kaggle_artifacts/6e42d303e0bd8820f090a1ef54a8fb14e284f3db44c2392010ed5c9cd0c5e655.tar.gz'}
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
