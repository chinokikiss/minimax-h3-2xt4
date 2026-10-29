import hashlib,io,json,os,pathlib,sys,tarfile,urllib.request
print('LIGHT_SOURCE_DOWNLOAD_START',flush=True)
PUB={'published': True, 'url': 'https://raw.githubusercontent.com/chinokikiss/minimax-h3-2xt4/b79b1bce1b8450a250e58bb8c8676630e58b4f59/runtime/kaggle_artifacts/3c0aa394f0d8da7c7e77dacc692cf1153da31d74945110a2075f017e35d7b30f.tar.gz', 'commit': 'b79b1bce1b8450a250e58bb8c8676630e58b4f59', 'sha256': '3c0aa394f0d8da7c7e77dacc692cf1153da31d74945110a2075f017e35d7b30f', 'bytes': 18672505, 'capsule_sha256': 'ca75f4b685f929060402a0860e0441e6d0eafff038b24495df810620ef0b4d52', 'zero_mounts': True, 'publication_verified': 'Git push and HTTP HEAD; full size/SHA checked during bootstrap'}
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
    '--distribution',str(DIST),'--source',str(SRC),'--output','/kaggle/working/light_setup','--mode','e2e']
if 'e2e'!='cpu':argv+=['--qualification','/kaggle/working/light_setup/cpu_qualification.json']
os.execve(sys.executable,argv,dict(os.environ))
