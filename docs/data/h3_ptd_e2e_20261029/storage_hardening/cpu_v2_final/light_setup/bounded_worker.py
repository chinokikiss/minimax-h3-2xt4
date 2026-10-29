import hashlib,io,json,os,pathlib,sys,tarfile,urllib.request
print('LIGHT_SOURCE_DOWNLOAD_START',flush=True)
PUB={'file': '1c961b1bd9113b0d6f81145535e3d239220d6140e70c5ef339a4755edbb72aa6.tar.gz', 'sha256': '1c961b1bd9113b0d6f81145535e3d239220d6140e70c5ef339a4755edbb72aa6', 'bytes': 18669059, 'capsule_sha256': 'f289791007e0b475381a2af3a3a6cf7ba54a96f78d32a117f73288a5ae4e4ccd', 'source_files': 882, 'core_pins': 78, 'kaggle_mounts': 0, 'published': True, 'commit': '4e550f7523faa64ddc453e55b1da9db3f6404dbe', 'url': 'https://raw.githubusercontent.com/chinokikiss/minimax-h3-2xt4/4e550f7523faa64ddc453e55b1da9db3f6404dbe/runtime/kaggle_artifacts/1c961b1bd9113b0d6f81145535e3d239220d6140e70c5ef339a4755edbb72aa6.tar.gz', 'zero_mounts': True, 'publication_verified': 'Acknowledged Git push and immutable source HTTP HEAD; full bytes/SHA checked during bootstrap'}
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
