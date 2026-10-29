import hashlib,io,json,os,pathlib,sys,tarfile,urllib.request
print('LIGHT_SOURCE_DOWNLOAD_START',flush=True)
PUB={'file': 'd5f6ac2f87036eea6954fa9f08461cba0ed5250275dc0e5999e36683b29b5b9a.tar.gz', 'sha256': 'd5f6ac2f87036eea6954fa9f08461cba0ed5250275dc0e5999e36683b29b5b9a', 'bytes': 18680225, 'capsule_sha256': 'c62c0f545eca173cfd4d08bc84fef38041f2428a299a7936cafd9828e9ee6ca9', 'source_files': 887, 'core_pins': 78, 'kaggle_mounts': 0, 'published': True, 'commit': 'fc3128f20408759e33b3807d07663789e955d81f', 'url': 'https://raw.githubusercontent.com/chinokikiss/minimax-h3-2xt4/fc3128f20408759e33b3807d07663789e955d81f/runtime/kaggle_artifacts/d5f6ac2f87036eea6954fa9f08461cba0ed5250275dc0e5999e36683b29b5b9a.tar.gz', 'zero_mounts': True, 'publication_verified': 'Git push acknowledged and immutable HTTP HEAD; bootstrap checks full bytes and hash'}
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
