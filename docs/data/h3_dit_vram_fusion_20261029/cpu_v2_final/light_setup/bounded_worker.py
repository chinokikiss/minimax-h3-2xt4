import hashlib,io,json,os,pathlib,sys,tarfile,urllib.request
print('LIGHT_SOURCE_DOWNLOAD_START',flush=True)
PUB={'file': '3b24746a0a09a9f87fbf8c39ae0f587fbd2bbabb8a8169be499feac814c21557.tar.gz', 'sha256': '3b24746a0a09a9f87fbf8c39ae0f587fbd2bbabb8a8169be499feac814c21557', 'bytes': 18680552, 'capsule_sha256': '7286ec2f971f1ec7b096db161eb534a7e2d26ec1a20cbdb517f87bc2fcbb6fc3', 'source_files': 887, 'core_pins': 78, 'kaggle_mounts': 0, 'published': True, 'commit': '089a371ca64be5fe726361112b893a092786ff58', 'url': 'https://raw.githubusercontent.com/chinokikiss/minimax-h3-2xt4/089a371ca64be5fe726361112b893a092786ff58/runtime/kaggle_artifacts/3b24746a0a09a9f87fbf8c39ae0f587fbd2bbabb8a8169be499feac814c21557.tar.gz', 'zero_mounts': True, 'publication_verified': 'Acknowledged Git push and immutable source HTTP HEAD; bootstrap checks full size and hash'}
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
