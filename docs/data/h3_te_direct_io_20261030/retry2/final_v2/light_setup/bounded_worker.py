import hashlib,io,json,os,pathlib,sys,tarfile,urllib.request
print('LIGHT_SOURCE_DOWNLOAD_START',flush=True)
PUB={'file': '8d64c9a531ac7eb89e8156472ad901297d8d4df058bb00e73337ae71228634d6.tar.gz', 'sha256': '8d64c9a531ac7eb89e8156472ad901297d8d4df058bb00e73337ae71228634d6', 'bytes': 18702276, 'capsule_sha256': '21abe7dd91f284eed9626bb018c6abf66adf1c066b2f76662d72ce53d46d41a1', 'source_files': 900, 'core_pins': 78, 'kaggle_mounts': 0, 'published': True, 'commit': 'd7c7528515d2a7961de5e6f961232d81d21cdc8b', 'url': 'https://raw.githubusercontent.com/chinokikiss/minimax-h3-2xt4/d7c7528515d2a7961de5e6f961232d81d21cdc8b/runtime/kaggle_artifacts/8d64c9a531ac7eb89e8156472ad901297d8d4df058bb00e73337ae71228634d6.tar.gz'}
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
