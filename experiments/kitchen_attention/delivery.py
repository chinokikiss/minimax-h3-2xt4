"""Source-only immutable delivery and explicit single-shot Kaggle submissions.

No status/list/log endpoints, credential embedding, mounts, or automatic retry.
"""
import argparse
import ast
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tarfile
import urllib.request
import ssl

ROOT=Path(__file__).resolve().parents[2]


def distribution(base,output):
    from benchmarks.build_kaggle_light_delivery import payload_from_distribution,write_distribution
    digest=hashlib.sha256(base.read_bytes()).hexdigest()
    assert base.name==digest+'.tar.gz'
    with tarfile.open(base) as archive:
        parts={m.name:archive.extractfile(m).read() for m in archive if m.isfile()}
    prior=payload_from_distribution(parts); files=dict(prior)
    manifest=json.loads(parts['delivery_manifest.json'])
    assert {n:hashlib.sha256(v).hexdigest() for n,v in prior.items()}==manifest['source_hashes']
    paths=[ROOT/'experiments/__init__.py',*[p for p in (ROOT/'experiments/kitchen_attention').rglob('*')
        if p.is_file() and '__pycache__' not in p.parts]]
    for path in paths:
        name=path.relative_to(ROOT).as_posix()
        assert name not in prior,'Preserve existing frozen capsule entries'
        files[name]=path.read_bytes()
    assert all(files[n]==v for n,v in prior.items()),'Existing frozen source changed'
    write_distribution(output,parts,files,parent=digest)


def publish(evidence,artifact=None):
    artifact=artifact or json.loads((evidence/'distribution/artifact.json').read_text())
    archive=evidence/'distribution'/artifact['file']
    def git(*args,env=None,data=None):
        return subprocess.check_output(['git',*args],cwd=ROOT,env=env,input=data).decode().strip()
    index=Path(git('rev-parse','--git-path','index'))
    if not index.is_absolute():index=ROOT/index
    before=hashlib.sha256(index.read_bytes()).hexdigest()
    custom=evidence/'publication.index'
    if custom.exists():raise FileExistsError('Preserve earlier publication')
    env=dict(os.environ,GIT_INDEX_FILE=str(custom));head=git('rev-parse','HEAD')
    git('read-tree',head,env=env)
    blob=git('hash-object','-w',str(archive));relative='runtime/kaggle_artifacts/'+archive.name
    git('update-index','--add','--cacheinfo','100644,'+blob+','+relative,env=env)
    commit=git('commit-tree',git('write-tree',env=env),'-p',head,env=env,
        data=b'Publish native Kitchen and frozen TileLang attention benchmark capsule\n')
    branch='codex/kitchen-attn-matrix-20261025-'+artifact['sha256'][:8]
    git('update-ref','refs/heads/'+branch,commit)
    transport=dict(os.environ,GIT_TERMINAL_PROMPT='0',GCM_INTERACTIVE='never')
    result=subprocess.run(['git','-c','http.version=HTTP/1.1','-c','http.sslVersion=tlsv1.2',
        '-c','http.postBuffer=52428800','-c','http.lowSpeedLimit=1024','-c','http.lowSpeedTime=60',
        '-c','credential.interactive=never','push','origin',branch],cwd=ROOT,env=transport,capture_output=True,text=True,timeout=600)
    (evidence/'publication_push.log').write_text(result.stdout+result.stderr)
    if result.returncode:raise RuntimeError('Source archive publication failed')
    url='https://raw.githubusercontent.com/chinokikiss/minimax-h3-2xt4/'+commit+'/'+relative
    context=ssl.create_default_context();context.minimum_version=context.maximum_version=ssl.TLSVersion.TLSv1_2
    with urllib.request.urlopen(url,timeout=90,context=context) as stream:raw=stream.read(artifact['bytes']+1)
    assert len(raw)==artifact['bytes'] and hashlib.sha256(raw).hexdigest()==artifact['sha256']
    assert hashlib.sha256(index.read_bytes()).hexdigest()==before
    record=artifact|dict(repository='chinokikiss/minimax-h3-2xt4',branch=branch,commit=commit,path=relative,
        url=url,published=True,working_tree_and_index_changed=False)
    (evidence/'publication.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(dict(commit=commit,bytes=artifact['bytes'],index_preserved=True)))


def notebook(evidence,mode,qualification=None,*,suite='matrix'):
    from benchmarks.t4_notebook_watchdog import supervise_worker
    pub=json.loads((evidence/'publication.json').read_text());assert pub['published']
    job=evidence/(mode+'_job');job.mkdir()
    core=None;new=None
    if mode=='gpu':
        core=json.loads((qualification/'light_setup/setup_receipt.json').read_text())
        new=json.loads((qualification/'native_cpu/result.json').read_text())
        assert core['status']==new['status']=='cpu_ready' and core['capsule_sha256']==pub['capsule_sha256']
        assert new['tests']['passed']
        assert new['cpu_native_compile']['status']=='passed','Qualify the native Linux build on CPU first'
    body='''import hashlib,io,json,os,pathlib,sys,tarfile,urllib.request
PUB=PUBLICATION
DIST=pathlib.Path('/kaggle/temp/h3_light_distribution');DIST.mkdir(parents=True,exist_ok=True)
SRC=pathlib.Path('/kaggle/temp/h3_light_source');SRC.mkdir(parents=True,exist_ok=True)
OUT=pathlib.Path('/kaggle/working/light_setup');OUT.mkdir(parents=True,exist_ok=True)
BASE=PUB.get('base')
if BASE:
 with urllib.request.urlopen(BASE['url'],timeout=90) as response:base_raw=response.read(BASE['bytes']+1)
 assert len(base_raw)==BASE['bytes'] and hashlib.sha256(base_raw).hexdigest()==BASE['sha256']
 with tarfile.open(fileobj=io.BytesIO(base_raw),mode='r:gz') as archive:archive.extractall(DIST,filter='data')
with urllib.request.urlopen(PUB['url'],timeout=90) as response:raw=response.read(PUB['bytes']+1)
assert len(raw)==PUB['bytes'] and hashlib.sha256(raw).hexdigest()==PUB['sha256']
with tarfile.open(fileobj=io.BytesIO(raw),mode='r:gz') as archive:archive.extractall(DIST,filter='data')
payload=(DIST/'capsule.tar.gz.bin').read_bytes();assert hashlib.sha256(payload).hexdigest()==PUB['capsule_sha256']
with tarfile.open(fileobj=io.BytesIO(payload),mode='r:gz') as archive:archive.extractall(SRC,filter='data')
(OUT/'delivery_binding.json').write_text(json.dumps(PUB,indent=2))
argv=[sys.executable,'-u',str(SRC/'benchmarks/prepare_kaggle_direct_pip.py'),
 '--distribution',str(DIST),'--source',str(SRC),'--output',str(OUT),'--mode',MODE]
QUAL=QUALIFICATION
if QUAL is not None:
 (OUT/'cpu_qualification.json').write_text(json.dumps(QUAL));argv+=['--qualification',str(OUT/'cpu_qualification.json')]
os.execve(sys.executable,argv,dict(os.environ))
'''.replace('PUBLICATION',repr(pub)).replace('MODE',repr('cpu' if mode=='cpu' else 'operators')).replace('QUALIFICATION',repr(core))
    code=supervise_worker(body,budget=240,output='/kaggle/working/light_setup')
    code+='''
import json,pathlib
setup=json.loads(pathlib.Path('/kaggle/working/light_setup/setup_receipt.json').read_text())
assert setup['status']==EXPECTED
'''.replace('EXPECTED',repr('cpu_ready' if mode=='cpu' else 'ready_for_device_gate'))
    execution='''import json,os,pathlib,sys
env=dict(os.environ);env.update(json.loads(pathlib.Path('/kaggle/working/light_setup/execution_env.json').read_text()))
os.chdir('/kaggle/temp/h3_light_source')
argv=[sys.executable,'-u','-m',MODULE,'--mode',MODE,'--output',OUTPUT]
NEW=NEW_QUALIFICATION
if NEW is not None:
 path=pathlib.Path('/kaggle/working/native_cpu_qualification.json');path.write_text(json.dumps(NEW));argv+=['--qualification',str(path)]
os.execve(sys.executable,argv,env)
'''.replace('MODE',repr(mode)).replace('MODULE',repr('experiments.kitchen_attention.dense_batch' if suite=='dense' else 'experiments.kitchen_attention.batch')).replace('OUTPUT',repr('/kaggle/working/native_cpu' if mode=='cpu' else '/kaggle/working/attention_matrix')).replace('NEW_QUALIFICATION',repr(new))
    code+=supervise_worker(execution,budget=360 if mode=='cpu' else (700 if suite=='dense' else 1100),output='/kaggle/working/attention_supervision')
    code+='''
guard=json.loads(pathlib.Path('/kaggle/working/attention_supervision/supervision.json').read_text())
assert guard['status']=='worker_exited' and guard['exit_code']==0,guard
'''
    ast.parse(code)
    meta=dict(id='pamcks/h3-kitchen-attn-'+('cpu' if mode=='cpu' else 't4')+'-20261025',
        title='H3 Kitchen Native Attention '+('CPU' if mode=='cpu' else 'T4 Matrix')+' 20261025',
        code_file='experiment.ipynb',language='python',kernel_type='notebook',is_private=True,
        enable_gpu=mode=='gpu',enable_tpu=False,enable_internet=True,docker_image_pinning_type='original',
        dataset_sources=[],kernel_sources=[],competition_sources=[],model_sources=[])
    if mode=='gpu':meta['machine_shape']='NvidiaTeslaT4'
    if suite=='dense':meta['title']='H3 Kitchen Dense SM75 '+('CPU' if mode=='cpu' else 'T4')+' 20261025'
    # New notebook IDs must match Kaggle's title-derived slug.
    meta['id']='pamcks/'+re.sub(r'[^a-z0-9]+','-',meta['title'].lower()).strip('-')
    book=dict(nbformat=4,nbformat_minor=5,metadata=dict(kernelspec=dict(name='python3',language='python',display_name='Python 3')),
        cells=[dict(id='native-attention',cell_type='code',execution_count=None,metadata={},outputs=[],source=code.splitlines(keepends=True))])
    (job/'experiment.ipynb').write_text(json.dumps(book));(job/'kernel-metadata.json').write_text(json.dumps(meta,indent=2))
    (job/'source_binding.json').write_text(json.dumps(dict(publication=pub,mode=mode,
        core_qualification_sha256=hashlib.sha256(json.dumps(core,sort_keys=True).encode()).hexdigest() if core else None,
        native_qualification_sha256=hashlib.sha256(json.dumps(new,sort_keys=True).encode()).hexdigest() if new else None,
        temporary_private_auth=False,dataset_mounts=0,kernel_mounts=0),indent=2))
    return job


def submit(evidence,mode):
    job=evidence/(mode+'_job');meta=json.loads((job/'kernel-metadata.json').read_text())
    path=evidence/(mode+'_submission.json')
    state=dict(kernel=meta['id'],submitted_utc=dt.datetime.now(dt.timezone.utc).isoformat(),returncode=None,
        notebook_sha256=hashlib.sha256((job/'experiment.ipynb').read_bytes()).hexdigest(),
        source_binding=json.loads((job/'source_binding.json').read_text()),final_collection_count=0,
        status_polling=False,automatic_final_collection=False,
        result_relative='native_cpu/result.json' if mode=='cpu' else 'attention_matrix/batch_results.json')
    with path.open('x') as stream:json.dump(state,stream,indent=2)
    result=subprocess.run([sys.executable,str(ROOT/'benchmarks/kaggle_tls12_runner.py'),'--cli',
        'kernels','push','-p',str(job)],capture_output=True,text=True,timeout=180)
    state.update(returncode=result.returncode,response=result.stdout+result.stderr)
    urls=re.findall(r'https://www\.kaggle\.com/code/([^\s]+)',state['response'])
    if result.returncode==0 and urls:
        state['requested_kernel']=state['kernel'];state['kernel']=urls[-1].rstrip('.')
    path.write_text(json.dumps(state,indent=2)+'\n')
    print(json.dumps(dict(kernel=state['kernel'],returncode=result.returncode,response=state['response'])))
    if result.returncode:raise RuntimeError('Single submission attempt failed; no automatic retry')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--action',choices=('distribution','publish','notebook','submit'),required=True)
    parser.add_argument('--evidence',type=Path,required=True);parser.add_argument('--base',type=Path)
    parser.add_argument('--mode',choices=('cpu','gpu'));parser.add_argument('--qualification',type=Path)
    args=parser.parse_args();args.evidence.mkdir(parents=True,exist_ok=True)
    if args.action=='distribution':distribution(args.base,args.evidence/'distribution')
    elif args.action=='publish':publish(args.evidence)
    elif args.action=='notebook':notebook(args.evidence,args.mode,args.qualification)
    else:submit(args.evidence,args.mode)
