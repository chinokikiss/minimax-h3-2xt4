"""One-off CPU transfer of existing private Kaggle input to private HF.

No new Kaggle data upload. The user explicitly authorizes temporary auth in
this private operator-input migration; never publish its generated notebook.
"""
import json
from pathlib import Path
import sys
from huggingface_hub import get_token
from t4_notebook_watchdog import supervise_worker


def main():
    original,manifest_path,output=map(Path,sys.argv[1:])
    if output.exists():raise FileExistsError('Preserve submitted sources')
    token=get_token()
    if not token:raise RuntimeError('Local HF login required')
    manifest=json.loads(manifest_path.read_text())
    notebook=json.loads((original/'experiment.ipynb').read_text());code=''.join(notebook['cells'][0]['source'])
    body='''import json,os,pathlib,sys,hashlib,threading,time
pathlib.Path(__file__).unlink(missing_ok=True)
env=json.loads(pathlib.Path('/kaggle/working/light_setup/execution_env.json').read_text())
os.environ.update(env);sys.path[:0]=env['PYTHONPATH'].split(os.pathsep)
os.environ.update(HF_HUB_DISABLE_XET='1',HF_HUB_DISABLE_PROGRESS_BARS='1')
from huggingface_hub import HfApi,CommitOperationAdd
OUT=pathlib.Path('/kaggle/working/input_migration');OUT.mkdir(exist_ok=True)
manifest=MANIFEST
paths=list(pathlib.Path('/kaggle/input').rglob(manifest['file']))
assert len(paths)==1
with paths[0].open('rb') as stream:assert hashlib.file_digest(stream,'sha256').hexdigest()==manifest['compressed_sha256']
assert paths[0].stat().st_size==manifest['compressed_bytes']
api=HfApi(token=TEMPORARY_AUTH)
repo='cnmds/h3-kaggle-private-inputs';info=api.repo_info(repo,repo_type='dataset',files_metadata=True)
assert info.private
manifest_path=OUT/'input_manifest.json';manifest_path.write_text(json.dumps(manifest,indent=2))
stop=threading.Event()
def heartbeat():
    start=time.monotonic()
    while not stop.wait(60):print('HF_MIGRATION_HEARTBEAT',round(time.monotonic()-start),flush=True)
threading.Thread(target=heartbeat,daemon=True).start()
try:
    print('HF_MIGRATION_UPLOAD_START',manifest['compressed_bytes'],flush=True)
    existing=next((f for f in info.siblings if f.rfilename==manifest['file']),None)
    if existing is not None and existing.lfs and existing.lfs.sha256==manifest['compressed_sha256']:
        revision=info.sha
    else:
        commit=api.create_commit(repo_id=repo,repo_type='dataset',commit_message='Preserve hash-bound H3 retained input',operations=[
            CommitOperationAdd(path_in_repo=manifest['file'],path_or_fileobj=paths[0]),
            CommitOperationAdd(path_in_repo='input_manifest.json',path_or_fileobj=manifest_path)])
        revision=commit.oid
    record=dict(repo=repo,repo_type='dataset',private=True,revision=revision,manifest=manifest)
    (OUT/'private_input_publication.json').write_text(json.dumps(record,indent=2))
    print('HF_MIGRATION_COMPLETE',revision,flush=True)
finally:stop.set()
'''.replace('MANIFEST',repr(manifest)).replace('TEMPORARY_AUTH',repr(token))
    code+='\n'+supervise_worker(body,budget=600,output='/kaggle/working/input_migration_supervision')
    code+="\nassert pathlib.Path('/kaggle/working/input_migration/private_input_publication.json').exists(), 'HF input migration incomplete'\n"
    output.mkdir(parents=True);notebook['cells'][0]['source']=code.splitlines(keepends=True)
    (output/'experiment.ipynb').write_text(json.dumps(notebook))
    meta=json.loads((original/'kernel-metadata.json').read_text())
    meta.update(id='pamcks/h3-hf-input-migration',title='H3 HF Input Migration',enable_gpu=False,
        dataset_sources=['pamcks/h3-retained-component-inputs-20261021'],kernel_sources=[])
    (output/'kernel-metadata.json').write_text(json.dumps(meta,indent=2))
    print(output)


if __name__=='__main__':main()
