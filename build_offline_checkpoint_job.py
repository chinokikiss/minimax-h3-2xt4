"""Reuse the qualified bootstrap in a separate CPU-only checkpoint publisher."""
import ast
import argparse
import base64
import hashlib
import io
import json
from pathlib import Path
import secrets
import tarfile

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--transport-ticket', type=Path)
    options = parser.parse_args()
    transport = None
    if options.transport_ticket:
        transport = json.loads(options.transport_ticket.read_text())
        if (len(transport['objects']) != 200 or transport['plan'] !=
                json.loads((ROOT/'semantic_results/offline_conversion_plan.json').read_text())):
            raise ValueError('Complete current Native upload authorization required')
    proof = json.loads((ROOT/'semantic_results/offline_portability.json').read_text())
    if not proof['passed']: raise RuntimeError('Portable binding must pass locally')
    for name, sha in proof['source_sha256'].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest() != sha:
            raise RuntimeError('Portable code changed after qualification')
    notebook = json.loads((ROOT/'kaggle_loading_audit/pipeline_audit.ipynb').read_text())
    code = ''.join(notebook['cells'][0]['source'])
    tree = ast.parse(code)
    payloads = [n.args[0].value for n in ast.walk(tree) if isinstance(n, ast.Call) and
        isinstance(n.func, ast.Attribute) and n.func.attr == 'b64decode' and
        n.args and isinstance(n.args[0], ast.Constant)]
    if len(payloads) != 1: raise RuntimeError('Expected one qualified source archive')
    buffer = io.BytesIO()
    with tarfile.open(fileobj=io.BytesIO(base64.b64decode(payloads[0])), mode='r:gz') as old:
        with tarfile.open(fileobj=buffer, mode='w:gz') as new:
            for member in old.getmembers():
                # CPU publication needs frozen identities/Native hashes, not
                # multi-megabyte profiler histories from local GPU requests.
                if (transport and member.name.startswith('semantic_results/') and
                        member.name not in ('semantic_results/pipeline_checkpoint_identity.json',)):
                    continue
                new.addfile(member, old.extractfile(member) if member.isfile() else None)
            for relative in ('benchmarks/offline_checkpoint_store.py',
                             'benchmarks/qualify_offline_portability.py',
                             'semantic_results/offline_conversion_plan.json',
                             'semantic_results/offline_portability.json'):
                new.add(ROOT/relative, arcname=relative)
            if transport:
                new.add(ROOT/'benchmarks/kaggle_lfs_transport.py', arcname='benchmarks/kaggle_lfs_transport.py')
                new.add(options.transport_ticket, arcname='upload_ticket.json')
    code = code.replace(payloads[0], base64.b64encode(buffer.getvalue()).decode())
    folder = ROOT/'kaggle_offline_checkpoint_job'; folder.mkdir(exist_ok=True)
    settings = folder/'destination_config.json'
    if not settings.exists():
        settings.write_text(json.dumps(dict(secret='HF_token',
            repo_slug='minimax-h3-tp2-int8-'+secrets.token_hex(4)), indent=2))
    destination = json.loads(settings.read_text())
    code = code.replace("gpu_budget_seconds=3000", "gpu_budget_seconds=0")
    code = code.replace("offline_mode='skip'", "offline_mode='publish',conversion_upload_budget_seconds=7200")
    start = code.index("scope='"); end = code.index("')", start)
    code = code[:start]+"scope='CPU-only rank-local conversion and incremental private HF publication; no GPU benchmarks'"+code[end+1:]
    code = code.replace("specs=json.loads((root/'semantic_results/pipeline_checkpoint_identity.json').read_text())",
        "specs=json.loads((root/'semantic_results/pipeline_checkpoint_identity.json').read_text())\nspecs={k:specs[k] for k in ('te','dit')}")
    code = code.replace("+8*2**30", "+46*2**30")
    preflight = """    preflight=\"\"\"import json,pathlib,sys,time
from kaggle_secrets import UserSecretsClient
from huggingface_hub import HfApi,snapshot_download
try:
    stage='kaggle_secret'
    for attempt in range(3):
        try:token=UserSecretsClient().get_secret(SECRET_VALUE);break
        except Exception as error:
            if (type(error).__name__!='ConnectionError' or attempt==2 or
                getattr(getattr(error,'__cause__',None),'code',None) in (400,401,403,404)):raise
            time.sleep(3)
    api=HfApi(token=token)
    stage='hf_identity'
    account=api.whoami()['name'];repo=account+'/'+SLUG_VALUE
    stage='hf_create_repo'
    api.create_repo(repo_id=repo,private=True,exist_ok=True)
    stage='hf_resume'
    files=api.list_repo_files(repo)
    if 'progress.json' in files:
        sha=api.repo_info(repo).sha
        snapshot_download(repo,revision=sha,token=api.token,local_dir='/tmp/rank_local_tp2',allow_patterns=['*.json','*.safetensors'],max_workers=2)
    pathlib.Path(DESTINATION_VALUE).write_text(json.dumps(dict(repo=repo)))
    print('VERIFIED HF destination',repo,flush=True)
except Exception as e:
    response=getattr(e,'response',None)
    cause=getattr(e,'__cause__',None)
    status=getattr(response,'status_code',None) or getattr(cause,'code',None)
    print('ERROR HF preflight',stage,type(e).__module__,type(e).__name__,status,
          type(cause).__name__ if cause else None,flush=True);sys.exit(1)
\"\"\".replace('SECRET_VALUE',repr(SECRET_LABEL)).replace('SLUG_VALUE',repr(REPO_SLUG)).replace('DESTINATION_VALUE',repr(str(root/'offline_destination.json')))
    command([sys.executable,'-u','-c',preflight],300,'hf_preflight.log')
""".replace('SECRET_LABEL',repr(destination['secret'])).replace('REPO_SLUG',repr(destination['repo_slug']))
    anchor="    command([sys.executable,'-m','pip','install','--quiet','--upgrade','torch"
    if not transport:
        code = code.replace(anchor, preflight+anchor, 1)
    begin = code.index('    probe='); end = code.index("except Exception as error:", begin)
    code = code[:begin]+"""    (out/'hardware.json').write_text(json.dumps(dict(cpu_only=True,gpu_requested=False)))
    manifest.update(status='converting_and_publishing',setup_seconds=time.monotonic()-start);save()
    repo=json.loads((root/'offline_destination.json').read_text())['repo']
    duration=command([sys.executable,'-u',str(root/'benchmarks/offline_checkpoint_store.py'),
        '--directory','/tmp/rank_local_tp2','--plan',str(root/'semantic_results/offline_conversion_plan.json'),
        '--checkpoints',str(root/'checkpoint_paths.json'),'--repo',repo,'--secret',SECRET_LABEL],7200,'offline_publication.log')
    bundle=json.loads(pathlib.Path('/tmp/rank_local_tp2/bundle.json').read_text())
    if not bundle['complete'] or len(bundle['blocks'])!=200:raise RuntimeError('Offline publication is incomplete')
    import shutil
    shutil.copyfile('/tmp/rank_local_tp2/bundle.json',out/'bundle.json')
    shutil.copyfile('/tmp/rank_local_tp2/publication.json',out/'publication.json')
    publication=json.loads((out/'publication.json').read_text())
    manifest.update(status='complete',repo=repo,commit=publication['commit'],conversion_upload_seconds=duration,total_seconds=time.monotonic()-start);save()
""".replace('SECRET_LABEL',repr(destination['secret']))+code[end:]
    if transport:
        begin = code.index("    repo=json.loads((root/'offline_destination.json')")
        end = code.index('except Exception as error:', begin)
        code = code[:begin]+"""    duration=command([sys.executable,'-u',str(root/'benchmarks/kaggle_lfs_transport.py'),
        'convert','--directory','/tmp/rank_local_tp2','--ticket',str(root/'upload_ticket.json'),
        '--checkpoints',str(root/'checkpoint_paths.json'),'--evidence',str(out/'publication_evidence')],7200,'offline_publication.log')
    receipts=json.loads((out/'publication_evidence/upload_receipts.json').read_text())
    if not receipts['complete'] or len(receipts['objects'])!=200:raise RuntimeError('Kaggle object upload incomplete')
    manifest.update(status='objects_uploaded_commit_pending',repo=receipts['repo'],
        conversion_upload_seconds=duration,total_seconds=time.monotonic()-start);save()
"""+code[end:]
    # Preserve small evidence even when /tmp checkpoints vanish after execution.
    code = code.replace("if line.startswith(('AUDIT','VERIFIED','Traceback','ERROR')):",
        "if line.startswith(('AUDIT','VERIFIED','PUBLISHED','READY','Traceback','ERROR')):")
    compile(code, 'offline_checkpoint_job', 'exec')
    notebook['cells'][0]['source'] = code.splitlines(keepends=True)
    (folder/'offline_checkpoint.ipynb').write_text(json.dumps(notebook))
    metadata = json.loads((ROOT/'kaggle_loading_audit/kernel-metadata.json').read_text())
    metadata.update(code_file='offline_checkpoint.ipynb',enable_gpu=False,enable_tpu=False,
                    kernel_sources=[],dataset_sources=[])
    (folder/'kernel-metadata.json').write_text(json.dumps(metadata,indent=2))
    print(folder)


if __name__ == '__main__': main()
