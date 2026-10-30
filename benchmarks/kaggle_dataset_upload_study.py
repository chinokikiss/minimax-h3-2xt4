"""CPU study of native Kaggle upload clients; private synthetic payloads only."""
import argparse
import hashlib
import importlib.metadata as md
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback


def write(path,value):path.write_text(json.dumps(value,indent=2),encoding='utf-8')


def prepare_payload(base,blocks=16):
    if blocks<1:raise ValueError('Positive payload block count required')
    base.mkdir(parents=True,exist_ok=False)
    folder=base/'payload';folder.mkdir();digest=hashlib.sha256()
    with (folder/'synthetic.dat').open('wb') as stream:
        for _ in range(blocks):
            block=os.urandom(2**20);stream.write(block);digest.update(block)
    return folder,digest.hexdigest()


def operation(kind,folder,handle):
    if kind=='cli':
        entries=[e for e in md.distribution('kaggle').entry_points
            if e.group=='console_scripts' and e.name=='kaggle']
        if len(entries)!=1:raise RuntimeError('Native Kaggle console entry point unavailable')
        # Kaggle 2.0.2 has a console script but no kaggle.__main__ module.
        # Loading this distribution's entry point also avoids another Python's CLI.
        sys.argv=['kaggle','datasets','create','-p',str(folder),'-q','-t']
        code=entries[0].load()()
        if code:raise SystemExit(code)
        return
    if kind=='api':
        from kaggle.api.kaggle_api_extended import KaggleApi
        api=KaggleApi();api.authenticate()
        result=api.dataset_create_new(str(folder),public=False,quiet=True,convert_to_csv=False)
        if getattr(result,'error',None):raise RuntimeError('Dataset create returned an error')
        print('PRIVATE_CREATE_REQUEST_RETURNED',flush=True)
    if kind=='hub':
        import kagglehub
        # Only update a dataset explicitly created private by CLI/API in this run.
        kagglehub.dataset_upload(handle,str(folder),version_notes='Synthetic bounded upload method probe')
        print('PRIVATE_VERSION_REQUEST_RETURNED',flush=True)


def study(a):
    assert not list(Path('/dev').glob('nvidia[0-9]*'))
    a.output.mkdir(parents=True,exist_ok=True)
    state=dict(status='running',cpu_only=True,installs=False,methods=[],
        payload_bytes=16*2**20,scope='Small synthetic private uploads; no H3 weights, activations, credentials or upload throughput extrapolation',
        no_status_or_list_polling=True,credentials_embedded=False,versions={})
    state['phase']='inventory';write(a.output/'result.json',state)
    for name in ('kaggle','kagglehub','kagglesdk','requests','httpx'):
        try:state['versions'][name]=md.version(name)
        except md.PackageNotFoundError:state['versions'][name]=None
    state['phase']='authentication';write(a.output/'result.json',state)
    # Reuse only notebook-native authentication or explicitly attached Secrets.
    # Never copy the local account token into the notebook.
    names=('KAGGLE_API_TOKEN','KAGGLE_USERNAME','KAGGLE_KEY')
    state['credential_presence']={name:bool(os.environ.get(name)) for name in names}
    if not os.environ.get('KAGGLE_API_TOKEN') and not (os.environ.get('KAGGLE_USERNAME') and os.environ.get('KAGGLE_KEY')):
        try:
            from kaggle_secrets import UserSecretsClient
            client=UserSecretsClient()
            for name in names:
                try:
                    value=client.get_secret(name)
                    if value:os.environ[name]=value
                except Exception:pass
        except Exception:pass
    secrets=[os.environ[name] for name in names if os.environ.get(name)]
    state['credential_presence_after_secret_lookup']={name:bool(os.environ.get(name)) for name in names}
    state['phase']='client_inspection';write(a.output/'result.json',state)
    for module_name,function_names in (
            ('kaggle.api.kaggle_api_extended',('KaggleApi',)),('kagglehub',('dataset_upload',))):
        try:
            import importlib
            module=importlib.import_module(module_name)
            if module_name=='kagglehub':
                function=module.dataset_upload
                state['hub_signature']=str(inspect.signature(function))
                (a.output/'native_hub_upload_source.txt').write_text(inspect.getsource(function),encoding='utf-8')
            else:
                cls=module.KaggleApi
                state['api_signatures']={n:str(inspect.signature(getattr(cls,n))) for n in ('dataset_create_new','dataset_create_version')}
        except (Exception,SystemExit) as error:state[module_name+'_inspection_error']=type(error).__name__
    state['phase']='payload_preparation';write(a.output/'result.json',state)
    base=Path('/kaggle/temp/h3_dataset_upload_probe_20261030')
    # Generate on disk in 1 MiB increments; never construct a payload-sized bytes object.
    folder,digest=prepare_payload(base)
    state['payload_sha256']=digest;write(a.output/'result.json',state)
    private_handle=None
    for kind in ('cli','api','hub'):
        state['phase']='upload_'+kind;write(a.output/'result.json',state)
        handle='pamcks/h3-upload-'+kind+'-probe-20261030'
        if kind=='hub':
            if not private_handle:
                state['methods'].append(dict(method=kind,status='blocked',reason='No private dataset create acknowledged in this run'));continue
            handle=private_handle
        if state['versions'].get('kagglehub' if kind=='hub' else 'kaggle') is None:
            state['methods'].append(dict(method=kind,status='unavailable',reason='Native client absent; no unpinned installation'));continue
        write(folder/'dataset-metadata.json',dict(id=handle,title='H3 synthetic '+kind+' upload probe 20261030',licenses=[dict(name='CC0-1.0')]))
        begin=time.monotonic();peak=0;guard=False
        log=base/(kind+'.log')
        with log.open('w') as stream:
            proc=subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),'--output',str(a.output),
                '--operation',kind,'--folder',str(folder),'--handle',handle],stdout=stream,stderr=subprocess.STDOUT)
            while proc.poll() is None:
                if time.monotonic()-begin>180:guard=True;proc.kill();break
                try:
                    status=Path(f'/proc/{proc.pid}/status').read_text()
                    rss=next(int(s.split()[1])*1024 for s in status.splitlines() if s.startswith('VmRSS:'))
                    peak=max(peak,rss)
                    if rss>512*2**20:guard=True;proc.kill();break
                except (OSError,StopIteration):pass
                time.sleep(.25)
            proc.wait(timeout=10)
        text=log.read_text(errors='replace')[-65536:]
        for secret in secrets:text=text.replace(secret,'[REDACTED]')
        (a.output/(kind+'_client.log')).write_text(text,encoding='utf-8')
        row=dict(method=kind,handle=handle,seconds=time.monotonic()-begin,returncode=proc.returncode,
            peak_child_rss_bytes=peak,guard_triggered=guard,status='request_returned' if proc.returncode==0 else 'blocked_or_failed',
            privacy='private create; hub only versions acknowledged private dataset',
            note='Request acknowledgement does not establish completed dataset processing or readback integrity')
        if proc.returncode==0 and kind in ('cli','api'):private_handle=handle
        state['methods'].append(row);write(a.output/'result.json',state)
    state.update(status='complete' if any(r['status']=='request_returned' for r in state['methods']) else 'completed_with_blockers',phase='finished',
        cleanup='Created private synthetic datasets retained; no deletion or status polling attempted')
    write(a.output/'result.json',state)
    print('UPLOAD_STUDY_DONE',state['status'],flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--operation',choices=('cli','api','hub'));p.add_argument('--folder',type=Path);p.add_argument('--handle')
    a=p.parse_args()
    if a.operation:
        operation(a.operation,a.folder,a.handle);return
    try:study(a)
    except BaseException as error:
        a.output.mkdir(parents=True,exist_ok=True)
        path=a.output/'result.json'
        state=json.loads(path.read_text(encoding='utf-8')) if path.exists() else dict(phase='startup',methods=[])
        state.update(status='failed',error_type=type(error).__name__)
        write(path,state)
        text=traceback.format_exc()
        for name in ('KAGGLE_API_TOKEN','KAGGLE_KEY'):
            if os.environ.get(name):text=text.replace(os.environ[name],'[REDACTED]')
        (a.output/'exception.txt').write_text(text,encoding='utf-8')
        raise


if __name__=='__main__':main()
