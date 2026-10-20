"""Read the pipeline audit's current session and download only its evidence."""
import argparse
import json
from pathlib import Path
import requests
import threading
import time
import fnmatch
from concurrent.futures import ThreadPoolExecutor
from kaggle.api.kaggle_api_extended import KaggleApi, ApiListKernelSessionOutputRequest

SLUG = 'chinokiki9/minimax-h3-pipeline-residency-audit'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--download', action='store_true')
    parser.add_argument('--slug',default=SLUG)
    parser.add_argument('--output',type=Path,default=Path('kaggle_pipeline_audit_output'))
    parser.add_argument('--prefix',default='pipeline_audit_results/')
    parser.add_argument('--resume',action='store_true',help='Reuse complete files and continue byte ranges')
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--exclude',action='append',default=[],help='Exclude output filename glob; repeatable')
    parser.add_argument('--include',action='append',default=[],help='Download only matching filenames; repeatable')
    options = parser.parse_args()
    root = options.output.resolve();root.mkdir(parents=True,exist_ok=True)
    archived={}
    workspace=Path(__file__).resolve().parent
    for receipt in (workspace/'semantic_results/artifact_retention').glob('compression_*.json'):
        for row in json.loads(receipt.read_text())['files']:
            archived[(workspace/row['path']).resolve()]=row
    api = KaggleApi();api.authenticate()
    try:
        status = api.kernels_status(options.slug)
    except (ValueError, requests.RequestException) as error:
        print('STATUS unavailable for this slug:',type(error).__name__,flush=True)
        return
    print('STATUS',status.status,status.failure_message or '',flush=True)
    if 'RUNNING' in str(status.status):
        def stream():
            try:
                with (root/'live.log').open('w',encoding='utf-8') as log:
                    for event in api.kernels_logs_stream(options.slug):
                        data=event.get('data','');log.write(data);log.flush()
                        if any(s in data for s in ('AUDIT','VERIFIED','PUBLISHED','READY','ERROR','NATIVE','TP ','Native ComfyUI','Traceback','Error:')):
                            print(data.rstrip(),flush=True)
            except Exception as error:
                print('Log stream unavailable:',type(error).__name__,flush=True)
        thread=threading.Thread(target=stream,daemon=True);thread.start();thread.join(8)
        # Completed-session output can still refer to the preceding version
        # while a new version runs. Only consume live logs during execution.
        return
    with api.build_kaggle_client() as client:
        request = ApiListKernelSessionOutputRequest()
        request.user_name,request.kernel_slug = options.slug.split('/')
        files = []
        printed_log = False
        while True:
            response = client.kernels.kernels_api_client.list_kernel_session_output(request)
            files.extend(response.files or [])
            if response.log and not printed_log:
                printed_log = True
                (root/'session.log').write_text(response.log,encoding='utf-8')
                try:
                    text=''.join(e.get('data','') for e in json.loads(response.log))
                except (ValueError,TypeError):text=response.log
                meaningful = [line for line in text.splitlines() if any(s in line for s in ('AUDIT','VERIFIED','PUBLISHED','READY','ERROR','NATIVE','TP ','Native ComfyUI','Traceback','Error:'))]
                print('\n'.join(meaningful[-18:]),flush=True)
            if not response.next_page_token:break
            request.page_token = response.next_page_token
    if options.download:
        def download(selected_item):
            # Resolve and validate once on the main thread. Reuse the exact
            # checked absolute path in workers, including while directories
            # are concurrently created by other downloads.
            item,target = selected_item
            try:
                target.parent.mkdir(parents=True,exist_ok=True)
                if options.resume:
                    # HEAD is unsupported by Kaggle's signed output endpoint.
                    with requests.get(item.url,headers={'Range':'bytes=0-0'},stream=True,timeout=(10,90)) as probe:
                        if ((probe.status_code==200 and probe.headers.get('Content-Length')=='0') or
                            (probe.status_code==416 and probe.headers.get('Content-Range')=='bytes */0')):
                            target.write_bytes(b'')
                            print('SAVED empty',item.file_name,flush=True);return
                        if probe.status_code!=206:raise RuntimeError('Output range probe rejected: '+item.file_name)
                        total=int(probe.headers['Content-Range'].split('/')[-1])
                    if target.exists() and target.stat().st_size==total:
                        print('REUSED',item.file_name,total,flush=True);return
                    record=archived.get(target)
                    zipped=target.with_suffix(target.suffix+'.gz')
                    if (not target.exists() and record and record['original_bytes']==total and
                            zipped.exists() and zipped.stat().st_size==record['compressed_bytes']):
                        print('REUSED verified archive',item.file_name,total,flush=True);return
                    temporary=target.with_suffix(target.suffix+'.partial')
                    offset=temporary.stat().st_size if temporary.exists() else 0
                    if offset>total:raise RuntimeError('Partial output exceeds expected size')
                    failures=0
                    while offset<total:
                        stop=min(total-1,offset+64*1024*1024-1)
                        try:
                            with requests.get(item.url,headers={'Range':f'bytes={offset}-{stop}'},
                                              stream=True,timeout=(10,90)) as response:
                                expected=f'bytes {offset}-{stop}/{total}'
                                if response.status_code!=206 or response.headers.get('Content-Range')!=expected:
                                    raise RuntimeError('Output range mismatch; partial file retained')
                                with temporary.open('ab') as output:
                                    for chunk in response.iter_content(1024*1024):
                                        output.write(chunk)
                            offset=temporary.stat().st_size
                            if offset!=stop+1:raise RuntimeError('Incomplete output range')
                            failures=0
                        except requests.RequestException:
                            offset=temporary.stat().st_size if temporary.exists() else 0
                            failures+=1
                            if failures>=3:raise RuntimeError('Evidence range download failed; partial retained') from None
                            time.sleep(2)
                    temporary.replace(target)
                    print('SAVED',item.file_name,total,flush=True);return
                temporary=target.with_suffix(target.suffix+'.partial');size=0
                with requests.get(item.url,stream=True,timeout=(10,90)) as response:
                    if response.status_code!=200:raise RuntimeError('Output HTTP '+str(response.status_code))
                    with temporary.open('wb') as output:
                        for chunk in response.iter_content(1024*1024):
                            output.write(chunk);size+=len(chunk)
                temporary.replace(target)
                print('SAVED',item.file_name,size,flush=True)
            except requests.RequestException:
                raise RuntimeError('Evidence download failed; signed URL omitted') from None
        selected=[];skipped=[]
        for item in files:
            if not item.file_name.startswith(options.prefix) or item.file_name.endswith('.pt'):continue
            if options.include and not any(fnmatch.fnmatch(item.file_name,pattern) for pattern in options.include):continue
            if any(fnmatch.fnmatch(item.file_name,pattern) for pattern in options.exclude):continue
            target=(root/item.file_name).resolve()
            if not target.is_relative_to(root):
                skipped.append(item.file_name);continue
            selected.append((item,target))
        if skipped:
            (root/'skipped_output_paths.json').write_text(json.dumps(skipped,indent=2))
            print('SKIPPED unsafe output paths:',len(skipped),flush=True)
        with ThreadPoolExecutor(max_workers=options.workers) as pool:
            list(pool.map(download,selected))


if __name__=='__main__':main()
