"""Private Kaggle upload using the CLI's official SDK and resumable file uploads.

Use bounded file concurrency to avoid one slow connection serializing 170 files.
Credentials and resumable upload tokens are never copied into experiment evidence.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import re
import threading
import time
from kaggle.api.kaggle_api_extended import KaggleApi,ResumableUploadResult
import requests
import io

FOLDER=Path('D:/attention_eval/t4_redo_20261013/dataset')
STATE=FOLDER.parent/'upload_state.json'


def redact(message):
    message=re.sub(r'https?://\S+','[redacted URL]',str(message))
    return re.sub(r'(upload_id|token|signature)=[^&\s)]+',r'\1=[redacted]',message)


class ParallelPrivateUpload(KaggleApi):
    def upload_complete(self,path,url,quiet,resume=False):
        """Send bounded GCS resumable chunks and record server-acknowledged bytes."""
        size=Path(path).stat().st_size
        start=0
        if resume:
            try:
                with requests.Session() as query:
                    response=query.put(url,headers={'Content-Length':'0','Content-Range':f'bytes */{size}'},timeout=(20,60))
            except requests.RequestException:return ResumableUploadResult.INCOMPLETE
            with self.upload_lock:
                self.upload_state['files'][Path(path).name]['resume_http_status']=response.status_code
                temp=STATE.with_suffix('.tmp');temp.write_text(json.dumps(self.upload_state,indent=2));temp.replace(STATE)
            if response.status_code in (404,410):
                # Provider confirms the resumable session no longer exists.
                Path(self.upload_context.get_upload_info_file_path(path)).unlink(missing_ok=True)
                return ResumableUploadResult.FAILED
            if response.status_code in (200,201):return ResumableUploadResult.COMPLETE
            if response.status_code!=308:
                return ResumableUploadResult.INCOMPLETE if response.status_code>=500 else ResumableUploadResult.FAILED
            start=self._get_bytes_already_uploaded(response,True)+1
        with requests.Session() as session,open(path,'rb') as stream:
            while start<size:
                stream.seek(start);block=stream.read(min(256*1024,size-start))
                end=start+len(block)-1
                try:
                    response=session.put(url,data=io.BytesIO(block),headers={'Content-Length':str(len(block)),
                        'Content-Range':f'bytes {start}-{end}/{size}'},timeout=(20,120))
                except requests.RequestException as error:
                    with self.upload_lock:
                        self.upload_state['files'][Path(path).name]['transfer_exception']=type(error).__name__
                        temp=STATE.with_suffix('.tmp');temp.write_text(json.dumps(self.upload_state,indent=2));temp.replace(STATE)
                    return ResumableUploadResult.INCOMPLETE
                if response.status_code in (200,201):accepted=size
                elif response.status_code==308:
                    value=response.headers.get('Range')
                    if not value:return ResumableUploadResult.INCOMPLETE
                    accepted=int(value.rsplit('-',1)[-1])+1
                else:
                    print('UPLOAD_HTTP_STATUS',Path(path).name,response.status_code,flush=True)
                    return ResumableUploadResult.INCOMPLETE if response.status_code>=500 else ResumableUploadResult.FAILED
                if accepted<=start:return ResumableUploadResult.INCOMPLETE
                start=accepted
                with self.upload_lock:
                    self.upload_state['files'][Path(path).name]['server_acknowledged_bytes']=accepted
                    temp=STATE.with_suffix('.tmp');temp.write_text(json.dumps(self.upload_state,indent=2));temp.replace(STATE)
                if response.status_code in (200,201):return ResumableUploadResult.COMPLETE
        return ResumableUploadResult.INCOMPLETE

    def upload_files(self, request, resources, folder, blob_type, upload_context,
                     quiet=False, dir_mode='skip', ignore_patterns=None):
        assert request.is_private, 'Private dataset required'
        assert dir_mode=='skip' and not ignore_patterns and not resources
        paths=sorted(p for p in Path(folder).iterdir() if p.is_file() and p.name!=self.DATASET_METADATA_FILE)
        state={'status':'uploading','private':True,'started':time.time(),'concurrency':8,
               'total_files':len(paths),'total_bytes':sum(p.stat().st_size for p in paths),'files':{}}
        lock=threading.Lock()
        self.upload_state,self.upload_lock,self.upload_context=state,lock,upload_context
        def save():
            temp=STATE.with_suffix('.tmp');temp.write_text(json.dumps(state,indent=2));temp.replace(STATE)
        def one(path):
            with lock:
                state['files'][path.name]={'status':'uploading','bytes':path.stat().st_size,'started':time.time()};save()
            print('UPLOAD_BEGIN',path.name,path.stat().st_size,flush=True)
            try:
                uploaded=None
                for attempt in range(5):
                    try:
                        uploaded=self._upload_file(path.name,str(path),blob_type,upload_context,True,resources)
                        if uploaded is None:raise RuntimeError('SDK returned no completed upload token')
                        break
                    except Exception as retry_error:
                        if attempt==4:raise
                        with lock:
                            state['files'][path.name].update(retries=attempt+1,last_retry_error=redact(retry_error));save()
                        time.sleep(min(2**(attempt+1),30))
                with lock:
                    state['files'][path.name].update(status='complete',finished=time.time());save()
                print('UPLOAD_COMPLETE',path.name,flush=True)
                return uploaded
            except Exception as error:
                message=redact(error)
                with lock:
                    state['files'][path.name].update(status='failed',error=message);save()
                print('UPLOAD_FAILED',path.name,type(error).__name__,flush=True)
                raise RuntimeError(path.name+': '+type(error).__name__) from None
        completed={};failures=[]
        with ThreadPoolExecutor(max_workers=8) as workers:
            futures={workers.submit(one,p):p for p in paths}
            for future in as_completed(futures):
                path=futures[future]
                try:completed[path.name]=future.result()
                except Exception as error:failures.append(str(error))
        if failures:
            state.update(status='failed',failures=failures);save()
            raise RuntimeError('Upload failed; resumable state retained: '+str(failures))
        for path in paths:
            request.files.append(self._new_file(completed[path.name]))
        state.update(status='blobs_uploaded_awaiting_dataset_creation',finished=time.time());save()


if __name__=='__main__':
    api=ParallelPrivateUpload();api.authenticate()
    for creation_attempt in range(5):
        try:
            response=api.dataset_create_new(str(FOLDER),public=False,quiet=True,convert_to_csv=False)
            break
        except requests.RequestException as error:
            print('UPLOAD_API_RETRY',creation_attempt+1,type(error).__name__,flush=True)
            if creation_attempt==4:raise
            time.sleep(min(2**(creation_attempt+1),30))
    state=json.loads(STATE.read_text()) if STATE.exists() else {}
    state.update(dataset_creation_status=response.status,dataset_url=response.url,
                 dataset_creation_error=response.error)
    if response.status.lower()=='ok':state['status']='dataset_processing'
    STATE.write_text(json.dumps(state,indent=2))
    print('DATASET_CREATE',response.status,response.url,flush=True)
    if response.status.lower()!='ok':raise RuntimeError(response.error)
