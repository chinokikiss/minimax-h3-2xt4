"""Register three already-uploaded native DiT blobs, without reuploading QKV.

Uses SDK resumable metadata and verifies provider completion. Tokens and URLs
remain within this process. Keep original resumable metadata for the full upload.
"""
import json
import argparse
from pathlib import Path
import requests
from kaggle.api.kaggle_api_extended import KaggleApi,ResumableFileUpload,ApiStartBlobUploadRequest,UploadFile

DATA=Path('D:/attention_eval/t4_redo_20261013/dataset')
OUT=Path('D:/attention_eval/quality_constrained_20261015/dit_subset_metadata');OUT.mkdir(exist_ok=True)
SELECTED=['dit_layer00_call09.pt','dit_layer10_call09.pt','dit_layer25_call09.pt']


class ExistingBlobs(KaggleApi):
    def upload_files(self,request,resources,folder,blob_type,upload_context,quiet=False,dir_mode='skip',ignore_patterns=None):
        assert request.is_private and not resources
        for name in SELECTED:
            path=DATA/name;req=ApiStartBlobUploadRequest();req.type=blob_type;req.name=name
            req.content_length=path.stat().st_size;req.last_modified_epoch_seconds=int(path.stat().st_mtime)
            # Do not register this metadata object in the new context: its exit
            # must not delete resumable metadata needed by the original upload.
            upload=ResumableFileUpload(str(path),req,upload_context);upload.load()
            if not upload.can_resume:raise RuntimeError('No valid completed SDK blob metadata for '+name)
            with requests.Session() as session:
                response=session.put(upload.start_blob_upload_response.create_url,
                    headers={'Content-Length':'0','Content-Range':f'bytes */{path.stat().st_size}'},timeout=(15,30))
            if response.status_code not in (200,201):raise RuntimeError(f'Blob completion not confirmed for {name}: HTTP{response.status_code}; no QKV uploaded')
            upload.upload_completed();item=UploadFile();item.token=upload.get_token();request.files.append(self._new_file(item))
            print('REUSED_COMPLETED_BLOB',name,'transferred_QKV_bytes=0',flush=True)
        manifest=OUT/'capture_manifest.json'
        item=self._upload_file(manifest.name,str(manifest),blob_type,upload_context,True,None)
        if item is None:raise RuntimeError('Small manifest upload failed')
        request.files.append(self._new_file(item))


def main():
    global SELECTED,OUT
    p=argparse.ArgumentParser();p.add_argument('--files',nargs='+');p.add_argument('--output',type=Path)
    p.add_argument('--id',default='chinokiki9/h3-native-dit-profile-subset');p.add_argument('--title',default='H3 Native DiT Profile Subset');args=p.parse_args()
    if args.files:SELECTED=args.files
    if args.output:OUT=args.output;OUT.mkdir(parents=True,exist_ok=True)
    source=json.loads((DATA/'capture_manifest.json').read_text());records=[r for r in source['records'] if r['file'] in SELECTED]
    assert len(records)==len(SELECTED)
    (OUT/'capture_manifest.json').write_text(json.dumps(source|{'records':records},indent=2))
    (OUT/'dataset-metadata.json').write_text(json.dumps({'title':args.title,'id':args.id,'licenses':[{'name':'other'}]},indent=2))
    api=ExistingBlobs();api.authenticate();response=api.dataset_create_new(str(OUT),public=False,quiet=True,convert_to_csv=False)
    record={'status':response.status,'error':response.error,'private':True,'reused_files':SELECTED,'new_QKV_upload_bytes':0}
    (OUT/'creation_status.json').write_text(json.dumps(record,indent=2));print('PRIVATE_SUBSET_CREATE',response.status,flush=True)
    if response.status.lower()!='ok':raise RuntimeError('Dataset registration failed; see redacted status')


if __name__=='__main__':main()
