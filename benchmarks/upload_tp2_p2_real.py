"""Authorized private operator-fixture publication using the existing uploader."""
import json
from pathlib import Path
import upload_real_attention_dataset as upload
import inspect
import textwrap

upload.FOLDER=Path('D:/attention_eval/tp2_p2_20261018/seed_dataset')
upload.STATE=upload.FOLDER.parent/'seed_upload_state.json'
# The historical 256KiB workaround is dominated by per-PUT round trips for
# these large matrices. Keep the official resumable protocol with 8MiB chunks.
source=textwrap.dedent(inspect.getsource(upload.ParallelPrivateUpload.upload_complete)).replace('256*1024','8*1024*1024')
exec(compile(source,__file__,'exec'),upload.__dict__)
upload.ParallelPrivateUpload.upload_complete=upload.upload_complete
api=upload.ParallelPrivateUpload();api.authenticate()
response=api.dataset_create_new(str(upload.FOLDER),public=False,quiet=True,convert_to_csv=False)
state=json.loads(upload.STATE.read_text())
state.update(dataset_creation_status=response.status,dataset_url=response.url,dataset_creation_error=response.error)
if response.status.lower()=='ok':state['status']='dataset_processing'
upload.STATE.write_text(json.dumps(state,indent=2))
print('P2_DATASET_CREATE',response.status,response.url,flush=True)
assert response.status.lower()=='ok'
