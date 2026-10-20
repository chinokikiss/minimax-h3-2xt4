"""Authorized private upload of a bounded source/reference bundle."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
import upload_real_attention_dataset as transfer
transfer.FOLDER=Path('D:/attention_eval/quality_first_20261014/assets')
transfer.STATE=transfer.FOLDER.parent/'asset_upload_state.json'
api=transfer.ParallelPrivateUpload();api.authenticate()
response=api.dataset_create_new(str(transfer.FOLDER),public=False,quiet=True,convert_to_csv=False)
print('PRIVATE_ASSETS_CREATE',response.status,response.url)
if response.status.lower()!='ok':raise RuntimeError(response.error)
