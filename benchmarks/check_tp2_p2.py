"""One-shot bounded status inspection for the submitted P2 versions."""
import json
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelSessionStatusRequest
api=KaggleApi();api.authenticate();records=[]
with api.build_kaggle_client() as client:
    for kind,version in [('transport',1),('transport',2),('attention',1),('attention',2),('attention',3),('restoration',1),('counter-probe',1),('transport-confirm',1),('scalar-profile',1)]:
        req=ApiGetKernelSessionStatusRequest();req.user_name='chinokiki9';req.kernel_slug='h3-tp2-p2-'+kind;req.version_label='v'+str(version)
        r=client.kernels.kernels_api_client.get_kernel_session_status(req)
        record=dict(kind=kind,version=version,status=str(r.status),failure_message=r.failure_message)
        records.append(record);print(kind,version,record['status'],flush=True)
Path('D:/attention_eval/tp2_p2_20261018/provider_status.json').write_text(json.dumps(records,indent=2))
