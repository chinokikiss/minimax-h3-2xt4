"""Small private transport manifest; never print signed download links."""
import json
import shutil
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi, ApiListKernelSessionOutputRequest

root=Path(__file__).resolve().parents[2]
source=root/'kaggle_native_audit_reference'
target=root/'kaggle_native_reference_transport';target.mkdir(exist_ok=True)
for name in ('native_result.json','reference_identity.json','dataset-metadata.json'):
    shutil.copyfile(source/name,target/name)
api=KaggleApi();api.authenticate()
links={}
with api.build_kaggle_client() as client:
    request=ApiListKernelSessionOutputRequest()
    request.user_name='chinokiki9';request.kernel_slug='minimax-h3-pipeline-residency-audit'
    while True:
        response=client.kernels.kernels_api_client.list_kernel_session_output(request)
        for item in response.files or []:
            if item.file_name.startswith('pipeline_audit_results/native_reference/'):
                links[Path(item.file_name).name]=item.url
        if not response.next_page_token:break
        request.page_token=response.next_page_token
expected={n+'.pt' for n in ('conditioning','video','audio','pixels','waveform')}
if set(links)!=expected:raise RuntimeError('Native download set is incomplete')
(target/'reference_links.json').write_text(json.dumps(links))
print('PRIVATE TRANSPORT READY; five links retained without logging their values')
