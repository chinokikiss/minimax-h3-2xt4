"""Version-bound source capture, including failed submissions, without signed URLs."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelRequest
from kagglesdk.datasets.types.dataset_api_service import ApiGetDatasetRequest

ap=argparse.ArgumentParser();ap.add_argument('kind');ap.add_argument('--version',type=int,required=True);args=ap.parse_args()
out=Path('D:/attention_eval/tp2_p2_20261018')/args.kind/('v'+str(args.version))/'provider';out.mkdir(parents=True,exist_ok=True)
api=KaggleApi();api.authenticate();req=ApiGetKernelRequest();req.user_name='chinokiki9';req.kernel_slug='h3-tp2-p2-'+args.kind;req.version_label='v'+str(args.version)
with api.build_kaggle_client() as client:
    response=client.kernels.kernels_api_client.get_kernel(req)
    source=response.blob.source
    (out/'experiment.ipynb').write_text(source,encoding='utf8')
    metadata=response.metadata
    info=dict(kernel='chinokiki9/'+req.kernel_slug,requested_version=args.version,metadata_current_version=metadata.current_version_number,
        docker_image=metadata.docker_image,dataset_sources=metadata.dataset_data_sources,machine_shape=metadata.machine_shape,
        provider_notebook_sha256=hashlib.sha256(source.encode()).hexdigest(),datasets=[])
    for dataset in metadata.dataset_data_sources:
        request=ApiGetDatasetRequest();request.owner_slug,request.dataset_slug=dataset.split('/')[:2]
        d=client.datasets.dataset_api_client.get_dataset(request)
        info['datasets'].append(dict(dataset=dataset,version=d.current_version_number,last_updated=str(d.last_updated)))
    (out/'identity.json').write_text(json.dumps(info,indent=2))
notebook=json.loads(source);code=''.join(notebook['cells'][0]['source']);tree=ast.parse(code)
current=None;hashes={}
for node in tree.body:
    if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='p' for t in node.targets) and isinstance(node.value,ast.BinOp) and isinstance(node.value.right,ast.Constant):current=node.value.right.value
    if isinstance(node,ast.Expr) and isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Attribute) and node.value.func.attr=='write_text' and isinstance(node.value.func.value,ast.Name) and node.value.func.value.id=='p' and current:
        content=ast.literal_eval(node.value.args[0]);dest=out/'submitted_source'/current;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text(content,encoding='utf8')
        hashes[current]=hashlib.sha256(content.encode()).hexdigest()
(out/'source_hashes.json').write_text(json.dumps(hashes,indent=2));print('ARCHIVED_PROVIDER',req.kernel_slug,args.version,len(hashes))
