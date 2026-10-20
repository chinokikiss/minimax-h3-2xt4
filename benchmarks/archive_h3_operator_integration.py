"""Bind the integration results to the provider's immutable submitted version."""
import hashlib
import ast
import json
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelRequest
from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelSessionStatusRequest

root=Path(__file__).resolve().parents[1]
evidence=Path('D:/attention_eval/h3_operator_integration_20261018')
provider=evidence/'t4/v1/provider';provider.mkdir(parents=True,exist_ok=True)
api=KaggleApi();api.authenticate();request=ApiGetKernelRequest()
request.user_name='chinokiki9';request.kernel_slug='h3-tp2-operator-integration';request.version_label='v1'
with api.build_kaggle_client() as client:
    response=client.kernels.kernels_api_client.get_kernel(request)
    query=ApiGetKernelSessionStatusRequest();query.user_name=request.user_name;query.kernel_slug=request.kernel_slug;query.version_label='v1'
    provider_status=str(client.kernels.kernels_api_client.get_kernel_session_status(query).status)
assert provider_status.endswith('.COMPLETE'),provider_status
source=response.blob.source
(provider/'experiment.ipynb').write_text(source,encoding='utf8')
identity=dict(kernel='chinokiki9/h3-tp2-operator-integration',version=1,
    notebook_sha256=hashlib.sha256(source.encode()).hexdigest(),docker_image=response.metadata.docker_image,
    machine_shape=response.metadata.machine_shape,datasets=response.metadata.dataset_data_sources)
(provider/'identity.json').write_text(json.dumps(identity,indent=2))
runtime=evidence/'t4/v1/integration'
expected=json.loads((evidence/'submitted_sources.json').read_text())
actual=json.loads((runtime/'source_hashes.json').read_text());assert actual==expected
notebook=json.loads(source);code=''.join(notebook['cells'][0]['source'])
embedded={};current=None
for node in ast.parse(code).body:
    if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='p' for t in node.targets) and isinstance(node.value,ast.BinOp):
        current=ast.literal_eval(node.value.right)
    if (isinstance(node,ast.Expr) and isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Attribute)
            and node.value.func.attr=='write_text' and isinstance(node.value.func.value,ast.Name) and node.value.func.value.id=='p' and current):
        embedded[current]=hashlib.sha256(ast.literal_eval(node.value.args[0]).encode()).hexdigest()
assert embedded==expected,'Provider embedded source differs from submitted/runtime bytes'
stable=[];changed=[]
for name,digest in actual.items():
    snapshot=evidence/'submitted_source'/name
    assert hashlib.sha256(snapshot.read_text(encoding='utf8').encode()).hexdigest()==digest
    (stable if hashlib.sha256((root/name).read_text(encoding='utf8').encode()).hexdigest()==digest else changed).append(name)
ranks=[json.loads((runtime/f'gates_rank{r}.json').read_text()) for r in (0,1)]
assert all(r['hardware']=='Tesla T4' and all(c['status'] in ('pass','bitwise_pass') for c in r['records']) for r in ranks)
status=json.loads((runtime/'status.json').read_text());assert status['status']=='complete'
manifest=json.loads((evidence/'t4/v1/download_manifest.json').read_text())
for record in manifest['files']:
    path=evidence/'t4/v1'/record['file']
    with path.open('rb') as stream:assert hashlib.file_digest(stream,'sha256').hexdigest()==record['sha256']
result=dict(status='pass',provider=identity,provider_status=provider_status,source_identity_gate=True,tested_sources=actual,
    final_checkout_identical_to_target_snapshot=stable,post_target_snapshot_changes=changed,
    ranks=ranks,runtime=status,
    scope='Operator integration gates only. Dependency preflight, current ComfyUI inline projection compatibility and final CPU/UI/local GPU checks were added afterward and are recorded separately; no full H3 pipeline acceptance or performance measurement.')
for name in ('local_gates','local_pipeline_wiring'):
    result[name]=json.loads((evidence/(name+'.json')).read_text())
(evidence/'integration_results.json').write_text(json.dumps(result,indent=2))
print('H3_INTEGRATION_EVIDENCE_PASS',sum(len(r['records']) for r in ranks),'rank gates;',len(stable),'unchanged sources; post-target changes:',changed)
