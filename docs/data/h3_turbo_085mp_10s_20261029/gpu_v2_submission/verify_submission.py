"""Inspect submission bytes/configuration only; no model execution or CPU smoke."""
import ast
import base64
import hashlib
import json
from pathlib import Path
import subprocess

root=Path(__file__).resolve().parent/'kaggle'
job=root/'gpu_job'
binding=json.loads((job/'source_binding.json').read_text(encoding='utf-8'))
plan=json.loads((root/'request.json').read_text(encoding='utf-8'))
meta=json.loads((job/'kernel-metadata.json').read_text(encoding='utf-8'))
assert (plan['width'],plan['height'],plan['frames'],plan['export_frames'])==(1216,704,243,240)
assert binding['new_cpu_smoke_run'] is False and binding['online_lora'] is False and binding['ptd'] is False
assert meta['id']=='pamcks/h3-turbo-0-85mp-10s-t4-20261029' and meta['enable_gpu'] and meta['is_private']
assert meta['dataset_sources']==meta['kernel_sources']==[]
code=''.join(json.loads((job/'experiment.ipynb').read_text(encoding='utf-8'))['cells'][0]['source'])
tree=ast.parse(code)
workers=[]
for node in ast.walk(tree):
    if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr=='b64decode' and node.args and isinstance(node.args[0],ast.Constant):
        value=base64.b64decode(node.args[0].value).decode()
        if 'target.write_bytes(payload)' in value:workers.append(value)
assert len(workers)==1
worker=workers[0]
assert '/kaggle/working/turbo_085mp_e2e' in worker and '/tmp/h3_turbo_085mp_10s_20261029' in worker
payloads=[base64.b85decode(n.args[0].value) for n in ast.walk(ast.parse(worker)) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='b85decode' and isinstance(n.args[0],ast.Constant)]
assert len(payloads)==1
sha=hashlib.sha256(payloads[0]).hexdigest()
overlay=binding['benchmark_overlay']
assert sha==overlay['sha256']
original=subprocess.check_output(['git','show',overlay['control_commit']+':'+overlay['path']])
assert payloads[0]==original

additions=next(ast.literal_eval(n.iter) for n in ast.walk(ast.parse(worker)) if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='item')
assert len(additions)==1
assert [{k:v for k,v in item.items() if k!='payload_b85'} for item in additions]==overlay['additional_files']
for item in additions:
    value=base64.b85decode(item['payload_b85'])
    assert hashlib.sha256(value).hexdigest()==item['sha256']
    assert value==subprocess.check_output(['git','show',overlay['control_commit']+':'+item['path']])
assert overlay['failure_kind']=='download'

prior=json.loads(Path('D:/attention_eval/h3_turbo_1mp_10s_20261029/revision2/delivery/plan.json').read_text(encoding='utf-8'))
changed={k for k in prior.keys()|plan.keys() if prior.get(k)!=plan.get(k)}
assert changed=={'width','height','name','source_note','geometry_retry_provenance'}
result=dict(benchmark_overlay_payload_verified=True,additional_overlay_payloads_verified=True,benchmark_overlay_sha256=sha,new_cpu_smoke_run=False,
    source_qualification_reused=True,changed_request_fields=sorted(changed),
    width=1216,height=704,megapixels=1216*704/1e6,generated_video_tokens=72*38*22,
    private=True,credentials=False,zero_mounts=True)
(root/'static_binding_verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result))
