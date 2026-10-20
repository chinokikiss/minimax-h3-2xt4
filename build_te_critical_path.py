"""Freeze Native-qualified TE ablations; weights/references transfer on Kaggle."""
import base64
import hashlib
import io
import json
from pathlib import Path
import re
import tarfile
import time

ROOT=Path(__file__).resolve().parent


def main():
    proof=json.loads((ROOT/'semantic_results/te_loading_qualification/coalesced_parallel_priority_profile/result.json').read_text())
    if not proof['passed'] or not proof['all_outputs_bit_exact_with_uninstrumented']:
        raise RuntimeError('Targeted TE capture must pass bit-exact local equivalence')
    # Reuse the exact physical source closure, preserving unrelated work.
    notebook=json.loads((ROOT/'kaggle_cpu_audit/cpu_audit.ipynb').read_text())
    code=''.join(notebook['cells'][0]['source'])
    encoded=re.search(r"b64decode\('([^']+)'\)",code).group(1)
    with tarfile.open(fileobj=io.BytesIO(base64.b64decode(encoded)),mode='r:gz') as archive:
        files={m.name:archive.extractfile(m).read() for m in archive.getmembers()}
    overlays=['patches/diagnostic_metrics.py','patches/te_loading.py','patches/host_weight_cache.py',
        'patches/weight_residency.py','patches/comfy_tp_runtime.py','benchmarks/causal_pipeline_audit.py',
        'benchmarks/cpu_observer_supervisor.py','benchmarks/te_pipeline_ablation.py','benchmarks/te_experiment_matrix.py',
        'analyze_causal_audit.py','analyze_cpu_path.py','analyze_te_path.py']
    for name in overlays:files[name]=(ROOT/name).read_bytes()
    for name,expected in proof['source_sha256'].items():
        if name in files and hashlib.sha256(files[name]).hexdigest()!=expected:
            raise RuntimeError('Qualified inference source mismatch: '+name)
    if hashlib.sha256(files['patches/te_loading.py']).hexdigest()!=proof['candidate_sha256']:
        raise RuntimeError('TE loading candidate changed after qualification')
    if hashlib.sha256(files['benchmarks/te_pipeline_ablation.py']).hexdigest()!=proof['observer_sha256']:
        raise RuntimeError('TE observer changed after qualification')
    plan=json.loads(files['diagnostic_plan.json'])
    plan['resume']['source_sha256']={name:hashlib.sha256(files[name]).hexdigest()
        for name in plan['resume']['source_sha256']}
    plan['resume']['scope']='Saved physical Native outputs; neutral bounded diagnostic cleanup and qualified TE loading/capture candidates'
    files['diagnostic_plan.json']=json.dumps(plan).encode()
    # Refresh object-scoped read capabilities; never print them or upload
    # local weights/tensors. The destination notebook is private.
    from kaggle.api.kaggle_api_extended import KaggleApi,ApiListKernelSessionOutputRequest
    import requests
    api=KaggleApi();api.authenticate();request=ApiListKernelSessionOutputRequest()
    request.user_name='chinokiki9';request.kernel_slug='minimax-h3-causal-pipeline-audit'
    objects={};old=json.loads(files['resume_reference_ticket.json'])
    with api.build_kaggle_client() as client:
        while True:
            page=client.kernels.kernels_api_client.list_kernel_session_output(request)
            for item in page.files or []:
                name=Path(item.file_name).name
                if not item.file_name.startswith('causal_audit_results/native/native_reference/') or name not in old['objects']:continue
                with requests.get(item.url,headers={'Range':'bytes=0-0'},stream=True,timeout=(10,45)) as response:
                    if response.status_code!=206:raise RuntimeError('Native reference unavailable')
                    count=int(response.headers['Content-Range'].split('/')[-1])
                objects[name]=dict(url=item.url,bytes=count,shape=old['objects'][name]['shape'])
            if not page.next_page_token:break
            request.page_token=page.next_page_token
    if set(objects)!=set(old['objects']):raise RuntimeError('Incomplete saved Native reference')
    files['resume_reference_ticket.json']=json.dumps(dict(created_at=time.time(),objects=objects)).encode()
    bootstrap=files['benchmarks/cpu_kaggle_bootstrap.py'].decode()
    bootstrap=bootstrap.replace('cpu_audit_results','te_latency_results')
    marker="        manifest['status']='short_diagnostic';save()"
    start=bootstrap.index(marker);end=bootstrap.index('    except BaseException as error:',start)
    bootstrap=bootstrap[:start]+'''        manifest.update(status='te_ablations',diagnostic_only=False,optimization_changes=True,
            matrix=['baseline','coalesced','coalesced_parallel'],scope='Full Native Ref2VA requests; opt-in TE loading ablations');save()
        command([sys.executable,'-u',str(root/'benchmarks/te_experiment_matrix.py'),'--output',str(out)],12600,'te_matrix.log')
        manifest.update(status='complete',seconds=time.monotonic()-start);save()
'''+bootstrap[end:]
    compile(bootstrap,'te_kaggle_bootstrap.py','exec')
    files['benchmarks/te_kaggle_bootstrap.py']=bootstrap.encode()
    files['semantic_results/te_loading_qualification/result.json']=json.dumps(proof).encode()
    buffer=io.BytesIO()
    with tarfile.open(fileobj=buffer,mode='w:gz') as archive:
        for name,data in files.items():
            member=tarfile.TarInfo(name);member.size=len(data);archive.addfile(member,io.BytesIO(data))
    source='''import base64,io,pathlib,runpy,tarfile
root=pathlib.Path('/kaggle/working/causal_audit_source');root.mkdir(exist_ok=True)
with tarfile.open(fileobj=io.BytesIO(base64.b64decode(PAYLOAD)),mode='r:gz') as archive:archive.extractall(root,filter='data')
runpy.run_path(str(root/'benchmarks/te_kaggle_bootstrap.py'),run_name='__main__')
'''.replace('PAYLOAD',repr(base64.b64encode(buffer.getvalue()).decode()))
    compile(source,'te_notebook','exec')
    if len(source)>1_000_000:raise RuntimeError('Notebook source limit exceeded')
    notebook['cells'][0]['source']=source.splitlines(keepends=True)
    out=ROOT/'kaggle_te_critical_path';out.mkdir(exist_ok=True)
    (out/'te_critical_path.ipynb').write_text(json.dumps(notebook))
    metadata=json.loads((ROOT/'kaggle_cpu_audit/kernel-metadata.json').read_text())
    metadata.update(id='chinokiki9/minimax-h3-te-critical-path-ablations',
        title='MiniMax H3 TE Critical Path Ablations',code_file='te_critical_path.ipynb')
    (out/'kernel-metadata.json').write_text(json.dumps(metadata,indent=2))
    (out/'source_binding.json').write_text(json.dumps({name:hashlib.sha256(data).hexdigest()
        for name,data in files.items() if name!='resume_reference_ticket.json'},indent=2))
    print('FROZEN TE ablations and targeted capture; read capabilities omitted',len(source),flush=True)


if __name__=='__main__':main()
