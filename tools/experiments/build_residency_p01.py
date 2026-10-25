"""Freeze qualified residency policies; only source uploads from this machine."""
import base64
import hashlib
import io
import json
from pathlib import Path
import re
import tarfile
import time

ROOT=Path(__file__).resolve().parents[2]


def main():
    notebook=json.loads((ROOT/'kaggle_te_io_audit/te_io_audit.ipynb').read_text())
    encoded=re.search(r"b64decode\('([^']+)'\)",''.join(notebook['cells'][0]['source'])).group(1)
    with tarfile.open(fileobj=io.BytesIO(base64.b64decode(encoded)),mode='r:gz') as archive:
        files={m.name:archive.extractfile(m).read() for m in archive.getmembers()}
    overlays=('patches/host_memory_pressure.py','patches/host_weight_cache.py','patches/weight_residency.py',
        'patches/comfy_tp_runtime.py','benchmarks/causal_trace.py','benchmarks/residency_pipeline_ablation.py',
        'benchmarks/residency_experiment_matrix.py','benchmarks/residency_metrics.py',
        'benchmarks/test_residency_policy.py','benchmarks/test_residency_spawn.py',
        'tools/experiments/analyze_causal_audit.py','tools/experiments/analyze_cpu_path.py','tools/experiments/analyze_te_path.py')
    for name in overlays:files[name]=(ROOT/name).read_bytes()
    for policy in ('p0','p0_p1'):
        proof=json.loads((ROOT/f'semantic_results/residency_qualification/{policy}/result.json').read_text())
        if not proof['passed'] or not proof['all_outputs_bit_exact_with_uninstrumented']:
            raise RuntimeError('Native qualification failed: '+policy)
        for name,sha in proof['source_sha256'].items():
            if name.startswith('patches/') and name in files and hashlib.sha256(files[name]).hexdigest()!=sha:
                raise RuntimeError('Inference source differs from qualification: '+name)
        files[f'semantic_results/residency_qualification/{policy}.json']=json.dumps(proof).encode()
    # Native artifacts are restored on Kaggle; expiring read capabilities stay private.
    from kaggle.api.kaggle_api_extended import KaggleApi,ApiListKernelSessionOutputRequest
    import requests
    api=KaggleApi();api.authenticate();request=ApiListKernelSessionOutputRequest()
    request.user_name='chinokiki9';request.kernel_slug='minimax-h3-causal-pipeline-audit'
    old=json.loads(files['resume_reference_ticket.json']);objects={}
    with api.build_kaggle_client() as client:
        while True:
            page=client.kernels.kernels_api_client.list_kernel_session_output(request)
            for item in page.files or []:
                name=Path(item.file_name).name
                if not item.file_name.startswith('causal_audit_results/native/native_reference/') or name not in old['objects']:continue
                try:
                    with requests.get(item.url,headers={'Range':'bytes=0-0'},stream=True,timeout=(10,45)) as response:
                        if response.status_code not in (200,206):
                            raise RuntimeError('Native reference unavailable (HTTP '+str(response.status_code)+')')
                        size=int(response.headers['Content-Range'].split('/')[-1]) if response.status_code==206 else int(response.headers['Content-Length'])
                except requests.RequestException:
                    raise RuntimeError('Native reference transport unavailable; capabilities omitted') from None
                objects[name]=dict(url=item.url,bytes=size,shape=old['objects'][name]['shape'])
            if not page.next_page_token:break
            request.page_token=page.next_page_token
    if set(objects)!=set(old['objects']):raise RuntimeError('Incomplete Native reference set')
    files['resume_reference_ticket.json']=json.dumps(dict(created_at=time.time(),objects=objects)).encode()
    plan=json.loads(files['diagnostic_plan.json'])
    plan['resume']['source_sha256']={name:hashlib.sha256(files[name]).hexdigest()
        for name in plan['resume']['source_sha256']}
    plan['resume']['scope']='Preserved physical Native outputs; locally qualified opt-in P0/P1 scheduling changes'
    files['diagnostic_plan.json']=json.dumps(plan).encode()
    bootstrap=files['benchmarks/te_kaggle_bootstrap.py'].decode().replace('te_latency_results','residency_results')
    start=bootstrap.index("        manifest.update(status='te_ablations'")
    end=bootstrap.index('    except BaseException as error:',start)
    bootstrap=bootstrap[:start]+'''        command([sys.executable,str(root/'benchmarks/test_residency_policy.py')],60,'policy_contracts.log')
        command([sys.executable,str(root/'benchmarks/test_residency_spawn.py')],60,'spawn_contract.log')
        manifest.update(status='residency_ablations',diagnostic_only=False,optimization_changes=True,
            matrix=['baseline','p0','p0_p1'],scope='Full Native Ref2VA, bounded P0 lifecycle and independent cgroup-aware P1');save()
        command([sys.executable,'-u',str(root/'benchmarks/residency_experiment_matrix.py'),'--output',str(out)],14400,'residency_matrix.log')
        manifest.update(status='complete',seconds=time.monotonic()-start);save()
'''+bootstrap[end:]
    bootstrap=bootstrap.replace('deadline=start+28800','deadline=start+19800').replace('budget_seconds=28800','budget_seconds=19800')
    compile(bootstrap,'residency_kaggle_bootstrap.py','exec')
    files['benchmarks/residency_kaggle_bootstrap.py']=bootstrap.encode()
    # Do not carry unused read-probe scripts/results into this new job.
    for name in list(files):
        if name in ('benchmarks/te_io_quickstart.py','benchmarks/te_io_probe.py','benchmarks/te_io_pipeline.py','benchmarks/te_io_kaggle_bootstrap.py'):
            del files[name]
    buffer=io.BytesIO()
    with tarfile.open(fileobj=buffer,mode='w:gz') as archive:
        for name,data in files.items():
            member=tarfile.TarInfo(name);member.size=len(data);archive.addfile(member,io.BytesIO(data))
    source="""import base64,io,pathlib,runpy,tarfile
root=pathlib.Path('/kaggle/working/causal_audit_source');root.mkdir(exist_ok=True)
with tarfile.open(fileobj=io.BytesIO(base64.b64decode(PAYLOAD)),mode='r:gz') as archive:archive.extractall(root,filter='data')
runpy.run_path(str(root/'benchmarks/residency_kaggle_bootstrap.py'),run_name='__main__')
""".replace('PAYLOAD',repr(base64.b64encode(buffer.getvalue()).decode()))
    compile(source,'residency_notebook','exec')
    if len(source)>1_000_000:raise RuntimeError('Notebook source limit exceeded')
    notebook['cells'][0]['source']=source.splitlines(keepends=True)
    out=ROOT/'kaggle_residency_p01';out.mkdir(exist_ok=True)
    (out/'residency_p01.ipynb').write_text(json.dumps(notebook))
    metadata=json.loads((ROOT/'kaggle_te_io_audit/kernel-metadata.json').read_text())
    metadata.update(id='chinokiki9/minimax-h3-residency-p0-p1',title='MiniMax H3 Residency P0 P1',code_file='residency_p01.ipynb')
    (out/'kernel-metadata.json').write_text(json.dumps(metadata,indent=2))
    (out/'source_binding.json').write_text(json.dumps({name:hashlib.sha256(data).hexdigest()
        for name,data in files.items() if name!='resume_reference_ticket.json'},indent=2))
    print('FROZEN qualified residency policies; private capabilities omitted; source bytes',len(source),flush=True)


if __name__=='__main__':main()
