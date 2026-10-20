"""Freeze a qualified, single-configuration CPU diagnostic on physical dual T4."""
import base64
import hashlib
import io
import json
from pathlib import Path
import re
import sys
import tarfile
import time

ROOT=Path(__file__).resolve().parent


def main():
    proof=json.loads((ROOT/'semantic_results/cpu_qualification/result.json').read_text())
    previous=json.loads((ROOT/'kaggle_causal_resume_output/causal_audit_results/continuous/offline_p1_p2/timing/result.json').read_text())
    if not proof['passed'] or not proof['all_outputs_bit_exact_with_uninstrumented']:
        raise RuntimeError('CPU observers must pass numerical equivalence first')
    for name,sha in {**proof['source_sha256'],**proof['cpu_source_sha256']}.items():
        if name.startswith('patches/') and name not in previous['source_sha256']:
            # Preserve unrelated, concurrently edited helpers on disk. Freeze
            # only the original physical pipeline's source closure.
            symbol=Path(name).stem
            if any(symbol in (ROOT/active).read_text(encoding='utf-8') for active in previous['source_sha256']):
                raise RuntimeError('Unqualified helper referenced by physical pipeline: '+name)
            continue
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=sha:
            raise RuntimeError('CPU qualification source changed: '+name)
    import build_causal_audit as original
    sys.argv=[original.__file__,'--offline-source','public'];original.main()
    notebook=json.loads((ROOT/'kaggle_causal_audit/causal_audit.ipynb').read_text())
    source=''.join(notebook['cells'][0]['source'])
    encoded=re.search(r"b64decode\('([^']+)'\)",source).group(1)
    files={}
    with tarfile.open(fileobj=io.BytesIO(base64.b64decode(encoded)),mode='r:gz') as archive:
        for member in archive.getmembers():files[member.name]=archive.extractfile(member).read()
    files={name:data for name,data in files.items() if not name.startswith('patches/') or name in previous['source_sha256']}
    for name in ('cpu_causal_trace.py','cpu_pipeline_audit.py','cpu_os_sampler.py','cpu_observer_supervisor.py'):
        files['benchmarks/'+name]=(ROOT/'benchmarks'/name).read_bytes()
    files['analyze_cpu_path.py']=(ROOT/'analyze_cpu_path.py').read_bytes()
    files['semantic_results/cpu_qualification/result.json']=(ROOT/'semantic_results/cpu_qualification/result.json').read_bytes()
    reference=ROOT/'kaggle_causal_audit_output/causal_audit_results/native/native_result.json'
    native=json.loads(reference.read_text())
    if not (native['passed'] and native['physical'] and native['layers']==50):raise RuntimeError('Saved Native proof required')
    for name,sha in native['source_sha256'].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=sha:raise RuntimeError('Native source changed')
    for name,sha in previous['source_sha256'].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=sha:raise RuntimeError('Physical inference source changed')
    from kaggle.api.kaggle_api_extended import KaggleApi,ApiListKernelSessionOutputRequest
    import requests
    api=KaggleApi();api.authenticate();request=ApiListKernelSessionOutputRequest()
    request.user_name='chinokiki9';request.kernel_slug='minimax-h3-causal-pipeline-audit'
    objects={};names={n+'.pt' for n in ('conditioning','video','audio','pixels','waveform')}
    with api.build_kaggle_client() as client:
        while True:
            page=client.kernels.kernels_api_client.list_kernel_session_output(request)
            for item in page.files or []:
                if item.file_name.startswith('causal_audit_results/native/native_reference/'):
                    name=Path(item.file_name).name
                    if name not in names:continue
                    try:
                        with requests.get(item.url,headers={'Range':'bytes=0-0'},stream=True,timeout=(10,45)) as response:
                            if response.status_code!=206:raise RuntimeError('Saved reference unavailable')
                            count=int(response.headers['Content-Range'].split('/')[-1])
                    except requests.RequestException as error:
                        raise RuntimeError('Reference metadata failed: '+type(error).__name__) from None
                    objects[name]=dict(url=item.url,bytes=count,shape=previous['requests'][0]['shapes'][name[:-3]])
            if not page.next_page_token:break
            request.page_token=page.next_page_token
    if set(objects)!=names:raise RuntimeError('Complete Native reference outputs required')
    files['resume_reference_ticket.json']=json.dumps(dict(created_at=time.time(),objects=objects)).encode()
    plan=dict(offline_source='public',resume=dict(source_notebook=request.kernel_slug,source_version=1,
        source_sha256=previous['source_sha256'],native_result_sha256=hashlib.sha256(reference.read_bytes()).hexdigest(),
        scope='Reuse qualified reference and inference sources; new CPU observers separately qualified'))
    files['diagnostic_plan.json']=json.dumps(plan).encode()
    bootstrap=(ROOT/'benchmarks/causal_kaggle_bootstrap.py').read_text(encoding='utf-8')
    bootstrap=bootstrap.replace('/kaggle/working/causal_audit_results','/kaggle/working/cpu_audit_results')
    bootstrap=bootstrap.replace("['direct_sync','direct_p1','direct_p1_p2','offline_p1_p2']","['offline_p1_p2']")
    bootstrap=bootstrap.replace("['direct_p1','direct_p1_p2','offline_p1_p2']","['offline_p1_p2']")
    bootstrap=bootstrap.replace('continuous=dict(requests=4,warmup=1)','continuous=dict(requests=2,warmup=0)')
    bootstrap=bootstrap.replace("('continuous',4,1)","('continuous',2,0)")
    bootstrap=bootstrap.replace("'simpleeval','blake3','comfy-angle'","'simpleeval','blake3','comfy-angle','py-spy==0.4.1'")
    marker="                    if case=='offline_p1_p2':args+=['--rank-checkpoint-dir','/tmp/rank_local_tp2']"
    if marker not in bootstrap:raise RuntimeError('Bootstrap runner insertion point changed')
    bootstrap=bootstrap.replace(marker,marker+"\n                    if capture=='kineto':\n                        args[2]=str(root/'benchmarks/cpu_pipeline_audit.py')\n                        args=[sys.executable,'-u',str(root/'benchmarks/cpu_observer_supervisor.py'),'--output',str(directory),'--',*args]")
    marker="                        analysis=json.loads((directory/'causal_analysis.json').read_text())"
    bootstrap=bootstrap.replace(marker,"                        command([sys.executable,str(root/'analyze_cpu_path.py'),str(directory)],900,'cpu_path_analysis.log')\n"+marker)
    bootstrap=bootstrap.replace("        if nsys:\n            manifest['status']='nsys_matched_request'","        if False: # No additional request pass for this targeted CPU diagnostic.\n            manifest['status']='nsys_matched_request'")
    compile(bootstrap,'cpu_kaggle_bootstrap.py','exec')
    files['benchmarks/cpu_kaggle_bootstrap.py']=bootstrap.encode()
    buffer=io.BytesIO()
    with tarfile.open(fileobj=buffer,mode='w:gz') as archive:
        for name,data in files.items():
            member=tarfile.TarInfo(name);member.size=len(data);archive.addfile(member,io.BytesIO(data))
    code='''import base64,io,pathlib,runpy,tarfile
root=pathlib.Path('/kaggle/working/causal_audit_source');root.mkdir(exist_ok=True)
with tarfile.open(fileobj=io.BytesIO(base64.b64decode(PAYLOAD)),mode='r:gz') as archive:archive.extractall(root,filter='data')
runpy.run_path(str(root/'benchmarks/cpu_kaggle_bootstrap.py'),run_name='__main__')
'''.replace('PAYLOAD',repr(base64.b64encode(buffer.getvalue()).decode()))
    compile(code,'cpu_notebook','exec')
    if len(code)>1_000_000:raise RuntimeError('Kaggle notebook source limit exceeded')
    notebook['cells'][0]['source']=code.splitlines(keepends=True)
    output=ROOT/'kaggle_cpu_audit';output.mkdir(exist_ok=True)
    (output/'cpu_audit.ipynb').write_text(json.dumps(notebook))
    metadata=json.loads((ROOT/'kaggle_causal_audit/kernel-metadata.json').read_text())
    metadata.update(id='chinokiki9/minimax-h3-cpu-critical-path-audit',title='MiniMax H3 CPU Critical Path Audit',code_file='cpu_audit.ipynb')
    (output/'kernel-metadata.json').write_text(json.dumps(metadata,indent=2))
    binding={name:hashlib.sha256(data).hexdigest() for name,data in files.items() if name!='resume_reference_ticket.json'}
    (output/'source_binding.json').write_text(json.dumps(binding,indent=2))
    print('FROZEN CPU diagnostic: existing Offline+P1+P2, two cold/warm requests per matched pass; read capabilities omitted',flush=True)


if __name__=='__main__':main()
