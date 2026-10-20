"""Reuse the frozen physical source closure for targeted TE file I/O causality."""
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
    proof=json.loads((ROOT/'semantic_results/te_loading_qualification/baseline_io/result.json').read_text())
    if not proof['passed'] or not proof['all_outputs_bit_exact_with_uninstrumented']:
        raise RuntimeError('I/O observer must pass unchanged Native outputs')
    for name,sha in proof['io_source_sha256'].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=sha:raise RuntimeError('I/O observer changed after qualification')
    notebook=json.loads((ROOT/'kaggle_te_critical_path/te_critical_path.ipynb').read_text())
    source=''.join(notebook['cells'][0]['source'])
    encoded=re.search(r"b64decode\('([^']+)'\)",source).group(1)
    with tarfile.open(fileobj=io.BytesIO(base64.b64decode(encoded)),mode='r:gz') as archive:
        files={m.name:archive.extractfile(m).read() for m in archive.getmembers()}
    for name,sha in proof['source_sha256'].items():
        if name.startswith('patches/') and name in files and hashlib.sha256(files[name]).hexdigest()!=sha:
            raise RuntimeError('Physical inference closure differs from qualified local source: '+name)
    for name in ('benchmarks/te_io_pipeline.py','benchmarks/te_io_telemetry.py','benchmarks/te_io_probe.py',
                 'benchmarks/te_io_quickstart.py','benchmarks/test_te_io_telemetry.py','analyze_te_io.py','analyze_cpu_path.py'):
        files[name]=(ROOT/name).read_bytes()
    for name in ('benchmarks/te_pipeline_ablation.py','benchmarks/cpu_causal_trace.py'):
        if files[name]!=(ROOT/name).read_bytes():raise RuntimeError('Frozen observer dependency differs from qualification')
    # Refresh saved Native output read capabilities without local tensor upload.
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
                        if response.status_code!=206:raise RuntimeError('Native reference unavailable')
                        size=int(response.headers['Content-Range'].split('/')[-1])
                except requests.RequestException:raise RuntimeError('Native reference metadata transport failed; capability omitted') from None
                objects[name]=dict(url=item.url,bytes=size,shape=old['objects'][name]['shape'])
            if not page.next_page_token:break
            request.page_token=page.next_page_token
    if set(objects)!=set(old['objects']):raise RuntimeError('Incomplete saved Native references')
    files['resume_reference_ticket.json']=json.dumps(dict(created_at=time.time(),objects=objects)).encode()
    bootstrap=files['benchmarks/te_kaggle_bootstrap.py'].decode().replace('te_latency_results','te_io_results')
    marker="        ticket=root/'offline_download_ticket.json'"
    bootstrap=bootstrap.replace(marker,"""        # Isolate the small I/O matrix before downloads/installers compete
        # for disk/CPU. Only four verified shards are needed for first evidence.
        manifest['status']='early_controlled_io_probes';save()
        command([sys.executable,'-u',str(root/'benchmarks/te_io_quickstart.py'),
            '--directory','/tmp/rank_local_tp2','--output',str(out/'probe')],1200,'early_io_probe.log')
"""+marker)
    start=bootstrap.index("        manifest.update(status='te_ablations'")
    end=bootstrap.index('    except BaseException as error:',start)
    bootstrap=bootstrap[:start]+'''        manifest.update(status='te_io_pipeline',diagnostic_only=True,optimization_changes=False,
            scope='Unchanged full cold/warm requests plus controlled file-cache I/O probes',
            matrix=['baseline read telemetry','single/dual cold/hot','reused destination','sequential/willneed','bounded pressure']);save()
        command([sys.executable,str(root/'benchmarks/test_te_io_telemetry.py')],60,'telemetry_smoke.log')
        directory=out/'pipeline';directory.mkdir(exist_ok=True)
        command([sys.executable,'-u',str(root/'benchmarks/te_io_pipeline.py'),
            '--case','offline_p1_p2','--capture','nsys','--output',str(directory),
            '--checkpoints',str(root/'checkpoint_paths.json'),'--reference',str(out/'native/native_reference'),
            '--rank-checkpoint-dir','/tmp/rank_local_tp2','--requests','2','--warmup','0','--steps','4'],2400,'io_pipeline.log')
        result=json.loads((directory/'result.json').read_text())
        if not result['passed'] or len(result['requests'])!=2 or any(r['nfe']!=4 for r in result['requests']):
            raise RuntimeError('Incomplete or unequal physical requests')
        manifest['jobs'].append(dict(stage='full_pipeline_read_diagnostic',passed=True,requests=2));save()
        # No profiler or extra evaluations: real pipeline workers close before
        # applying any experimental cache hints or anonymous pressure.
        manifest['status']='io_analysis';save()
        command([sys.executable,str(root/'analyze_te_io.py'),str(out)],180,'io_analysis.log')
        manifest.update(status='complete',seconds=time.monotonic()-start);save()
'''+bootstrap[end:]
    bootstrap=bootstrap.replace('deadline=start+28800','deadline=start+6900')
    bootstrap=bootstrap.replace('budget_seconds=28800','budget_seconds=6900')
    compile(bootstrap,'te_io_kaggle_bootstrap.py','exec')
    files['benchmarks/te_io_kaggle_bootstrap.py']=bootstrap.encode()
    files['semantic_results/te_io_qualification.json']=json.dumps(proof).encode()
    buffer=io.BytesIO()
    with tarfile.open(fileobj=buffer,mode='w:gz') as archive:
        for name,data in files.items():
            member=tarfile.TarInfo(name);member.size=len(data);archive.addfile(member,io.BytesIO(data))
    source="""import base64,io,pathlib,runpy,tarfile
root=pathlib.Path('/kaggle/working/causal_audit_source');root.mkdir(exist_ok=True)
with tarfile.open(fileobj=io.BytesIO(base64.b64decode(PAYLOAD)),mode='r:gz') as archive:archive.extractall(root,filter='data')
runpy.run_path(str(root/'benchmarks/te_io_kaggle_bootstrap.py'),run_name='__main__')
""".replace('PAYLOAD',repr(base64.b64encode(buffer.getvalue()).decode()))
    compile(source,'te_io_notebook','exec')
    if len(source)>1_000_000:raise RuntimeError('Notebook source exceeds API limit')
    notebook['cells'][0]['source']=source.splitlines(keepends=True)
    out=ROOT/'kaggle_te_io_audit';out.mkdir(exist_ok=True)
    (out/'te_io_audit.ipynb').write_text(json.dumps(notebook))
    metadata=json.loads((ROOT/'kaggle_te_critical_path/kernel-metadata.json').read_text())
    metadata.update(id='chinokiki9/minimax-h3-te-file-read-causality',title='MiniMax H3 TE File Read Causality',code_file='te_io_audit.ipynb')
    (out/'kernel-metadata.json').write_text(json.dumps(metadata,indent=2))
    (out/'source_binding.json').write_text(json.dumps({name:hashlib.sha256(data).hexdigest()
        for name,data in files.items() if name!='resume_reference_ticket.json'},indent=2))
    print('FROZEN qualified TE file-read causality; read capabilities omitted',len(source),flush=True)


if __name__=='__main__':main()
