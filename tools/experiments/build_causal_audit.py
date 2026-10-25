"""Freeze locally qualified observers and unchanged operators into a private job."""
import base64
import argparse
import hashlib
import io
import json
from pathlib import Path
import tarfile
import time

ROOT=Path(__file__).resolve().parents[2]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--offline-source',choices=('public','ticket','secret','regenerate'),default='public')
    parser.add_argument('--resume-from',type=Path,help='Recovered version-1 evidence directory')
    options=parser.parse_args()
    proof=ROOT/'semantic_results/causal_qualification/result.json'
    result=json.loads(proof.read_text())
    if not result['passed']:raise RuntimeError('Diagnostic local qualification must pass')
    for name,sha in result['source_sha256'].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=sha:
            raise RuntimeError('Changed after diagnostic qualification: '+name)
    vae=json.loads((ROOT/'semantic_results/parallel_vae_pipeline.json').read_text())
    if not vae['passed']:raise RuntimeError('Default VAE qualification missing')
    for name,sha in vae['source_sha256'].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=sha:
            raise RuntimeError('Default pipeline changed after qualification: '+name)
    files=list((ROOT/'patches').glob('*.py'))
    files += [ROOT/'benchmarks'/name for name in (
        'causal_trace.py','causal_pipeline_audit.py','causal_kaggle_bootstrap.py',
        'pipeline_residency_audit.py','retest.py','full_models.py','fixture_inputs.py',
        'logical_dual_gpu.py','qualify_comfy_sampler.py','ref2va_workloads.py',
        'native_resident_reference.py','restore_offline_bundle.py','offline_checkpoint_store.py','prepare_causal_offline.py',
        'restore_causal_reference.py')]
    files += [ROOT/'tools/experiments/analyze_causal_audit.py',proof,ROOT/'semantic_results/parallel_vae_pipeline.json',
        ROOT/'semantic_results/pipeline_checkpoint_identity.json',ROOT/'semantic_results/offline_conversion_plan.json',
        ROOT/'semantic_results/causal_offline_helper.json']
    if not json.loads((ROOT/'semantic_results/causal_offline_helper.json').read_text())['passed']:
        raise RuntimeError('Offline regeneration helper Native comparison must pass')
    if options.offline_source in ('public','secret'):
        files+=list((ROOT/'semantic_results/kaggle_hf_publication_v6').glob('*.json'))
    output=ROOT/'kaggle_causal_audit';output.mkdir(exist_ok=True)
    resume=None
    if options.resume_from:
        evidence=options.resume_from.resolve();runs=[]
        for relative in ('short/direct_p1_p2/timing','short/direct_p1_p2/kineto',
                         'continuous/direct_sync/timing','continuous/direct_sync/kineto'):
            path=evidence/relative/'result.json';run=json.loads(path.read_text())
            expected=2 if relative.startswith('short/') else 4
            if not (run['passed'] and run['physical'] and len(run['requests'])==expected
                    and len(run['warmup'])==(0 if expected==2 else 1)
                    and all(r['nfe']==4 for r in run['requests'])):
                raise RuntimeError('Incomplete recovery evidence: '+relative)
            if not all(all(record[name]['passed'] for name in ('conditioning','video','audio','pixels','waveform'))
                       for record in run['correctness']) or len(run['correctness'])!=expected:
                raise RuntimeError('Recovered Native correctness records missing or failed')
            if runs and run['workload']!=runs[0]['workload']:
                raise RuntimeError('Recovered workloads differ')
            for name,sha in run['source_sha256'].items():
                if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=sha:
                    raise RuntimeError('Inference/capture sources changed since recovered run: '+name)
            if run['capture']=='kineto':
                analysis=json.loads((path.parent/'causal_analysis.json').read_text())
                if analysis['analysis_source_sha256']!=hashlib.sha256((ROOT/'tools/experiments/analyze_causal_audit.py').read_bytes()).hexdigest():
                    raise RuntimeError('Analyze recovered traces with the current analyzer before resuming')
                if not analysis['block_loading'] or not analysis['collective_calls'] or any(
                        r['ranks'][str(rank)]['kernel_count']==0 for r in analysis['requests'] for rank in (0,1)):
                    raise RuntimeError('Recovered trace quality gate failed')
            runs.append(dict(relative=relative,result_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                case=run['case'],capture=run['capture'],workload=run['workload'],
                latency_ms=[r['latency_ms'] for r in run['requests']],correctness=run['correctness']))
        native=json.loads((evidence/'native/native_result.json').read_text())
        if not (native['passed'] and native['physical'] and native['layers']==50 and native['steps']==4):
            raise RuntimeError('Saved physical Native reference proof missing')
        for name,sha in native['source_sha256'].items():
            if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=sha:
                raise RuntimeError('Native source changed since reference: '+name)
        # Only read capabilities are issued locally. Reference tensors travel
        # directly from the original Kaggle output to the continuation instance.
        from kaggle.api.kaggle_api_extended import KaggleApi,ApiListKernelSessionOutputRequest
        api=KaggleApi();api.authenticate();request=ApiListKernelSessionOutputRequest()
        request.user_name='chinokiki9';request.kernel_slug='minimax-h3-causal-pipeline-audit'
        objects={};expected_names={n+'.pt' for n in ('conditioning','video','audio','pixels','waveform')}
        with api.build_kaggle_client() as client:
            while True:
                page=client.kernels.kernels_api_client.list_kernel_session_output(request)
                for item in page.files or []:
                    if item.file_name.startswith('causal_audit_results/native/native_reference/'):
                        name=Path(item.file_name).name
                        if name in expected_names:
                            import requests
                            try:
                                with requests.get(item.url,headers={'Range':'bytes=0-0'},stream=True,timeout=(10,45)) as response:
                                    if response.status_code!=206:raise RuntimeError('Reference output range probe failed')
                                    size=int(response.headers['Content-Range'].split('/')[-1])
                            except requests.RequestException as error:
                                raise RuntimeError('Reference output metadata unavailable: '+type(error).__name__) from None
                            objects[name]=dict(url=item.url,bytes=size,
                                shape=run['requests'][0]['shapes'][name.removesuffix('.pt')])
                if not page.next_page_token:break
                request.page_token=page.next_page_token
        if set(objects)!=expected_names:raise RuntimeError('Saved Native tensor outputs unavailable')
        ticket=output/'resume_reference_ticket.json'
        ticket.write_text(json.dumps(dict(created_at=time.time(),objects=objects)))
        resume=dict(source_notebook='chinokiki9/minimax-h3-causal-pipeline-audit',source_version=1,
            completed=runs,source_sha256=run['source_sha256'],
            native_result_sha256=hashlib.sha256((evidence/'native/native_result.json').read_bytes()).hexdigest(),
            native_reference='Saved physical Native tensors; identical source/checkpoint/runtime bindings',
            comparison_limitation='Remaining cases use a new T4 allocation; cross-allocation variability is not isolated')
    buffer=io.BytesIO()
    with tarfile.open(fileobj=buffer,mode='w:gz') as archive:
        for file in files:archive.add(file,arcname=file.relative_to(ROOT).as_posix())
        if options.offline_source=='ticket':
            archive.add(ROOT/'kaggle_direct_loading_audit/offline_download_ticket.json',arcname='offline_download_ticket.json')
        if resume:archive.add(ticket,arcname='resume_reference_ticket.json')
        data=json.dumps(dict(offline_source=options.offline_source,resume=resume)).encode()
        member=tarfile.TarInfo('diagnostic_plan.json');member.size=len(data);archive.addfile(member,io.BytesIO(data))
    code='''import base64,io,pathlib,runpy,tarfile
root=pathlib.Path('/kaggle/working/causal_audit_source');root.mkdir(exist_ok=True)
with tarfile.open(fileobj=io.BytesIO(base64.b64decode(PAYLOAD)),mode='r:gz') as archive:archive.extractall(root,filter='data')
runpy.run_path(str(root/'benchmarks/causal_kaggle_bootstrap.py'),run_name='__main__')
'''.replace('PAYLOAD',repr(base64.b64encode(buffer.getvalue()).decode()))
    compile(code,'causal_audit_notebook','exec')
    if len(code)>1_000_000:raise ValueError('Notebook code exceeds Kaggle source limit')
    notebook=dict(cells=[dict(cell_type='code',execution_count=None,metadata={},outputs=[],source=code.splitlines(keepends=True))],
        metadata=json.loads((ROOT/'kaggle_comfy_sampler/sampler_gate.ipynb').read_text())['metadata'],nbformat=4,nbformat_minor=5)
    (output/'causal_audit.ipynb').write_text(json.dumps(notebook))
    metadata=json.loads((ROOT/'kaggle_comfy_sampler/kernel-metadata.json').read_text())
    metadata.update(id='chinokiki9/minimax-h3-causal-pipeline-audit',title='MiniMax H3 Causal Pipeline Audit',code_file='causal_audit.ipynb',kernel_sources=[])
    if resume:metadata.update(id='chinokiki9/minimax-h3-causal-pipeline-resume',title='MiniMax H3 Causal Pipeline Resume')
    (output/'kernel-metadata.json').write_text(json.dumps(metadata,indent=2))
    binding={file.relative_to(ROOT).as_posix():hashlib.sha256(file.read_bytes()).hexdigest() for file in files}
    (output/'source_binding.json').write_text(json.dumps(binding,indent=2))
    print('FROZEN causal diagnostic',len(files),'source files; private download capabilities omitted from output')


if __name__=='__main__':main()
