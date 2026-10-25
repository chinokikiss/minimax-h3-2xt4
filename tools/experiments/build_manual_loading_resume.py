"""Make a credential-free cell for the user's existing manual Kaggle session."""
import base64
import hashlib
import io
import json
from pathlib import Path
import tarfile

ROOT=Path(__file__).resolve().parents[2]


def main():
    archived=json.loads((ROOT/'semantic_results/manual_native_timeout/native_result.json').read_text())
    guards=dict(archived['source_sha256'])
    proof_paths=[
        ROOT/'semantic_results/loading_native_placement/result.json',
        ROOT/'semantic_results/rank_local_loader_final.json',
        ROOT/'semantic_results/loading_pipeline_direct/sync_result.json',
        ROOT/'semantic_results/loading_pipeline_direct/adaptive_result.json',
        ROOT/'semantic_results/loading_pipeline_offline/adaptive_result.json',
        ROOT/'semantic_results/loading_sampler_2/result.json',
        *[ROOT/'semantic_results/loading_wrapper_p012_v2'/kind/(strategy+'_result.json')
          for kind,strategy in [('sync','sync'),('adaptive','adaptive'),
                               ('offline','adaptive'),('adaptive_p2_off','adaptive')]]]
    proofs=[]
    for path in proof_paths:
        r=json.loads(path.read_text())
        if not r['passed']:raise ValueError('Failed qualification: '+str(path))
        for name,sha in r['source_sha256'].items():
            name=name.replace('\\','/')
            if name in guards and guards[name]!=sha:
                raise ValueError('Qualification/source disagreement: '+name)
            guards[name]=sha
        proofs.append(dict(path=path.relative_to(ROOT).as_posix(),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    for name,sha in guards.items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=sha:
            raise ValueError('Numerical source changed: '+name)
    result_sources=dict(archived['source_sha256'])
    result_sources['benchmarks/run_loading_audit.py']=hashlib.sha256((ROOT/'benchmarks/run_loading_audit.py').read_bytes()).hexdigest()
    guard=dict(passed=True,source_sha256=guards,result_source_sha256=result_sources,
        comfy_commit='d49e888586dd8ae012c0667b33466b815fee07f7',proofs=proofs,
        model_operator_changes=False,local_simulation_is_performance_evidence=False)
    guard_path=ROOT/'semantic_results/manual_resume_qualification.json'
    guard_path.write_text(json.dumps(guard,indent=2))
    names=['benchmarks/monitor_pipeline_job.py','benchmarks/observe_loading_audit.py',
           'benchmarks/resume_loading_audit.py','tools/experiments/analyze_loading_audit.py',
           'tools/experiments/analyze_pipeline_audit.py','semantic_results/manual_resume_qualification.json']
    buffer=io.BytesIO()
    with tarfile.open(fileobj=buffer,mode='w:gz') as tar:
        for name in names:
            p=ROOT/name
            if p.suffix=='.py':compile(p.read_text(),name,'exec')
            tar.add(p,arcname=name)
    payload=buffer.getvalue();sha=hashlib.sha256(payload).hexdigest()
    cell='''# Paste this cell into the SAME Kaggle session after the failed audit.
# It needs the existing /tmp checkpoints; it never downloads or installs.
import base64,hashlib,io,json,pathlib,signal,subprocess,sys,tarfile
root=pathlib.Path('/kaggle/working/pipeline_audit_source')
if not (root/'checkpoint_paths.json').is_file():
    raise RuntimeError('Original session files are missing; this is a resume-only cell.')
payload=base64.b64decode(PAYLOAD)
assert hashlib.sha256(payload).hexdigest()==EXPECTED_SHA
with tarfile.open(fileobj=io.BytesIO(payload),mode='r:gz') as tar:
    members=tar.getmembers()
    if set(m.name for m in members)!=set(EXPECTED_NAMES) or not all(m.isfile() for m in members):
        raise RuntimeError('Unexpected diagnostic archive members')
    guard=json.load(tar.extractfile('semantic_results/manual_resume_qualification.json'))
    for name,sha in guard['source_sha256'].items():
        if hashlib.sha256((root/name).read_bytes()).hexdigest()!=sha:
            raise RuntimeError('Qualified model source changed: '+name)
    tar.extractall(root,filter='data')
proc=subprocess.Popen([sys.executable,'-u',str(root/'benchmarks/resume_loading_audit.py')],
    stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,start_new_session=True)
try:
    for line in proc.stdout:print(line.rstrip(),flush=True)
    code=proc.wait()
    if code:raise RuntimeError('Resume stopped; inspect its log and summary_evidence.zip.')
except KeyboardInterrupt:
    proc.send_signal(signal.SIGINT)
    try:proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        print('Resume cleanup still running; do not start another GPU audit.',flush=True)
    raise
'''.replace('PAYLOAD',repr(base64.b64encode(payload).decode())).replace('EXPECTED_SHA',repr(sha)).replace('EXPECTED_NAMES',repr(names))
    compile(cell,'resume_cell','exec')
    folder=ROOT/'kaggle_manual_resume';folder.mkdir(exist_ok=True)
    (folder/'resume_cell.py').write_text(cell)
    notebook=dict(nbformat=4,nbformat_minor=5,metadata=dict(kernelspec=dict(display_name='Python 3',language='python',name='python3')),
        cells=[dict(id='resume-existing-session',cell_type='code',metadata={},execution_count=None,outputs=[],source=cell.splitlines(keepends=True))])
    (folder/'resume_existing_session.ipynb').write_text(json.dumps(notebook,indent=2))
    (folder/'build_manifest.json').write_text(json.dumps(dict(passed=True,payload_sha256=sha,files=names,
        model_operator_changes=False,embedded_credentials=False,reconversion=False,downloads=False,
        existing_session_required=True,cell_bytes=len(cell.encode()),proofs=proofs),indent=2))
    print('READY',folder/'resume_cell.py','bytes',len(cell.encode()))


if __name__=='__main__':main()
