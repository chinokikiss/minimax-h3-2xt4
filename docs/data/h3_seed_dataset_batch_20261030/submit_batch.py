"""Validate prepared bytes and submit exactly one GPU plus two CPU jobs."""
import argparse
import ast
import base64
import datetime as dt
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT=Path(__file__).resolve().parent
JOBS=[dict(kind='gpu',root=ROOT/'gpu/kaggle',job='gpu_job',timeout=8100,result='turbo_085mp_seed20261021_e2e/batch_results.json'),
    dict(kind='dataset_read',root=ROOT/'cpu_read',job='job',timeout=1260,result='dataset_read/result.json'),
    dict(kind='dataset_upload',root=ROOT/'cpu_upload',job='job',timeout=840,result='dataset_upload/result.json')]


def validate():
    for spec in JOBS:
        job=spec['root']/spec['job'];meta=json.loads((job/'kernel-metadata.json').read_text())
        binding=json.loads((job/'source_binding.json').read_text())
        if spec['kind']=='gpu':
            nb=json.loads((job/'experiment.ipynb').read_text());script=''.join(nb['cells'][0]['source'])
            plan=json.loads((spec['root']/'request.json').read_text())
            prior=json.loads(Path('D:/attention_eval/h3_turbo_085mp_10s_20261029/retry2/kaggle/request.json').read_text())
            changed={k for k in prior.keys()|plan.keys() if prior.get(k)!=plan.get(k)}
            assert changed=={'name','seed','source_note','geometry_retry_provenance'},changed
            assert plan['seed']==20261021 and (plan['width'],plan['height'])==(1216,704)
            assert meta['enable_gpu'] and not meta['dataset_sources'] and not meta['kernel_sources']
            tree=ast.parse(script);worker=None
            for node in ast.walk(tree):
                if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr=='b64decode' and isinstance(node.args[0],ast.Constant):
                    decoded=base64.b64decode(node.args[0].value).decode()
                    if 'target.write_bytes(payload)' in decoded:worker=decoded
            assert worker and '/kaggle/working/turbo_085mp_seed20261021_e2e' in worker
            tree=ast.parse(worker)
            payloads=[base64.b85decode(n.args[0].value) for n in ast.walk(tree)
                if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='b85decode' and isinstance(n.args[0],ast.Constant)]
            overlay=binding['benchmark_overlay'];assert len(payloads)==1
            assert hashlib.sha256(payloads[0]).hexdigest()==overlay['sha256']
            additions=next(ast.literal_eval(n.iter) for n in ast.walk(tree) if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='item')
            assert len(additions)==1
            for item in additions:assert hashlib.sha256(base64.b85decode(item['payload_b85'])).hexdigest()==item['sha256']
            baseline=json.loads(Path('D:/attention_eval/h3_turbo_085mp_10s_20261029/retry2/kaggle/gpu_final/turbo_085mp_e2e/generation/result.json').read_text())
            binding['seed_comparison']=dict(baseline_seed=20261020,seed=20261021,
                baseline_inputs=baseline['inputs'],changed_request_fields=sorted(changed),
                acceptance='Seed sensitivity only; neither TE nor VAE attribution')
            (job/'source_binding.json').write_text(json.dumps(binding,indent=2))
        else:
            script=(job/'experiment.py').read_text();ast.parse(script)
            assert not meta['enable_gpu'] and not meta['enable_tpu']
            assert len(meta['dataset_sources'])==(6 if spec['kind']=='dataset_read' else 0)
            assert not meta['kernel_sources']
        assert meta['is_private']
        assert not re.search(r'(?:hf_[A-Za-z0-9]{25,}|gh[pousr]_[A-Za-z0-9]{30,})',script)
        spec['metadata']=meta;spec['binding']=binding
    proof=dict(status='prepared_and_static_verified',jobs=[dict(kind=s['kind'],kernel=s['metadata']['id']) for s in JOBS],
        gpu_instances=1,cpu_instances=2,seed=20261021,read_dataset_counts=[1,2,4,6],reader_counts=[1,2,4,8,16,32,64],
        no_embedded_credentials=True,no_new_h3_cpu_smoke_job=True,no_kaggle_polling=True)
    (ROOT/'preparation_verification.json').write_text(json.dumps(proof,indent=2))


def submit(spec):
    job=spec['root']/spec['job'];meta=spec['metadata'];filename=meta['code_file']
    receipt=spec['root']/'submission.json'
    state=dict(kernel=meta['id'],status='submitting',submitted_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        source_binding=spec['binding'],run_timeout_seconds=spec['timeout'],earliest_final_collection_seconds=spec['timeout'],
        final_collection_count=0,result_relative=spec['result'],experiment_kind=spec['kind'],
        submitted_code_sha256=hashlib.sha256((job/filename).read_bytes()).hexdigest())
    with receipt.open('x') as stream:json.dump(state,stream,indent=2)
    argv=[sys.executable,'-m','kaggle','kernels','push','-p',str(job),'-t',str(spec['timeout'])]
    if spec['kind']=='gpu':argv+=['--accelerator','NvidiaTeslaT4']
    try:
        proc=subprocess.run(argv,capture_output=True,text=True,timeout=180)
        output=proc.stdout+proc.stderr
        state.update(status='accepted' if proc.returncode==0 else 'acceptance_uncertain',returncode=proc.returncode)
        match=re.search(r'Kernel version (\d+)',output)
        if match:state['version']=int(match[1])
        match=re.search(r'https://www\.kaggle\.com/code/([\w-]+/[\w-]+)',output)
        if match:state.update(requested_kernel=state['kernel'],kernel=match[1],url=match[0])
    except subprocess.TimeoutExpired:
        state.update(status='acceptance_uncertain',returncode=-1);output='Acknowledgement timeout; no automatic retry.'
    receipt.write_text(json.dumps(state,indent=2));(spec['root']/'submission_cli.log').write_text(output,encoding='utf-8')
    print(json.dumps({k:state.get(k) for k in ('kernel','status','version','url')}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--submit',action='store_true');a=p.parse_args()
    validate()
    if a.submit:
        for job in JOBS:submit(job)
    else:print('All three jobs prepared; no submission made')
