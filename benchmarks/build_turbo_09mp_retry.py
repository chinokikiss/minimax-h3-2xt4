"""User-authorized direct GPU retry, reusing CPU-qualified runtime/core bytes.

Only the benchmark canvas contract is overlaid after qualified setup. Its hash
is recorded separately; the old CPU receipt is never relabelled as new source.
"""
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

from private_reference_delivery import private_reference_prefix,mirror_reference_prefix
from t4_notebook_watchdog import supervise_worker
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from benchmarks.turbo_challenge_contract import validate

ROOT=Path(__file__).resolve().parents[1]
MODULE='benchmarks/turbo_challenge_contract.py'


def main():
    p=argparse.ArgumentParser();p.add_argument('--previous',type=Path,required=True)
    p.add_argument('--evidence',type=Path,required=True);p.add_argument('--control-commit',required=True)
    p.add_argument('--reference-directory',type=Path,required=True);p.add_argument('--reference-mirror',type=Path,required=True)
    p.add_argument('--megapixels',choices=('0.9','0.85'),default='0.9')
    p.add_argument('--failed-result',type=Path)
    p.add_argument('--submit',action='store_true');args=p.parse_args()
    assert re.fullmatch('[0-9a-f]{40}',args.control_commit)
    previous=args.previous.resolve();args.evidence.mkdir(parents=True,exist_ok=True)
    pubpath=previous/'publication.json';pub=json.loads(pubpath.read_text())
    setup_path=previous/'kaggle/cpu_final/light_setup/setup_receipt.json'
    gate_path=previous/'kaggle/cpu_final/turbo_challenge_cpu/result.json'
    setup=json.loads(setup_path.read_text());gate=json.loads(gate_path.read_text())
    assert setup['status']=='cpu_ready' and setup['review_suite']['passed'] and setup['disk_bake_cpu']['passed']
    assert gate['status']=='turbo_challenge_cpu_ready' and gate['lora_targets']==208
    assert setup['capsule_sha256']==gate['capsule_sha256']==pub['capsule_sha256']
    failed_path=args.failed_result or previous/'kaggle/gpu_final/turbo_1mp_e2e/generation/result.json'
    failed=json.loads(failed_path.read_text(encoding='utf-8'))
    assert failed['status']=='failed' and 'out of memory' in failed['error'].lower()
    width,height,tag,slug={'0.9':(1280,704,'09mp','0-9mp'),'0.85':(1216,704,'085mp','0-85mp')}[args.megapixels]
    plan=json.loads((previous/'delivery/plan.json').read_text());assert (plan['width'],plan['height'])==(1344,768)
    plan.update(width=width,height=height,name=f'turbo_{tag}_10s',
        source_note=f'Explicitly authorized {args.megapixels}MP retry after GPU OOM; no new CPU smoke, same regular Turbo inputs/settings')
    validate(plan)
    payload=subprocess.check_output(['git','show',args.control_commit+':'+MODULE],cwd=ROOT)
    assert payload.replace(b'\r\n',b'\n')==(ROOT/MODULE).read_bytes().replace(b'\r\n',b'\n')
    ast.parse(payload)
    overlay=dict(path=MODULE,original_sha256=setup['source_hashes'][MODULE],sha256=hashlib.sha256(payload).hexdigest(),
        control_commit=args.control_commit,qualified_parent_capsule_sha256=pub['capsule_sha256'],
        scope='Benchmark-only canvas/export validation; production runtime/operators/core unchanged',
        cpu_smoke_skipped_by_explicit_user_request=True,
        original_cpu_qualification_sha256=hashlib.sha256(setup_path.read_bytes()).hexdigest(),
        prior_failed_result_sha256=hashlib.sha256(failed_path.read_bytes()).hexdigest())
    plan['geometry_retry_provenance']=overlay
    (args.evidence/'request.json').write_text(json.dumps(plan,indent=2))
    job=args.evidence/'gpu_job'
    subprocess.run([sys.executable,str(ROOT/'benchmarks/build_kaggle_light_job.py'),'--mode','e2e',
        '--publication',str(pubpath),'--qualification',str(setup_path),'--output',str(job),'--setup-budget','600'],check=True)
    nbpath=job/'experiment.ipynb';nb=json.loads(nbpath.read_text());code=''.join(nb['cells'][0]['source'])
    marker="assert receipt['status']=='ready_for_device_gate'\n";assert code.count(marker)==1
    code=code[:code.index(marker)+len(marker)]
    refs=plan['reference_images'];mirrors=json.loads(args.reference_mirror.read_text())['files']
    private=[r for r in refs if r['file'] not in mirrors];mirrored=[r for r in refs if r['file'] in mirrors]
    for r in mirrored:assert (mirrors[r['file']]['sha256'],mirrors[r['file']]['bytes'])==(r['sha256'],r['bytes'])
    destination='/kaggle/temp/h3_light_source/bench_assets/l1_current'
    code+=private_reference_prefix(private,args.reference_directory,destination)
    code+=mirror_reference_prefix(mirrored,{r['file']:mirrors[r['file']]['url'] for r in mirrored},destination)
    execution="""import base64,hashlib,json,os,pathlib,sys
env=dict(os.environ);env.update(json.loads(pathlib.Path('/kaggle/working/light_setup/execution_env.json').read_text()))
source=pathlib.Path('/kaggle/temp/h3_light_source');out=pathlib.Path('/kaggle/working/turbo_09mp_e2e');out.mkdir()
binding=OVERLAY
manifest=json.loads(pathlib.Path('/kaggle/temp/h3_light_distribution/delivery_manifest.json').read_text())
assert manifest['capsule_sha256']==binding['qualified_parent_capsule_sha256']
for name,expected in manifest['source_hashes'].items():
 assert hashlib.sha256((source/name).read_bytes()).hexdigest()==expected,'Qualified parent source changed: '+name
target=source/binding['path'];assert hashlib.sha256(target.read_bytes()).hexdigest()==binding['original_sha256']
payload=base64.b85decode(PAYLOAD);assert hashlib.sha256(payload).hexdigest()==binding['sha256']
target.write_bytes(payload)
effective=dict(manifest['source_hashes']);effective[binding['path']]=binding['sha256']
(out/'source_overlay.json').write_text(json.dumps(dict(binding=binding,effective_source_hashes=effective),indent=2))
(out/'retry_request.json').write_text(REQUEST)
argv=[sys.executable,'-u',str(source/'benchmarks/run_h3_l1_turbo_batch.py'),'--source',str(source),
 '--output',str(out),'--plan',str(out/'retry_request.json'),'--scratch','/tmp/h3_turbo_09mp_10s_20261029','--e2e-timeout','5400']
os.execve(sys.executable,argv,env)
""".replace('turbo_09mp',f'turbo_{tag}').replace('OVERLAY',repr(overlay)).replace('PAYLOAD',repr(base64.b85encode(payload).decode())).replace('REQUEST',repr(json.dumps(plan)))
    code+=supervise_worker(execution,budget=7200,output=f'/kaggle/working/turbo_{tag}_supervision')
    code+=f"assert json.loads(pathlib.Path('/kaggle/working/turbo_{tag}_e2e/batch_results.json').read_text())['status']=='complete'\n"
    ast.parse(code);assert not re.search(r'(?:hf_[A-Za-z0-9]{25,}|gh[pousr]_[A-Za-z0-9]{30,})',code)
    nb['cells'][0]['source']=code.splitlines(keepends=True);nbpath.write_text(json.dumps(nb),encoding='utf-8')
    assert nbpath.stat().st_size<9_900_000
    meta=json.loads((job/'kernel-metadata.json').read_text())
    meta.update(id=f'pamcks/h3-turbo-{slug}-10s-t4-20261029',title=f'H3 Turbo {args.megapixels}MP 10s T4 20261029')
    assert meta['is_private'] and meta['dataset_sources']==meta['kernel_sources']==[]
    (job/'kernel-metadata.json').write_text(json.dumps(meta,indent=2))
    binding=json.loads((job/'source_binding.json').read_text())
    binding.update(full_pipeline_authorized=True,ptd=False,online_lora=False,request_contract=validate(plan),
        benchmark_overlay=overlay,new_cpu_smoke_run=False,credentials=False,
        prior_input_qualification_sha256=hashlib.sha256(gate_path.read_bytes()).hexdigest())
    (job/'source_binding.json').write_text(json.dumps(binding,indent=2))
    if not args.submit:return
    state=dict(kernel=meta['id'],status='submitting',submitted_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        source_binding=binding,run_timeout_seconds=8100,earliest_final_collection_seconds=8100,
        final_collection_count=0,result_relative=f'turbo_{tag}_e2e/batch_results.json',
        notebook_sha256=hashlib.sha256(nbpath.read_bytes()).hexdigest())
    receipt=args.evidence/'gpu_submission.json'
    with receipt.open('x') as f:json.dump(state,f,indent=2)
    try:
        process=subprocess.run([sys.executable,'-m','kaggle','kernels','push','-p',str(job),'-t','8100',
            '--accelerator','NvidiaTeslaT4'],capture_output=True,text=True,timeout=180)
        output=process.stdout+process.stderr
        state.update(status='accepted' if process.returncode==0 else 'acceptance_uncertain',returncode=process.returncode)
        match=re.search(r'Kernel version (\d+)',output)
        if match:state['version']=int(match.group(1))
        match=re.search(r'https://www\.kaggle\.com/code/([\w-]+/[\w-]+)',output)
        if match:state.update(requested_kernel=state['kernel'],kernel=match.group(1),url=match.group(0))
    except subprocess.TimeoutExpired:
        output='Acknowledgement timeout; preserve claim, do not retry automatically.'
        state.update(status='acceptance_uncertain',returncode=-1)
    receipt.write_text(json.dumps(state,indent=2));(args.evidence/'gpu_submission_cli.log').write_text(output,encoding='utf-8')
    print(output[-1500:])


if __name__=='__main__':main()
