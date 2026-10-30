"""One private T4 submission using the already qualified H3 runtime/core capsule."""
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

from private_reference_delivery import private_reference_prefix, mirror_reference_prefix
from t4_notebook_watchdog import supervise_worker

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ['benchmarks/te_direct_io.py', 'benchmarks/test_te_direct_io.py',
           'benchmarks/run_te_direct_io.py']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--qualified', type=Path, required=True)
    parser.add_argument('--evidence', type=Path, required=True)
    parser.add_argument('--reference-directory', type=Path, required=True)
    parser.add_argument('--reference-mirror', type=Path, required=True)
    parser.add_argument('--submit', action='store_true')
    args = parser.parse_args()
    args.evidence.mkdir(parents=True, exist_ok=True)
    job = args.evidence/'gpu_job'
    publication = args.qualified/'publication.json'
    cpu = args.qualified/'kaggle/cpu_final/light_setup/setup_receipt.json'
    qualification = json.loads(cpu.read_text())
    pub = json.loads(publication.read_text())
    assert qualification['status'] == 'cpu_ready'
    assert qualification['capsule_sha256'] == pub['capsule_sha256'] == '21abe7dd91f284eed9626bb018c6abf66adf1c066b2f76662d72ce53d46d41a1'
    assert qualification['core_pins']['comfy-kitchen'] == '0.2.37'
    assert qualification['core_pins']['tilelang'] == '0.1.13'
    subprocess.run([sys.executable, str(ROOT/'benchmarks/build_kaggle_light_job.py'),
        '--mode','e2e','--publication',str(publication),'--qualification',str(cpu),
        '--setup-budget','600','--output',str(job)], check=True)
    notebook_path = job/'experiment.ipynb'
    notebook = json.loads(notebook_path.read_text())
    code = ''.join(notebook['cells'][0]['source'])
    marker = "assert receipt['status']=='ready_for_device_gate'\n"
    assert code.count(marker) == 1
    code = code[:code.index(marker)+len(marker)]
    plan = json.loads((args.qualified/'delivery/plan.json').read_text())
    mirrors = json.loads(args.reference_mirror.read_text())['files']
    private = [r for r in plan['reference_images'] if r['file'] not in mirrors]
    public = [r for r in plan['reference_images'] if r['file'] in mirrors]
    destination = '/kaggle/temp/h3_light_source/bench_assets/l1_current'
    code += private_reference_prefix(private, args.reference_directory, destination)
    code += mirror_reference_prefix(public, {r['file']:mirrors[r['file']]['url'] for r in public}, destination)
    files = {name:(ROOT/name).read_bytes() for name in SOURCES}
    for raw in files.values():
        ast.parse(raw)
    hashes = {name:hashlib.sha256(raw).hexdigest() for name,raw in files.items()}
    commit = subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT, text=True).strip()
    for name, raw in files.items():
        assert subprocess.check_output(['git','show',commit+':'+name], cwd=ROOT) == raw.replace(b'\r\n', b'\n'), name
    with (args.evidence/'adapter_cpu_tests.log').open('w') as log:
        check = subprocess.run([sys.executable,'-m','unittest','benchmarks.test_te_direct_io','-v'],
                               cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    assert check.returncode == 0
    execution = ('import base64,hashlib,json,os,pathlib,sys\n'
        + 'FILES='+repr({name:base64.b64encode(raw).decode() for name,raw in files.items()})+'\n'
        + 'HASHES='+repr(hashes)+'\n'
        + 'CAPSULE='+repr(pub['capsule_sha256'])+'\n'
        + '''
source=pathlib.Path('/kaggle/temp/h3_light_source')
expected=json.loads(pathlib.Path('/kaggle/temp/h3_light_distribution/delivery_manifest.json').read_text())
assert expected['capsule_sha256']==CAPSULE
for name,digest in expected['source_hashes'].items():
 assert hashlib.sha256((source/name).read_bytes()).hexdigest()==digest,name
for name,encoded in FILES.items():
 raw=base64.b64decode(encoded);assert hashlib.sha256(raw).hexdigest()==HASHES[name]
 (source/name).write_bytes(raw)
env=dict(os.environ);env.update(json.loads(pathlib.Path('/kaggle/working/light_setup/execution_env.json').read_text()))
argv=[sys.executable,'-u',str(source/'benchmarks/run_te_direct_io.py'),'--source',str(source),
 '--output','/kaggle/working/te_direct_io','--scratch','/tmp/h3_te_direct_io_20261030']
os.chdir(source)
os.execve(sys.executable,argv,env)
''')
    code += supervise_worker(execution, budget=4200, output='/kaggle/working/te_direct_supervision')
    code += "\nassert json.loads(pathlib.Path('/kaggle/working/te_direct_io/result.json').read_text())['status']=='complete'\n"
    ast.parse(code)
    assert not re.search(r'(?:hf_[A-Za-z0-9]{25,}|gh[pousr]_[A-Za-z0-9]{30,}|KGAT_[A-Za-z0-9_-]{20,})',code)
    notebook['cells'][0]['source'] = code.splitlines(keepends=True)
    notebook_path.write_text(json.dumps(notebook), encoding='utf-8')
    metadata = json.loads((job/'kernel-metadata.json').read_text())
    metadata.update(id='pamcks/h3-te-direct-io-t4-20261030', title='H3 TE Direct IO T4 20261030')
    assert metadata['is_private'] and metadata['enable_gpu'] and metadata['dataset_sources'] == metadata['kernel_sources'] == []
    (job/'kernel-metadata.json').write_text(json.dumps(metadata,indent=2))
    binding = dict(source_commit=commit, benchmark_sources=hashes, publication=pub,
        qualification_sha256=hashlib.sha256(cpu.read_bytes()).hexdigest(),
        previous_runtime_core_qualification_reused=True, new_cpu_instance=False,
        frozen_runtime_and_operators_unchanged=True, benchmark_adaptation='TE-only unused DiT reader and aligned weight I/O observers',
        te_verification=True, hash_profile=['download_file_sha256','inference_file_sha256','tensor_sha256'],
        actual_reference_workload=dict(width=1216,height=704,prompt_sha256=plan['prompt_sha256'],
            reference_sha256=[row['sha256'] for row in plan['reference_images']]),
        te_only=True, dit=False, vae=False, lora=False, full_pipeline=False,
        benchmark_order=['warmup_buffered','buffered','direct','direct','buffered'],
        global_drop_caches=False, buffered_fallback_for_direct=False, profile_passes_separate=True,
        setup_budget_seconds=600, execution_budget_seconds=4200, provider_limit_seconds=5100,
        private_image_bytes_in_private_notebook_only=True, embedded_credentials=False,
        notebook_sha256=hashlib.sha256(notebook_path.read_bytes()).hexdigest())
    (job/'source_binding.json').write_text(json.dumps(binding,indent=2))
    print('Built:',job,flush=True)
    if not args.submit:
        return
    receipt = args.evidence/'gpu_submission.json'
    state = dict(kernel=metadata['id'],status='submitting',submitted_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        source_binding=binding, run_timeout_seconds=5100, earliest_final_collection_seconds=5100,
        final_collection_count=0, experiment_kind='te_direct_io', result_relative='te_direct_io/result.json')
    with receipt.open('x') as stream:
        json.dump(state,stream,indent=2)
    try:
        process = subprocess.run([sys.executable,'-m','kaggle','kernels','push','-p',str(job),
            '--accelerator','NvidiaTeslaT4','-t','5100'],capture_output=True,text=True,timeout=180)
        output = process.stdout+process.stderr
        state.update(status='accepted' if process.returncode == 0 else 'acceptance_uncertain',returncode=process.returncode)
        version = re.search(r'Kernel version (\d+)',output)
        if version:
            state['version'] = int(version.group(1))
        url = re.search(r'https://www\.kaggle\.com/code/[\w-]+/[\w-]+',output)
        if url:
            state['url'] = url.group(0)
    except subprocess.TimeoutExpired:
        output = 'Submission acknowledgement timed out; preserve claim, no automatic retry.'
        state.update(status='acceptance_uncertain',returncode=-1)
    receipt.write_text(json.dumps(state,indent=2))
    (args.evidence/'gpu_submission_cli.log').write_text(output,encoding='utf-8')
    print(output[-1500:],flush=True)


if __name__ == '__main__':
    main()
