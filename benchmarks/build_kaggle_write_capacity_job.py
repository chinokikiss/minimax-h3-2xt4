"""One zero-mount, stdlib-only CPU disk probe; no package or GPU setup."""
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

from t4_notebook_watchdog import supervise_worker


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--commit', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--submit', action='store_true')
    args = parser.parse_args()
    if not re.fullmatch('[0-9a-f]{40}', args.commit):
        parser.error('An immutable full source commit is required')
    root = Path(__file__).resolve().parents[1]
    names = ('kaggle_write_capacity.py', 'test_kaggle_write_capacity.py')
    payload, hashes = {}, {}
    for name in names:
        data = (root/'benchmarks'/name).read_bytes()
        committed = subprocess.check_output(['git', 'show', args.commit+':benchmarks/'+name], cwd=root)
        if committed != data:
            raise ValueError('Probe source differs from the immutable source commit: '+name)
        hashes[name] = hashlib.sha256(data).hexdigest()
        payload[name] = base64.b64encode(data).decode()
    binding = dict(source_commit=args.commit, source_sha256=hashes,
        source_urls={n:f'https://raw.githubusercontent.com/chinokikiss/minimax-h3-2xt4/{args.commit}/benchmarks/{n}' for n in names},
        embedded_sources=True, cpu_only=True, stdlib_only=True, installs=False,
        h3_runtime_loaded=False, native_packages_unchanged=True, credentials=False,
        dataset_sources=[], kernel_sources=[], per_disk_byte_cap=512*2**30,
        per_disk_seconds=1500, overall_probe_seconds=3000, watchdog_seconds=3150,
        report_headroom_bytes=512*2**20)
    body = '''import base64,hashlib,json,os,pathlib,subprocess,sys
OUT=pathlib.Path('/kaggle/working/disk_capacity');OUT.mkdir(parents=True,exist_ok=True)
BINDING=SOURCE_BINDING
FILES=EMBEDDED_FILES
(OUT/'source_binding.json').write_text(json.dumps(BINDING,indent=2))
assert not list(pathlib.Path('/dev').glob('nvidia[0-9]*')),'CPU-only probe requires no GPU allocation'
source=OUT/'source';(source/'benchmarks').mkdir(parents=True,exist_ok=True)
(source/'benchmarks/__init__.py').write_text('')
for name,encoded in FILES.items():
    data=base64.b64decode(encoded)
    assert hashlib.sha256(data).hexdigest()==BINDING['source_sha256'][name]
    (source/'benchmarks'/name).write_bytes(data)
env=dict(os.environ,PYTHONPATH=str(source),CUDA_VISIBLE_DEVICES='')
with (OUT/'cpu_tests.log').open('w') as log:
    result=subprocess.run([sys.executable,'-m','unittest','benchmarks.test_kaggle_write_capacity','-v'],
                          cwd=source,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=45)
assert result.returncode==0,'Disk probe CPU contracts failed; no capacity fill was attempted'
print('DISK_PROBE_CPU_CONTRACTS_PASSED',flush=True)
subprocess.run([sys.executable,'-u',str(source/'benchmarks/kaggle_write_capacity.py'),
                '--output',str(OUT),'--cap-gib','512','--budget','3000'],env=env,check=True,timeout=3050)
'''.replace('SOURCE_BINDING', repr(binding)).replace('EMBEDDED_FILES', repr(payload))
    script = supervise_worker(body, budget=3150, output='/kaggle/working/disk_capacity_supervision')
    script += "\nassert json.loads(pathlib.Path('/kaggle/working/disk_capacity_supervision/supervision.json').read_text())['exit_code']==0\n"
    ast.parse(script)
    metadata = dict(id='pamcks/h3-writable-disk-capacity-cpu-20261029',
        title='H3 Writable Disk Capacity CPU 20261029', code_file='experiment.py',
        language='python', kernel_type='script', is_private=True, enable_gpu=False,
        enable_tpu=False, enable_internet=False, docker_image_pinning_type='original',
        dataset_sources=[], kernel_sources=[], competition_sources=[], model_sources=[])
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/'experiment.py').write_text(script, encoding='utf-8')
    (args.output/'kernel-metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    (args.output/'source_binding.json').write_text(json.dumps(binding, indent=2), encoding='utf-8')
    if not args.submit:
        print(json.dumps(dict(status='built', output=str(args.output), kernel=metadata['id'])))
        return
    state = dict(status='submitting', kernel=metadata['id'], submitted_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        source_binding=binding, script_sha256=hashlib.sha256(script.encode()).hexdigest(),
        run_timeout_seconds=3300, final_collection_count=0, cpu_only=True,
        result_relative='disk_capacity/result.json', collection='One final artifact retrieval; no status/live-log polls')
    receipt = args.output.parent/'submission.json'
    with receipt.open('x', encoding='utf-8') as stream:
        json.dump(state, stream, indent=2)
    try:
        result = subprocess.run([sys.executable, '-m', 'kaggle', 'kernels', 'push', '-p', str(args.output),
                                 '-t', '3300'], capture_output=True, text=True, timeout=180)
        output = result.stdout+result.stderr
        state.update(returncode=result.returncode, status='accepted' if result.returncode == 0 else 'acceptance_uncertain')
        version = re.search(r'Kernel version (\d+)', output)
        if version:
            state['version'] = int(version[1])
        url = re.search(r'https://www\.kaggle\.com/code/[\w-]+/[\w-]+', output)
        if url:
            state['url'] = url[0]
    except subprocess.TimeoutExpired:
        output = 'Submission acknowledgement timed out; preserve claim and do not duplicate automatically.'
        state.update(status='acceptance_uncertain', returncode=-1)
    receipt.write_text(json.dumps(state, indent=2), encoding='utf-8')
    (args.output.parent/'submission_cli.log').write_text(output, encoding='utf-8')
    print(output[-1500:])


if __name__ == '__main__':
    main()
