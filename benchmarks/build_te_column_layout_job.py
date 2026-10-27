"""Build one source-bound T4 layout ablation using the qualified frozen setup."""
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

ROOT = Path(__file__).resolve().parents[1]
SOURCES = (
    'patches/dit_column_linear.py',
    'benchmarks/te_column_layout.py',
    'benchmarks/test_te_column_layout.py',
    'benchmarks/run_te_column_layout.py',
    'benchmarks/te_parallel_matrix.py',
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--qualified', type=Path, required=True)
    parser.add_argument('--evidence', type=Path, required=True)
    parser.add_argument('--submit', action='store_true')
    args = parser.parse_args()
    job = args.evidence / 'gpu_job'
    if job.exists():
        raise FileExistsError('Preserve existing submissions; no automatic duplicate')
    pub = json.loads((args.qualified / 'source_publication_v2.json').read_text())
    receipt_path = args.qualified / 'cpu_outputs/light_setup/setup_receipt.json'
    cpu = json.loads(receipt_path.read_text())
    assert cpu['status'] == 'cpu_ready' and cpu['capsule_sha256'] == pub['capsule_sha256']
    assert cpu['review_suite']['passed']
    assert cpu['core_pins']['comfy-kitchen'] == '0.2.37' and cpu['core_pins']['tilelang'] == '0.1.13'
    local = json.loads((args.evidence / 'local_preview_final/rank0.json').read_text())
    assert local['status'] == 'complete' and not local['physical_t4_qualification']
    assert local['signed_extremes_int32_exact'] and len(local['cases']) == 4
    assert all(all(c['byte_exact'][name]['byte_exact'] and c['byte_exact'][name]['column_gemm_calls'] == 5
                   for name in ('legacy_copy', 'torch_view', 'kitchen_direct')) for c in local['cases'])
    log = (args.evidence / 'cpu_tests.log').read_text(encoding='utf-8')
    assert re.search(r'Ran 19 tests', log) and log.rstrip().endswith('OK')
    files = {name: (ROOT / name).read_bytes() for name in SOURCES}
    credential = re.compile(rb'(?:hf_[A-Za-z0-9]{25,}|gh[pousr]_[A-Za-z0-9]{30,}|KGAT_[A-Za-z0-9_\-]{20,})')
    assert not any(credential.search(raw) for raw in files.values())
    for raw in files.values():
        ast.parse(raw)
    hashes = {name: hashlib.sha256(raw).hexdigest() for name, raw in files.items()}
    manifest = json.loads((args.qualified / 'distribution_v2/delivery_manifest.json').read_text())
    patch_name = 'patches/dit_column_linear.py'
    runtime_names = ('tp_qwen3vl', 'h3_operator_policy', 'h3_tensor_flow', 'tilelang_group32_masked',
                     'kitchen_te_attention', 'optimized_tp', 'tilelang_transport_p2',
                     'tilelang_swiglu_quant', 'parallel_int8')
    unchanged = {f'patches/{name}.py': hashlib.sha256((ROOT / 'patches' / f'{name}.py').read_bytes()).hexdigest()
                 for name in runtime_names}
    assert all(manifest['source_hashes'][name] == value for name, value in unchanged.items())
    setup = (args.qualified / 'gpu_outputs/light_setup/bounded_worker.py').read_text()
    assert setup.count("'--mode','vae-parallel'") == 1
    setup = setup.replace("'--mode','vae-parallel'", "'--mode','operators'")
    ast.parse(setup)
    code = "import json,pathlib\nOUT=pathlib.Path('/kaggle/working/light_setup');OUT.mkdir(exist_ok=True)\n"
    code += "(OUT/'cpu_qualification.json').write_text(" + repr(json.dumps(cpu)) + ')\n'
    code += supervise_worker(setup, budget=300, output='/kaggle/working/light_setup')
    code += "\nassert json.loads((OUT/'setup_receipt.json').read_text())['status']=='ready_for_device_gate'\n"
    execution = '''import base64,hashlib,json,os,pathlib,subprocess,sys
src=pathlib.Path('/kaggle/temp/h3_light_source')
FILES=SOURCE_FILES
HASHES=SOURCE_HASHES
old=src/'patches/dit_column_linear.py'
assert hashlib.sha256(old.read_bytes()).hexdigest()==OLD_PATCH_SHA
for name,encoded in FILES.items():
 raw=base64.b64decode(encoded);assert hashlib.sha256(raw).hexdigest()==HASHES[name]
 path=src/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(raw)
out=pathlib.Path('/kaggle/working/te_layout');out.mkdir(exist_ok=True)
(out/'source_hashes.json').write_text(json.dumps(HASHES,indent=2))
env=dict(os.environ);env.update(json.loads(pathlib.Path('/kaggle/working/light_setup/execution_env.json').read_text()))
env['PYTHONPATH']=str(src)+os.pathsep+env.get('PYTHONPATH','');os.chdir(src)
with (out/'cpu_tests.log').open('w') as log:
 test=subprocess.run([sys.executable,'-m','unittest','benchmarks.test_te_column_layout','-v'],env=env,stdout=log,stderr=subprocess.STDOUT)
assert test.returncode==0,'New layout CPU contracts failed on frozen Linux setup'
argv=[sys.executable,'-u','-m','benchmarks.run_te_column_layout','--source',str(src),'--output',str(out)]
os.execve(sys.executable,argv,env)
'''.replace('SOURCE_FILES', repr({name: base64.b64encode(raw).decode() for name, raw in files.items()}))
    execution = execution.replace('SOURCE_HASHES', repr(hashes)).replace('OLD_PATCH_SHA', repr(manifest['source_hashes'][patch_name]))
    ast.parse(execution)
    code += supervise_worker(execution, budget=900, output='/kaggle/working/layout_supervision')
    code += "\nguard=json.loads(pathlib.Path('/kaggle/working/layout_supervision/supervision.json').read_text())\nassert guard['status']=='worker_exited' and guard['exit_code']==0\n"
    ast.parse(code)
    job.mkdir(parents=True)
    metadata = dict(id='pamcks/h3-te-column-layout-t4-20261027', title='H3 TE Column Layout T4 20261027',
                    code_file='experiment.ipynb', language='python', kernel_type='notebook',
                    is_private=True, enable_gpu=True, enable_tpu=False, enable_internet=True,
                    machine_shape='NvidiaTeslaT4', docker_image_pinning_type='original',
                    dataset_sources=[], kernel_sources=[], competition_sources=[], model_sources=[])
    notebook = dict(nbformat=4, nbformat_minor=5,
                    metadata=dict(kernelspec=dict(name='python3', language='python', display_name='Python 3')),
                    cells=[dict(id='te-column-layout', cell_type='code', execution_count=None,
                                metadata={}, outputs=[], source=code.splitlines(keepends=True))])
    (job / 'experiment.ipynb').write_text(json.dumps(notebook))
    (job / 'kernel-metadata.json').write_text(json.dumps(metadata, indent=2))
    binding = dict(publication=pub, supplemental_sources=hashes,
                   cpu_qualification_sha256=hashlib.sha256(receipt_path.read_bytes()).hexdigest(),
                   bootstrap_sha256=hashlib.sha256(setup.encode()).hexdigest(),
                   unchanged_runtime_hashes=unchanged,
                   runtime_refactor=dict(path=patch_name, before=manifest['source_hashes'][patch_name],
                       after=hashes[patch_name], semantics='Extract pinned column GEMM call; default remains legacy contiguous transpose'),
                   local_cpu_tests=19, local_preview_sha256=hashlib.sha256((args.evidence / 'local_preview_final/rank0.json').read_bytes()).hexdigest(),
                   dataset_mounts=0, kernel_mounts=0, private_auth=False,
                   setup_budget_seconds=300, execution_budget_seconds=900,
                   attention='kitchen_causal', lengths=[128, 353, 1306, 4096], warmups=5, repeats=9,
                   scope='Column operator and current trained TP2 block; no full TE or H3 generation')
    (job / 'source_binding.json').write_text(json.dumps(binding, indent=2))
    print('Built:', job, flush=True)
    if not args.submit:
        return
    receipt = args.evidence / 'gpu_submission.json'
    state = dict(kernel=metadata['id'], submitted_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                 status='submitting', returncode=None, source_binding=binding,
                 final_collection_count=0, earliest_final_collection_seconds=1500,
                 experiment_kind='te_column_layout', result_relative='te_layout/result.json',
                 run_timeout_seconds=1500,
                 notebook_sha256=hashlib.sha256((job / 'experiment.ipynb').read_bytes()).hexdigest())
    with receipt.open('x') as stream:
        json.dump(state, stream, indent=2)
    try:
        result = subprocess.run([sys.executable, '-m', 'kaggle', 'kernels', 'push', '-p', str(job),
                                 '--accelerator', 'NvidiaTeslaT4', '-t', '1500'],
                                capture_output=True, text=True, timeout=180)
        output = result.stdout + result.stderr
        state.update(returncode=result.returncode, status='accepted' if result.returncode == 0 else 'submission_failed')
        version = re.search(r'Kernel version (\d+)', output)
        if version:
            state['version'] = int(version.group(1))
    except subprocess.TimeoutExpired:
        output = 'Submission acknowledgement timed out; no automatic retry.'
        state.update(returncode=-1, status='acceptance_uncertain')
    (args.evidence / 'gpu_submission_cli.log').write_text(output, encoding='utf-8')
    receipt.write_text(json.dumps(state, indent=2))
    print(output[-1500:], flush=True)


if __name__ == '__main__':
    main()
