"""One CPU preflight then one explicitly authorized T4 E2E; never polling."""
import argparse
import ast
import datetime as dt
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

from t4_notebook_watchdog import supervise_worker
from private_reference_delivery import private_reference_prefix

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=('cpu', 'gpu'), required=True)
    parser.add_argument('--evidence', type=Path, required=True)
    parser.add_argument('--reference-delivery', choices=('staged', 'public_url'), default='public_url')
    parser.add_argument('--submit', action='store_true')
    args = parser.parse_args()
    publication = args.evidence/'source_publication.json'
    job = args.evidence/(args.mode+'_job')
    if job.exists():raise FileExistsError('Preserve previous jobs; no automatic duplicate')
    argv = [sys.executable, str(ROOT/'benchmarks/build_kaggle_light_job.py'), '--mode',
            'cpu' if args.mode == 'cpu' else 'e2e', '--publication', str(publication), '--output', str(job)]
    if args.mode == 'cpu':
        argv += ['--review-suite', '--setup-budget', '600']
    else:
        qualifier = args.evidence/'cpu_outputs/anime_inputs/input_qualification.json'
        check = json.loads(qualifier.read_text())
        assert check['status'] == 'cpu_input_ready' and check['lora_targets'] == 208
        collection = json.loads((args.evidence/'cpu_submission_collection_status.json').read_text())
        assert collection['status'] == 'collected' and collection['kaggle_requests'] == 1
        argv += ['--qualification', str(args.evidence/'cpu_outputs/light_setup/setup_receipt.json'),
                 '--turbo-e2e', '--setup-budget', '300', '--execution-budget', '4800']
    subprocess.run(argv, check=True)
    notebook_path = job/'experiment.ipynb'
    notebook = json.loads(notebook_path.read_text())
    code = ''.join(notebook['cells'][0]['source'])
    if args.mode == 'cpu':
        probe = '''import json,os,pathlib,subprocess,sys
src=pathlib.Path('/kaggle/temp/h3_light_source')
out=pathlib.Path('/kaggle/working/anime_inputs');out.mkdir(exist_ok=True)
env=dict(os.environ);env.update(json.loads(pathlib.Path('/kaggle/working/light_setup/execution_env.json').read_text()))
env['CUDA_VISIBLE_DEVICES']='-1'
tests="""import sys,unittest
sys.argv=['tests','--cpu']
import comfy.options;comfy.options.enable_args_parsing()
import comfy.cli_args
sys.argv=['tests']
names=['benchmarks.test_anime_e2e','benchmarks.test_te_column_layout']
result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromNames(names))
assert result.wasSuccessful()
import inspect
from patches.kitchen_attention import int8_attention,source_hashes
assert {'q','k','v','scale','attn_mask'} <= set(inspect.signature(int8_attention).parameters)
assert source_hashes()
"""
with (out/'extra_cpu_tests.log').open('w') as log:
 check=subprocess.run([sys.executable,'-c',tests],env=env,stdout=log,stderr=subprocess.STDOUT,cwd=src)
assert check.returncode==0,'New workload/runtime CPU contracts failed'
argv=[sys.executable,'-u','-m','benchmarks.qualify_anime_inputs','--source',str(src),'--output',str(out)]
os.chdir(src);os.execve(sys.executable,argv,env)
'''
        code += supervise_worker(probe, budget=240, output='/kaggle/working/anime_input_supervision')
        code += "\nassert json.loads(pathlib.Path('/kaggle/working/anime_inputs/input_qualification.json').read_text())['status']=='cpu_input_ready'\n"
    plan_dir=args.evidence/'plan_files'
    if not plan_dir.exists():plan_dir=args.evidence.parent/'plan_files'
    references=json.loads((plan_dir/'l1_assets_identity.json').read_text())['images']
    destination='/kaggle/working/anime_inputs' if args.mode=='cpu' else '/kaggle/temp/h3_light_source/bench_assets/l1_current'
    if args.reference_delivery=='staged':
        code=private_reference_prefix(references,plan_dir.parent,destination)+code
    ast.parse(code)
    notebook['cells'][0]['source'] = code.splitlines(keepends=True)
    notebook_path.write_text(json.dumps(notebook))
    if notebook_path.stat().st_size >= 10 * 1024 * 1024:
        raise ValueError('Private notebook exceeds the conservative 10 MiB delivery budget')
    meta_path = job/'kernel-metadata.json'
    meta = json.loads(meta_path.read_text())
    meta.update(id='pamcks/h3-anime-ref2va-'+('cpu-preflight' if args.mode=='cpu' else '8step-t4')+'-20261027',
                title='H3 Anime REF2VA '+('CPU Preflight' if args.mode=='cpu' else '8Step T4')+' 20261027')
    meta_path.write_text(json.dumps(meta, indent=2))
    binding = json.loads((job/'source_binding.json').read_text())
    binding.update(full_pipeline_authorized=True, workload_contract='anime_ref2va_8step', private_auth=False,
                   reference_delivery=('Exact bytes in private notebook only; absent from public source archive'
                       if args.reference_delivery=='staged' else 'Direct public URLs, mandatory original SHA256 verification'),
                   reference_encoding='base85' if args.reference_delivery=='staged' else None,
                   notebook_bytes=notebook_path.stat().st_size,
                   reference_sha256=[row['sha256'] for row in references])
    (job/'source_binding.json').write_text(json.dumps(binding, indent=2))
    if not args.submit:
        print('Built:', job)
        return
    receipt = args.evidence/(args.mode+'_submission.json')
    state = dict(kernel=meta['id'], submitted_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        status='submitting', returncode=None, source_binding=binding, final_collection_count=0,
        earliest_final_collection_seconds=900 if args.mode=='cpu' else 5400,
        experiment_kind='anime_e2e_cpu' if args.mode=='cpu' else 'anime_e2e',
        result_relative='anime_inputs/input_qualification.json' if args.mode=='cpu' else 'turbo_e2e/generation/result.json',
        run_timeout_seconds=900 if args.mode=='cpu' else 5400,
        notebook_sha256=hashlib.sha256(notebook_path.read_bytes()).hexdigest())
    with receipt.open('x') as stream:json.dump(state, stream, indent=2)
    command = [sys.executable, '-m', 'kaggle', 'kernels', 'push', '-p', str(job), '-t', str(state['run_timeout_seconds'])]
    if args.mode == 'gpu':command += ['--accelerator', 'NvidiaTeslaT4']
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=180)
        output = result.stdout + result.stderr
        state.update(returncode=result.returncode,status='accepted' if result.returncode==0 else 'acceptance_uncertain')
        if result.returncode:
            state['acceptance_reason']='CLI returned no successful acknowledgement; no automatic resubmission'
        version = re.search(r'Kernel version (\d+)', output)
        if version:state['version']=int(version.group(1))
        resolved = re.search(r'https://www\.kaggle\.com/code/([\w-]+/[\w-]+)', output)
        if resolved:
            state.update(requested_kernel=state['kernel'],kernel=resolved.group(1),url=resolved.group(0))
    except subprocess.TimeoutExpired:
        output='Submission acknowledgement timed out; do not automatically resubmit.'
        state.update(returncode=-1,status='acceptance_uncertain')
    (args.evidence/(args.mode+'_submission_cli.log')).write_text(output, encoding='utf-8')
    receipt.write_text(json.dumps(state, indent=2))
    print(output[-1500:], flush=True)


if __name__ == '__main__':main()
