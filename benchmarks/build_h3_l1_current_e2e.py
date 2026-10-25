"""Build a fresh frozen-source L1 retry; never submit historical E2E setup."""
import argparse
import ast
import base64
import gzip
import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile
import build_kaggle_native_torch_jobs as source
from t4_notebook_watchdog import wrap_notebook,supervise_worker

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--preflight',type=Path,required=True);parser.add_argument('--assets-receipt',type=Path,required=True)
    parser.add_argument('--prior-result',type=Path,required=True);parser.add_argument('--retry-assets',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError('Preserve submitted/prepared sources')
    plan=json.loads((ROOT/'kaggle_l1_offline_memory/plan.json').read_text())['L1']
    plan.update(name='L1_current_896x512',width=896,height=512,fps=24,offline_lora=True,memory_profile=False,
        operators={'dit_attention':'tilelang_scalar'},
        source_note='Previous L1 prompt/references/seed/four LCM-simple steps; current production defaults except preserved scalar DiT attention')
    assets=json.loads((args.retry_assets/'assets_identity.json').read_text())
    extra={'l1_current_plan.json':json.dumps(plan,indent=2).encode(),
        'l1_model_identity.json':(ROOT/'semantic_results/pipeline_checkpoint_identity.json').read_bytes(),
        'l1_assets_identity.json':json.dumps(assets,indent=2).encode()}
    for record in assets['images']:
        data=(args.retry_assets/'assets'/record['file']).read_bytes()
        assert hashlib.sha256(data).hexdigest()==record['sha256']
        # Keep the notebook small. Runtime retrieves only these public images
        # and accepts them solely when their frozen size and SHA256 match.
    source.BENCHMARKS+=['run_h3_l1_current_batch.py','run_h3_l1_current_e2e.py','ref2va_e2e_run.py','gpu_memory_overhead.py']
    original=source.capsule
    def capsule():
        data,hashes=original();buffer=io.BytesIO()
        with tarfile.open(fileobj=buffer,mode='w') as dest:
            with tarfile.open(fileobj=io.BytesIO(data),mode='r:gz') as old:
                for item in old:
                    dest.addfile(item,old.extractfile(item))
            for name,data in sorted(extra.items()):
                item=tarfile.TarInfo(name);item.size=len(data);dest.addfile(item,io.BytesIO(data))
                hashes[name]=hashlib.sha256(data).hexdigest()
        return gzip.compress(buffer.getvalue(),mtime=0),hashes
    source.capsule=capsule
    import build_t4_component_followup as previous
    sys.argv=['builder','--diagnosis-only','--output',str(args.output),'--preflight',str(args.preflight),
        '--assets-receipt',str(args.assets_receipt),'--prior-result',str(args.prior_result)]
    previous.main()
    path=args.output/'experiment.ipynb';notebook=json.loads(path.read_text());code=''.join(notebook['cells'][0]['source'])
    code=code.replace("/kaggle/working/t4_validation","/kaggle/working/l1_setup")
    code=code.replace('BUDGET=240\n','BUDGET=600\n')
    start=code.index('    asset_files=');stop=code.index('    bundles=',start)
    code=code[:start]+code[stop:]
    start=code.index('    launcher=');stop=code.index('except BaseException as error:',start)
    code=code[:start]+'''    assert 'release 12.8' in state.get('compiler',''),'Validated preinstalled nvcc 12.8 required'
    assert len({d['uuid'] for d in state['hardware']['devices']})==2,'Two distinct physical device identities required'
    safe_names={'H3_RUNTIME_LOCK','COMFY_ROOT','PYTHONPATH','CUDACXX','CUDA_HOME','CUDA_PATH',
        'OMP_NUM_THREADS','NCCL_P2P_DISABLE','NCCL_DEBUG','NCCL_MIN_NCHANNELS','NCCL_MAX_NCHANNELS',
        'PYTHONUNBUFFERED','TILELANG_CACHE_DIR','CUDA_CACHE_PATH'}
    (OUT/'experiment_env.json').write_text(json.dumps({k:v for k,v in env.items() if k in safe_names},indent=2))
    state['status']='ready_for_e2e';save()
'''+code[stop:]
    code=code.replace("'scope':'Bounded physical TP2 component/operator tests; no H3 sampler or full generation'",
        "'scope':'Setup gate for explicitly authorized full long-sequence E2E retry; no resolver or environment experiment'")
    setup=wrap_notebook(code,budget=600,output='/kaggle/working/l1_setup',omitted_stages=('find_asset_receipt',))
    check="""import json,pathlib
setup_state=json.loads(pathlib.Path('/kaggle/working/l1_setup/batch_results.json').read_text())
supervision=json.loads(pathlib.Path('/kaggle/working/l1_setup/supervision.json').read_text())
if (setup_state['status']!='ready_for_e2e' or supervision['status']!='worker_exited'
        or supervision['exit_code']!=0):raise RuntimeError('Setup gate failed; do not start generation')
"""
    execution="""import json,os,pathlib,sys
print('L1_EXECUTION_WORKER started',flush=True)
allowed=json.loads(pathlib.Path('/kaggle/working/l1_setup/experiment_env.json').read_text())
env=dict(os.environ);env.update(allowed)
source=pathlib.Path('/kaggle/temp/h3_t4_source')
os.execve(sys.executable,[sys.executable,'-u',str(source/'benchmarks/run_h3_l1_current_batch.py'),
    '--source',str(source),'--output','/kaggle/working/l1_e2e'],env)
"""
    execution=supervise_worker(execution,budget=4800,output='/kaggle/working/l1_execution_supervision')
    final_check="""execution_state=json.loads(pathlib.Path('/kaggle/working/l1_execution_supervision/supervision.json').read_text())
if execution_state['status']!='worker_exited' or execution_state['exit_code']!=0:
    raise RuntimeError('L1 execution failed or exceeded its deadline; inspect saved artifacts')
if json.loads(pathlib.Path('/kaggle/working/l1_e2e/batch_results.json').read_text())['status']!='complete':
    raise RuntimeError('L1 batch did not complete')
"""
    full=setup+'\n'+check+'\n'+execution+'\n'+final_check
    ast.parse(full);notebook['cells'][0]['id']='l1-current-frozen';notebook['cells'][0]['source']=full.splitlines(keepends=True)
    path.write_text(json.dumps(notebook))
    meta_path=args.output/'kernel-metadata.json';meta=json.loads(meta_path.read_text())
    meta.update(id='pamcks/h3-l1-current-offline-e2e',title='H3 L1 Current Offline E2E',enable_internet=True,machine_shape='NvidiaTeslaT4',
        kernel_sources=['pamcks/h3-rowpartial-reference-cpu-preflight','pamcks/h3-native-torch-t4-validation'])
    meta_path.write_text(json.dumps(meta,indent=2))
    binding_path=args.output/'source_binding.json';binding=json.loads(binding_path.read_text());binding.pop('assets_receipt_sha256',None)
    binding.update(scope='Explicitly authorized current-infra full L1 request with offline LoRA',plan=plan,
        setup_deadline_seconds=600,execution_deadline_seconds=4800,full_pipeline=True,
        watchdog_generator_sha256=hashlib.sha256((ROOT/'benchmarks/t4_notebook_watchdog.py').read_bytes()).hexdigest(),
        previous_component_result_is_not_generation_acceptance=True)
    binding_path.write_text(json.dumps(binding,indent=2))
    (args.output/'plan.json').write_text(json.dumps(plan,indent=2))
    print(args.output)


if __name__=='__main__':main()
