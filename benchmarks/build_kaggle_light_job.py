"""Tiny CPU/GPU notebooks with zero dataset and kernel-output attachments."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import textwrap
from t4_notebook_watchdog import supervise_worker


def private_auth_execution(execution):
    """Only invoked after the CLI's task-specific private-auth scope check."""
    from huggingface_hub import get_token
    token=get_token()
    if not token:raise RuntimeError('Normal local HF login is required; do not put a token in CLI arguments')
    auth="""import runpy,kaggle_secrets
pathlib.Path(__file__).unlink(missing_ok=True)
temporary_hf_auth=[TEMPORARY_AUTH]
class TemporaryPrivateHF:
    def get_secret(self,name):
        if name in ('HF_TOKEN','HF_token') and temporary_hf_auth:return temporary_hf_auth.pop()
        raise KeyError(name)
kaggle_secrets.UserSecretsClient=TemporaryPrivateHF
os.environ.update(env)
sys.path[:0]=env['PYTHONPATH'].split(os.pathsep)
sys.argv=argv[2:]
runpy.run_path(argv[2],run_name='__main__')
""".replace('TEMPORARY_AUTH',repr(token))
    del token
    assert execution.count('os.execve(sys.executable,argv,env)')==1
    return execution.replace('os.execve(sys.executable,argv,env)',auth)


def execution_supervisor(execution, *, budget, output, private_auth=False):
    if private_auth:execution=private_auth_execution(execution)
    result=supervise_worker(execution,budget=budget,output=output)
    if private_auth:
        # Also remove the credential-bearing launcher if it fails before its
        # self-unlink. The private notebook is the only retained auth surface.
        result='try:\n'+textwrap.indent(result,'    ')+"\nfinally:\n    pathlib.Path("+repr(output)+").joinpath('bounded_worker.py').unlink(missing_ok=True)\n"
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=('cpu','e2e','operators','dit-memory','dit-profile','review-operators','vae-parallel'),required=True)
    p.add_argument('--review-suite',action='store_true',help='Qualify the review follow-up on a CPU instance')
    p.add_argument('--review-recovery',action='store_true',help='Retry only unfinished review LoRA timings; complete the previously unstarted trace')
    p.add_argument('--turbo-e2e',action='store_true',help='Current-default 0.5MP E2E with Native TP2 offline LoRA persisted to disk')
    p.add_argument('--vae-compare-preflight',action='store_true',help='CPU-only authentication/hash gate for the private Video decoder input')
    p.add_argument('--vae-input-qualification',type=Path,help='Matching CPU-only real-latent authentication receipt')
    p.add_argument('--vae-ab-bake',action='store_true',help='Private Dataset VAE checkpoint replay and bake-only profile; CPU gate or two T4s')
    p.add_argument('--dit-pair',action='store_true',help='CPU gate or two consecutive 59134-token DiT calls; no full pipeline')
    p.add_argument('--pair-qualification',type=Path)
    p.add_argument('--publication',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--qualification',type=Path);p.add_argument('--private-input',type=Path)
    p.add_argument('--temporary-private-auth',action='store_true',help='Explicit user-authorized temporary auth in a private operator submission only')
    p.add_argument('--profile-source',type=Path,help='Directory containing the separately qualified read-only profiler')
    p.add_argument('--profile-qualification',type=Path)
    p.add_argument('--setup-budget',type=int,default=900)
    p.add_argument('--execution-budget',type=int)
    p.add_argument('--dit-lifetimes',action='store_true',help='Enable paired full-shape column gates before the DiT-only lifetime profile')
    p.add_argument('--dit-attention',choices=('kitchen','tilelang_scalar'),default='kitchen')
    args=p.parse_args()
    if args.output.exists():raise FileExistsError('Preserve submitted notebooks')
    publication=json.loads(args.publication.read_text());assert publication['published']
    if args.mode!='cpu' and args.qualification is None:p.error('GPU jobs require the new direct-pip CPU receipt')
    if args.dit_pair:
        if args.mode not in ('cpu','e2e'):p.error('--dit-pair requires cpu or e2e setup')
        if args.vae_ab_bake or args.turbo_e2e or args.private_input or args.temporary_private_auth:p.error('DiT pair is a standalone fixture job without private auth')
        if args.mode=='e2e' and not args.pair_qualification:p.error('DiT pair requires its CPU fixture qualification')
    if args.vae_ab_bake and args.mode not in ('cpu','e2e'):p.error('--vae-ab-bake requires cpu or e2e setup')
    if args.vae_ab_bake and (args.temporary_private_auth or args.private_input):p.error('VAE A/B uses a private Dataset, without embedded HF credentials')
    if args.vae_ab_bake and args.mode=='e2e' and not args.vae_input_qualification:p.error('VAE A/B requires its CPU Dataset gate')
    if (args.mode in ('operators','vae-parallel') or args.vae_compare_preflight) and args.private_input is None:p.error('This run requires a private HF capture locator')
    if args.review_suite and args.mode!='cpu':p.error('--review-suite requires --mode cpu')
    if args.review_recovery and args.mode!='review-operators':p.error('--review-recovery requires --mode review-operators')
    if args.turbo_e2e and args.mode!='e2e':p.error('--turbo-e2e requires --mode e2e')
    if args.vae_compare_preflight and args.mode!='cpu':p.error('--vae-compare-preflight requires --mode cpu')
    if args.mode=='vae-parallel' and args.vae_input_qualification is None:p.error('Video comparison requires its CPU input-authentication gate')
    if args.temporary_private_auth and not (args.mode in ('operators','vae-parallel') or (args.mode=='cpu' and args.vae_compare_preflight)):
        p.error('Temporary auth is restricted to the explicitly authorized private operator/Video comparison scopes')
    if not 60<=args.setup_budget<=900:p.error('Setup budget must be 60–900 seconds')
    if args.execution_budget is not None and not 30<=args.execution_budget<=4800:p.error('Invalid execution budget')
    if args.mode=='dit-memory' and (args.profile_source is None or args.profile_qualification is None):
        p.error('DiT memory requires source and its CPU qualification')
    if args.dit_lifetimes and args.mode!='dit-memory':p.error('--dit-lifetimes requires --mode dit-memory')
    if args.dit_attention!='kitchen' and args.mode!='dit-profile':p.error('--dit-attention requires --mode dit-profile')
    profile_files={n:(args.profile_source/n).read_text(encoding='utf-8') for n in
        ('run_dit_memory_51220.py','dit_memory_lines.py','analyze_gpu_memory_overhead.py')} if args.profile_source else {}
    profile_hashes={n:hashlib.sha256(v.encode()).hexdigest() for n,v in profile_files.items()}
    if args.mode=='dit-memory':
        pq=json.loads(args.profile_qualification.read_text())
        assert pq['status']=='profile_cpu_ready' and pq['driver_sha256']==profile_hashes['run_dit_memory_51220.py']
        assert pq['line_observer_sha256']==profile_hashes['dit_memory_lines.py']
        assert pq['analyzer_sha256']==profile_hashes['analyze_gpu_memory_overhead.py']
    qualification=json.loads(args.qualification.read_text()) if args.qualification else None
    if qualification:
        assert qualification['status']=='cpu_ready' and qualification['capsule_sha256']==publication['capsule_sha256']
        if args.mode=='review-operators':assert qualification.get('review_suite',{}).get('passed')
        if args.turbo_e2e:
            assert qualification.get('disk_bake_cpu',{}).get('passed'),'Fresh matching disk-bake CPU qualification required'
            assert qualification.get('review_suite',{}).get('passed')
    private=json.loads(args.private_input.read_text()) if args.private_input else None
    if args.dit_pair and args.mode=='e2e':
        gate=json.loads(args.pair_qualification.read_text())
        assert gate['status']=='pair_cpu_ready' and gate['source_capsule_sha256']==publication['capsule_sha256']
        assert gate['input_identity']['tokens']==59134
        assert qualification.get('review_suite',{}).get('passed') and qualification.get('disk_bake_cpu',{}).get('passed')
    if args.vae_ab_bake and args.mode=='e2e':
        gate=json.loads(args.vae_input_qualification.read_text())
        assert gate['status']=='cpu_input_ready' and gate['source_capsule_sha256']==publication['capsule_sha256']
        assert gate['input_sha256']=='08a10be3efdda85ea6415c832eb2654101da5d5fabe54b6e34002071ee2dde25'
        assert qualification.get('review_suite',{}).get('passed')
    if args.mode=='vae-parallel':
        assert qualification.get('review_suite',{}).get('passed'),'Fresh decoder CPU review required'
        gate=json.loads(args.vae_input_qualification.read_text())
        assert gate['status']=='cpu_input_ready' and gate['source_capsule_sha256']==publication['capsule_sha256']
        assert gate['input']['file_sha256']==private['sha256']
    args.output.mkdir(parents=True)
    prelude="""print('LIGHT_NOTEBOOK_START zero dataset/kernel mounts',flush=True)
import json,pathlib
OUT=pathlib.Path('/kaggle/working/light_setup');OUT.mkdir(exist_ok=True)
"""
    if qualification:prelude+="(OUT/'cpu_qualification.json').write_text("+repr(json.dumps(qualification))+')\n'
    if private:prelude+="(OUT/'private_input_locator.json').write_text("+repr(json.dumps(private))+')\n'
    body='''import hashlib,io,json,os,pathlib,sys,tarfile,urllib.request
print('LIGHT_SOURCE_DOWNLOAD_START',flush=True)
PUB=PUBLICATION
DIST=pathlib.Path('/kaggle/temp/h3_light_distribution');DIST.mkdir(parents=True,exist_ok=True)
SRC=pathlib.Path('/kaggle/temp/h3_light_source');SRC.mkdir(parents=True,exist_ok=True)
with urllib.request.urlopen(PUB['url'],timeout=90) as response:
    raw=response.read(PUB['bytes']+1)
assert len(raw)==PUB['bytes'] and hashlib.sha256(raw).hexdigest()==PUB['sha256'],'Frozen distribution checksum mismatch'
with tarfile.open(fileobj=io.BytesIO(raw),mode='r:gz') as archive:archive.extractall(DIST,filter='data')
payload=(DIST/'capsule.tar.gz.bin').read_bytes()
assert hashlib.sha256(payload).hexdigest()==PUB['capsule_sha256']
with tarfile.open(fileobj=io.BytesIO(payload),mode='r:gz') as archive:archive.extractall(SRC,filter='data')
pathlib.Path('/kaggle/working/light_setup/delivery_binding.json').write_text(json.dumps(PUB,indent=2))
print('LIGHT_SOURCE_READY',flush=True)
argv=[sys.executable,'-u',str(SRC/'benchmarks/prepare_kaggle_direct_pip.py'),
    '--distribution',str(DIST),'--source',str(SRC),'--output','/kaggle/working/light_setup','--mode',MODE]
if MODE!='cpu':argv+=['--qualification','/kaggle/working/light_setup/cpu_qualification.json']
os.execve(sys.executable,argv,dict(os.environ))
'''.replace('PUBLICATION',repr(publication)).replace('MODE',repr('operators' if args.mode=='dit-memory' else args.mode))
    if args.review_suite:
        body=body.replace('os.execve(sys.executable,argv,dict(os.environ))',
            "argv+=['--review-suite']\nos.execve(sys.executable,argv,dict(os.environ))")
    code=prelude+supervise_worker(body,budget=args.setup_budget,output='/kaggle/working/light_setup')
    code+='''
receipt=json.loads((OUT/'setup_receipt.json').read_text())
guard=json.loads((OUT/'supervision.json').read_text())
if guard['status']!='worker_exited' or guard['exit_code']!=0:raise RuntimeError('Light setup worker failed')
'''
    profile_prelude=''
    if profile_files:
        profile_prelude="""import hashlib
PROFILE=pathlib.Path('/kaggle/temp/h3_dit_memory_profiler');PROFILE.mkdir(exist_ok=True)
for name,value in PROFILE_FILES.items():
    assert hashlib.sha256(value.encode()).hexdigest()==PROFILE_HASHES[name]
    (PROFILE/name).write_text(value)
""".replace('PROFILE_FILES',repr(profile_files)).replace('PROFILE_HASHES',repr(profile_hashes))
    if args.mode=='cpu':
        code+="assert receipt['status']=='cpu_ready'\n"
        if args.dit_pair:
            probe="""import json,os,pathlib,sys
env=dict(os.environ);env.update(json.loads(pathlib.Path('/kaggle/working/light_setup/execution_env.json').read_text()))
source=pathlib.Path('/kaggle/temp/h3_light_source')
argv=[sys.executable,'-u',str(source/'benchmarks/run_dit_pair_batch.py'),
 '--source',str(source),'--output','/kaggle/working/dit_pair_preflight','--cpu-preflight']
os.execve(sys.executable,argv,env)
"""
            code+=execution_supervisor(probe,budget=180,output='/kaggle/working/dit_pair_preflight_supervision')
            code+="assert json.loads(pathlib.Path('/kaggle/working/dit_pair_preflight/batch_results.json').read_text())['status']=='pair_cpu_ready'\n"
        if args.vae_ab_bake:
            probe="""import json,os,pathlib,sys
env=dict(os.environ);env.update(json.loads(pathlib.Path('/kaggle/working/light_setup/execution_env.json').read_text()))
source=pathlib.Path('/kaggle/temp/h3_light_source')
argv=[sys.executable,'-u',str(source/'benchmarks/run_vae_bake_batch.py'),
 '--source',str(source),'--output','/kaggle/working/vae_bake_preflight','--cpu-preflight']
os.execve(sys.executable,argv,env)
"""
            code+=execution_supervisor(probe,budget=240,output='/kaggle/working/vae_bake_preflight_supervision')
            code+="assert json.loads(pathlib.Path('/kaggle/working/vae_bake_preflight/result.json').read_text())['status']=='cpu_input_ready'\n"
        if args.vae_compare_preflight:
            probe="""import json,os,pathlib,sys
env=dict(os.environ);env.update(json.loads(pathlib.Path('/kaggle/working/light_setup/execution_env.json').read_text()))
env.update(HF_HUB_DISABLE_XET='1',HF_HUB_DISABLE_PROGRESS_BARS='1')
source=pathlib.Path('/kaggle/temp/h3_light_source')
argv=[sys.executable,'-u',str(source/'benchmarks/run_vae_temporal_batch.py'),
 '--source',str(source),'--distribution','/kaggle/temp/h3_light_distribution',
 '--private-input','/kaggle/working/light_setup/private_input_locator.json',
 '--output','/kaggle/working/vae_preflight','--cpu-preflight']
os.execve(sys.executable,argv,env)
"""
            code+=execution_supervisor(probe,budget=180,output='/kaggle/working/vae_preflight_supervision',private_auth=args.temporary_private_auth)
            code+="assert json.loads(pathlib.Path('/kaggle/working/vae_preflight/batch_results.json').read_text())['status']=='cpu_input_ready'\n"
        if profile_files:
            probe="""import json,os,pathlib,sys
env=dict(os.environ);env.update(json.loads(pathlib.Path('/kaggle/working/light_setup/execution_env.json').read_text()))
"""+profile_prelude+"""
os.execve(sys.executable,[sys.executable,'-u',str(PROFILE/'run_dit_memory_51220.py'),
 '--source','/kaggle/temp/h3_light_source','--output','/kaggle/working/profile_cpu','--cpu-preflight'],env)
"""
            code+=supervise_worker(probe,budget=120,output='/kaggle/working/profile_cpu_supervision')
            code+="assert json.loads(pathlib.Path('/kaggle/working/profile_cpu/profile_qualification.json').read_text())['status']=='profile_cpu_ready'\n"
    else:
        code+="assert receipt['status']=='ready_for_device_gate'\n"
        execution='''import json,os,pathlib,sys
print('LIGHT_GPU_EXECUTION_START',flush=True)
env=dict(os.environ);env.update(json.loads(pathlib.Path('/kaggle/working/light_setup/execution_env.json').read_text()))
env.update(HF_HUB_DISABLE_XET='1',HF_HUB_DISABLE_PROGRESS_BARS='1')
source=pathlib.Path('/kaggle/temp/h3_light_source')
argv=[sys.executable,'-u',str(source/'benchmarks/launch_kaggle_light.py'),'--source',str(source),
 '--distribution','/kaggle/temp/h3_light_distribution','--output','/kaggle/working/light_execution','--mode',MODE]
if MODE=='operators':argv+=['--private-input','/kaggle/working/light_setup/private_input_locator.json']
if MODE=='dit-profile':argv+=['--dit-attention',ATTENTION]
os.execve(sys.executable,argv,env)
'''.replace('MODE',repr(args.mode)).replace('ATTENTION',repr(args.dit_attention))
        if args.mode=='dit-memory':
            start=execution.index('argv=')
            execution=execution[:start]+profile_prelude+"""
argv=[sys.executable,'-u',str(PROFILE/'run_dit_memory_51220.py'),'--source',str(source),
 '--output','/kaggle/working/dit_memory']
os.execve(sys.executable,argv,env)
"""
            if args.dit_lifetimes:
                execution=execution.replace(" '--output','/kaggle/working/dit_memory']", " '--output','/kaggle/working/dit_memory','--lifetime-scheduling']")
        if args.mode=='review-operators':
            execution=execution[:execution.index('argv=')]+"""
argv=[sys.executable,'-u',str(source/'benchmarks/run_review_t4.py'),
 '--output','/kaggle/working/review_operators','--assets','/kaggle/temp/h3_review_assets']
os.execve(sys.executable,argv,env)
"""
            if args.review_recovery:
                execution=execution.replace('os.execve(sys.executable,argv,env)',
                    "argv+=['--retry-lora-only']\nos.execve(sys.executable,argv,env)")
        if args.turbo_e2e:
            execution=execution[:execution.index('argv=')]+"""
argv=[sys.executable,'-u',str(source/'benchmarks/run_h3_l1_turbo_batch.py'),
 '--source',str(source),'--output','/kaggle/working/turbo_e2e']
os.execve(sys.executable,argv,env)
"""
        if args.vae_ab_bake:
            execution=execution[:execution.index('argv=')]+"""
argv=[sys.executable,'-u',str(source/'benchmarks/run_vae_bake_batch.py'),
 '--source',str(source),'--output','/kaggle/working/vae_bake']
os.execve(sys.executable,argv,env)
"""
        if args.dit_pair:
            execution=execution[:execution.index('argv=')]+"""
argv=[sys.executable,'-u',str(source/'benchmarks/run_dit_pair_batch.py'),
 '--source',str(source),'--output','/kaggle/working/dit_pair']
os.execve(sys.executable,argv,env)
"""
        if args.mode=='vae-parallel':
            execution=execution[:execution.index('argv=')]+"""
argv=[sys.executable,'-u',str(source/'benchmarks/run_vae_temporal_batch.py'),
 '--source',str(source),'--distribution','/kaggle/temp/h3_light_distribution',
 '--private-input','/kaggle/working/light_setup/private_input_locator.json',
 '--output','/kaggle/working/vae_parallel']
os.execve(sys.executable,argv,env)
"""
        code+=execution_supervisor(execution,budget=args.execution_budget or (4800 if args.mode=='e2e' else 1200),
            output='/kaggle/working/light_execution_supervision',private_auth=args.temporary_private_auth)
        code+='''
guard=json.loads(pathlib.Path('/kaggle/working/light_execution_supervision/supervision.json').read_text())
if guard['status']!='worker_exited' or guard['exit_code']!=0:raise RuntimeError('Light execution failed; inspect artifacts')
'''
    ast.parse(code)
    ids={'cpu':'h3-dit-memory-cpu' if profile_files else 'h3-light-cpu','e2e':'h3-l1-current-offline-e2e','operators':'h3-rowpartial-reference-t4','dit-memory':'h3-dit-memory-51220','dit-profile':'h3-dit-single-profile'}
    titles={'cpu':'H3 DiT Memory CPU' if profile_files else 'H3 Light CPU','e2e':'H3 L1 Current Offline E2E','operators':'H3 RowPartial Reference T4','dit-memory':'H3 DiT Memory 51220','dit-profile':'H3 DiT Single Profile'}
    if args.mode=='dit-profile' and args.dit_attention=='tilelang_scalar':
        ids['dit-profile']='h3-dit-scalar-single-profile';titles['dit-profile']='H3 DiT Scalar Single Profile'
    ids['review-operators']='h3-review-operators-t4-20261024';titles['review-operators']='H3 Review Operators T4 20261024'
    if args.review_suite:ids['cpu']='h3-review-operators-cpu-20261024';titles['cpu']='H3 Review Operators CPU 20261024'
    if args.turbo_e2e:ids['e2e']='h3-l1-turbo-05mp-t4';titles['e2e']='H3 L1 Turbo 0.5MP T4 Disk Bake'
    ids['vae-parallel']='h3-vae-query-temporal-t4-20261027';titles['vae-parallel']='H3 VAE Query Temporal T4 20261027'
    if args.vae_compare_preflight:ids['cpu']='h3-vae-temporal-cpu-20261027';titles['cpu']='H3 VAE Temporal CPU 20261027'
    if args.vae_ab_bake:
        ids[args.mode]='h3-vae-bake-cpu-20261028' if args.mode=='cpu' else 'h3-vae-checkpoints-bake-t4-20261028'
        titles[args.mode]='H3 VAE Bake CPU 20261028' if args.mode=='cpu' else 'H3 VAE Checkpoints Bake T4 20261028'
    if args.dit_pair:
        ids[args.mode]='h3-dit-pair-cpu-20261028' if args.mode=='cpu' else 'h3-dit-pair-59134-t4-20261028'
        titles[args.mode]='H3 DiT Pair CPU 20261028' if args.mode=='cpu' else 'H3 DiT Pair 59134 T4 20261028'
    meta=dict(id='pamcks/'+ids[args.mode],title=titles[args.mode],code_file='experiment.ipynb',language='python',
        kernel_type='notebook',is_private=True,enable_gpu=args.mode!='cpu',enable_tpu=False,enable_internet=True,
        docker_image_pinning_type='original',dataset_sources=[],kernel_sources=[],competition_sources=[],model_sources=[])
    if args.vae_ab_bake:meta['dataset_sources']=['pamcks/h3-vae-ab-latents-20261028']
    if args.mode!='cpu':meta['machine_shape']='NvidiaTeslaT4'
    notebook=dict(nbformat=4,nbformat_minor=5,metadata=dict(kernelspec=dict(name='python3',language='python',display_name='Python 3')),
        cells=[dict(id='light-delivery',cell_type='code',execution_count=None,metadata={},outputs=[],source=code.splitlines(keepends=True))])
    (args.output/'experiment.ipynb').write_text(json.dumps(notebook));(args.output/'kernel-metadata.json').write_text(json.dumps(meta,indent=2))
    (args.output/'source_binding.json').write_text(json.dumps(dict(mode=args.mode,publication=publication,
        qualification_sha256=hashlib.sha256(args.qualification.read_bytes()).hexdigest() if qualification else None,
        kaggle_dataset_sources=meta['dataset_sources'],kaggle_kernel_sources=[],temporary_private_auth=args.temporary_private_auth,
        profile_hashes=profile_hashes,setup_budget=args.setup_budget,
        execution_budget=args.execution_budget,
        dit_lifetimes=args.dit_lifetimes,
        dit_attention=args.dit_attention,
        review_suite=args.review_suite,
        review_recovery=args.review_recovery,
        turbo_e2e=args.turbo_e2e,
        vae_compare_preflight=args.vae_compare_preflight,
        vae_ab_bake=args.vae_ab_bake,
        dit_pair=args.dit_pair,
        pair_qualification_sha256=hashlib.sha256(args.pair_qualification.read_bytes()).hexdigest() if args.pair_qualification else None,
        vae_input_qualification_sha256=hashlib.sha256(args.vae_input_qualification.read_bytes()).hexdigest() if args.vae_input_qualification else None,
        profile_qualification_sha256=hashlib.sha256(args.profile_qualification.read_bytes()).hexdigest() if args.profile_qualification else None),indent=2))
    print(args.output)


if __name__=='__main__':main()
