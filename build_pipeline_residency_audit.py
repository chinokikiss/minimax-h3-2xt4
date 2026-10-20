"""Freeze a locally qualified full pipeline into one bounded dual-T4 audit."""
import base64
import argparse
import ast
import hashlib
import io
import json
from pathlib import Path
import tarfile

ROOT=Path(__file__).resolve().parent


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--reuse-reference',action='store_true',help='Mount the completed v3 Native reference; run TP comparisons only')
    parser.add_argument('--adaptive-only',action='store_true',help='Retest the failed adaptive strategy; retain the completed physical baseline')
    parser.add_argument('--cache-gib',type=float,default=3.)
    parser.add_argument('--qualification-dir',default='semantic_results/pipeline_audit_full_local')
    parser.add_argument('--native-gate',default='semantic_results/native_resident_qualification/result.json')
    parser.add_argument('--loading-study',action='store_true',help='Fresh Native, direct sync/adaptive, and offline adaptive; current P0/P1/P2 proofs required')
    parser.add_argument('--offline-mode',choices=('skip','convert','reuse'),default='skip',
                        help='Skip offline by default; optional conversion runs only after direct benchmarks')
    parser.add_argument('--offline-transport',type=Path)
    parser.add_argument('--p012-study',action='store_true',help='Separate P1 and P2; reuse immutable offline bundle')
    parser.add_argument('--wrapper-qualification-dir',default='semantic_results/loading_wrapper')
    parser.add_argument('--native-kernel-source',help='Reuse completed Native tensors from a previous notebook output')
    parser.add_argument('--native-reference-transport',type=Path,help='Private hash-verified Native download ticket when Kaggle refuses a failed kernel input')
    options=parser.parse_args()
    if options.native_kernel_source and not options.p012_study:
        raise ValueError('Native kernel reuse requires the qualified P0/P1/P2 study')
    if options.native_reference_transport:
        ticket=json.loads(options.native_reference_transport.read_text())
        if not options.native_kernel_source or ticket['kernel']!=options.native_kernel_source or ticket['version']!=1:
            raise ValueError('Native transport source identity mismatch')
        if set(ticket['objects'])!={'native_result.json',*[n+'.pt' for n in ('conditioning','video','audio','pixels','waveform')]}:
            raise ValueError('Native transport tensor set incomplete')
    transport=None
    if options.offline_mode=='reuse':
        if not options.loading_study or not options.offline_transport:
            raise ValueError('Offline reuse requires loading study and a private download ticket')
        transport=json.loads(options.offline_transport.read_text())
        publication=json.loads((ROOT/'semantic_results/kaggle_hf_publication_v6/publication.json').read_text())
        if (len(transport['objects'])!=200 or transport['revision']!=publication['commit'] or
                transport['bundle_sha256']!=publication['bundle_sha256']):
            raise ValueError('Verified immutable offline publication required')
    if options.p012_study and not transport:
        raise ValueError('P0/P1/P2 study requires offline reuse')
    proofs=[]
    native_gate=ROOT/options.native_gate
    gate=json.loads(native_gate.read_text())
    if not gate['passed']:raise RuntimeError('Native placement sampler qualification must pass')
    for name,sha in gate['source_sha256'].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=sha:
            raise RuntimeError('Native placement changed since qualification: '+name)
    for strategy in ('sync','adaptive'):
        path=ROOT/options.qualification_dir/f'{strategy}_result.json'
        result=json.loads(path.read_text())
        if not result['passed'] or result['physical'] or result['layers']!=50:
            raise RuntimeError('Full-depth local comparison must pass: '+strategy)
        for name,sha in result['source_sha256'].items():
            if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=sha:
                raise RuntimeError('Changed since execution: '+name)
        for request in result['requests']:
            if not all(check['passed'] for check in request['correctness'].values()):raise RuntimeError('Local Native comparison failed')
            if any(abs(rank['host_cache']['capacity_bytes']/2**30-options.cache_gib)>1e-6 for rank in request['runtime']):
                raise RuntimeError('Requested cache capacity was not qualified locally')
        proofs.append(path)
    if options.loading_study:
        if options.reuse_reference or options.adaptive_only:
            raise RuntimeError('Loading study uses a fresh physical Native reference')
        for relative in ('semantic_results/loading_pipeline_offline/adaptive_result.json',
                         'semantic_results/rank_local_loader_final.json',
                         'semantic_results/loading_sampler_2/result.json',
                         options.wrapper_qualification_dir+'/sync/sync_result.json',
                         options.wrapper_qualification_dir+'/adaptive/adaptive_result.json',
                         options.wrapper_qualification_dir+'/offline/adaptive_result.json',
                         *([options.wrapper_qualification_dir+'/adaptive_p2_off/adaptive_result.json']
                           if options.p012_study else [])):
            path=ROOT/relative;result=json.loads(path.read_text())
            if not result['passed']:raise RuntimeError('Loading qualification failed: '+relative)
            for name,sha in result['source_sha256'].items():
                if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=sha:
                    raise RuntimeError('Loading source changed since execution: '+name)
            if 'loader_final' in relative and len(result['cases'])!=200:
                raise RuntimeError('All trained rank blocks must be qualified')
            if 'requests' in result:
                if len(result['requests'])!=2 or not all(c['is_bit_exact'] for r in result['requests'] for c in r['correctness'].values()):
                    raise RuntimeError('Offline cold/warm must be Native bit-exact')
            proofs.append(path)
    updates={}
    if options.reuse_reference:
        native=json.loads((ROOT/'kaggle_native_audit_reference/native_result.json').read_text())
        allowed={'patches/weight_residency.py':('ResidencyWindow','_room'),
                 'patches/comfy_tp_runtime.py':('PhysicalTPService','abort')}
        for name,old_sha in native['source_sha256'].items():
            current=(ROOT/name).read_bytes();new_sha=hashlib.sha256(current).hexdigest()
            if new_sha==old_sha:continue
            if name not in allowed:raise RuntimeError('Native operator source changed: '+name)
            old=(ROOT/'semantic_results/native_reference_memory_policy_before'/Path(name).name).read_bytes()
            if hashlib.sha256(old).hexdigest()!=old_sha:raise RuntimeError('Policy ancestor does not match Native reference')
            normalized=[]
            for raw in (old,current):
                tree=ast.parse(raw.decode('utf-8'));found=0
                for node in tree.body:
                    if isinstance(node,ast.ClassDef) and node.name==allowed[name][0]:
                        for method in node.body:
                            if isinstance(method,ast.FunctionDef) and method.name==allowed[name][1]:
                                method.body=[ast.Pass()];found+=1
                if found!=1:raise RuntimeError('Policy method scope is ambiguous')
                normalized.append(ast.dump(tree,include_attributes=False))
            if normalized[0]!=normalized[1]:raise RuntimeError('Changes outside qualified memory policy: '+name)
            updates[name]=dict(before=old_sha,after=new_sha,method='.'.join(allowed[name]),
                              native_execution_uses_method=False,unchanged_ast_outside_method=True)
    update_path=ROOT/'semantic_results/native_reference_policy_updates.json'
    update_path.write_text(json.dumps(updates,indent=2))
    specs=json.loads((ROOT/'semantic_results/full_checkpoint_identity.json').read_text())
    specs.update(video=dict(remote_file='vae/minimax_h3_video_vae_int8_convrot.safetensors',
        bytes=2811065184,sha256='52a2c8c73583c86e4f41cdcce3a6ad0ea562987bc0bf3d60a0cef5f5c8e60c0e'),
        audio=dict(remote_file='vae/minimax_h3_audio_vae_fp32.safetensors',bytes=605254808,
        sha256='8e505d95dd1561d47abd43d4238fd40d9bb1ae9e147ed0a4cba778d76ae4db48'))
    identity=ROOT/'semantic_results/pipeline_checkpoint_identity.json'
    identity.write_text(json.dumps(specs,indent=2))
    files=list((ROOT/'patches').glob('*.py'))+[ROOT/'benchmarks'/name for name in
        ('pipeline_residency_audit.py','full_models.py','fixture_inputs.py','retest.py',
         'logical_dual_gpu.py','qualify_comfy_sampler.py','ref2va_workloads.py',
         'monitor_pipeline_job.py','native_resident_reference.py','qualify_pipeline_depth4.py')]
    files+=[identity,*proofs,native_gate,ROOT/'semantic_results/local_allocator_environment.json',update_path]
    if options.loading_study:
        files += [ROOT/'benchmarks'/name for name in ('run_loading_audit.py',
                  'observe_loading_audit.py','qualify_loading_spawn.py','qualify_native_placement.py',
                  'qualify_rank_local_loader.py','qualify_loading_pipeline.py')]
        files += [ROOT/'analyze_loading_audit.py',ROOT/'analyze_pipeline_audit.py']
        spawn_proof=ROOT/'semantic_results/loading_spawn_qualification.json'
        if not json.loads(spawn_proof.read_text())['passed']:raise ValueError('Loading spawn qualification required')
        files.append(spawn_proof)
    if options.native_kernel_source:
        files.append(ROOT/'semantic_results/loading_launcher_repair.json')
    if transport:
        files += [ROOT/'benchmarks/offline_checkpoint_store.py',ROOT/'benchmarks/restore_offline_bundle.py',
                  ROOT/'semantic_results/offline_conversion_plan.json']
    buffer=io.BytesIO()
    with tarfile.open(fileobj=buffer,mode='w:gz') as archive:
        for path in files:
            if transport and path in proofs:
                # Keep qualification bindings/checks, omit large event histories.
                result=json.loads(path.read_text())
                summary={k:result[k] for k in ('passed','physical','layers','steps','source_sha256') if k in result}
                summary['original_artifact_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
                summary['requests']=[dict(temperature=r.get('temperature'),correctness=r['correctness'])
                                     for r in result.get('requests',[])]
                data=json.dumps(summary,indent=2).encode();member=tarfile.TarInfo(path.relative_to(ROOT).as_posix())
                member.size=len(data);archive.addfile(member,io.BytesIO(data))
            else:archive.add(path,arcname=path.relative_to(ROOT).as_posix())
        if transport:archive.add(options.offline_transport,arcname='offline_download_ticket.json')
        if options.native_reference_transport:archive.add(options.native_reference_transport,arcname='native_reference_download_ticket.json')
    payload=base64.b64encode(buffer.getvalue()).decode()
    code='''import base64,hashlib,io,json,os,pathlib,signal,subprocess,sys,tarfile,time,threading,shutil
root=pathlib.Path('/kaggle/working/pipeline_audit_source');root.mkdir(exist_ok=True)
out=pathlib.Path('/kaggle/working/pipeline_audit_results');out.mkdir(exist_ok=True)
with tarfile.open(fileobj=io.BytesIO(base64.b64decode(PAYLOAD)),mode='r:gz') as tar:tar.extractall(root,filter='data')
start=time.monotonic();setup_end=start+1200
manifest=dict(status='setup',setup_budget_seconds=1200,gpu_budget_seconds=2400,retry=False,
              scope='50-layer TE/DiT, Native reference encoding/vision, 124-frame 512x512 audiovisual sampling and full Native video/audio decode; four steps; cold/warm runtime caches')
def save():(out/'manifest.json').write_text(json.dumps(manifest,indent=2))
save()
def command(args,budget,logname):
    begin=time.monotonic()
    print('AUDIT STAGE BEGIN',logname,'budget_seconds',round(max(1,budget)),flush=True)
    with (out/logname).open('w') as log:
        proc=subprocess.Popen(args,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,start_new_session=True)
        def pump():
            for line in proc.stdout:
                log.write(line);log.flush()
                if line.startswith(('AUDIT','VERIFIED','Traceback','ERROR')):print(line.rstrip(),flush=True)
        reader=threading.Thread(target=pump);reader.start()
        stopped=threading.Event()
        def heartbeat():
            while not stopped.wait(30):
                memory={line.split(':')[0]:int(line.split()[1])*1024 for line in pathlib.Path('/proc/meminfo').read_text().splitlines() if line.startswith(('MemTotal:','MemAvailable:'))}
                snapshot=dict(stage=logname,seconds=round(time.monotonic()-begin),disk_free_gib=round(shutil.disk_usage('/tmp').free/2**30,2),memory=memory)
                (out/(logname+'.progress.json')).write_text(json.dumps(snapshot))
                print('AUDIT STAGE PROGRESS',json.dumps(snapshot),flush=True)
        watcher=threading.Thread(target=heartbeat,daemon=True);watcher.start()
        try:rc=proc.wait(timeout=max(1,budget))
        except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait();rc=124
        finally:stopped.set();watcher.join()
        if rc:
            try:os.killpg(proc.pid,signal.SIGKILL)
            except ProcessLookupError:pass
        reader.join()
    print('AUDIT STAGE END',logname,'return_code',rc,'seconds',round(time.monotonic()-begin,2),flush=True)
    if rc:raise RuntimeError(logname+' failed; return code '+str(rc))
    return time.monotonic()-begin
try:
    command([sys.executable,'-m','pip','install','--quiet','comfy-kitchen==0.2.37','comfy-aimdo==0.5.5','safetensors==0.8.0','einops','transformers','torchsde','sentencepiece','huggingface-hub==1.12.0','psutil','nvidia-ml-py','av>=17','pydantic-settings','kornia','alembic','sqlalchemy','simpleeval','blake3','comfy-angle'],setup_end-time.monotonic(),'dependencies.log')
    command([sys.executable,'-m','pip','install','--quiet','--upgrade','torch==2.11.0','torchvision==0.26.0','torchaudio==2.11.0','--index-url','https://download.pytorch.org/whl/cu130'],setup_end-time.monotonic(),'torch.log')
    command(['git','clone','--filter=blob:none','https://github.com/Comfy-Org/ComfyUI.git',str(root/'ComfyUI')],setup_end-time.monotonic(),'clone.log')
    command(['git','-C',str(root/'ComfyUI'),'checkout','d49e888586dd8ae012c0667b33466b815fee07f7'],setup_end-time.monotonic(),'checkout.log')
    os.environ.update(COMFY_ROOT=str(root/'ComfyUI'),NCCL_P2P_DISABLE='1',NCCL_IB_DISABLE='1',NCCL_DEBUG='INFO',NCCL_DEBUG_SUBSYS='INIT,GRAPH,COLL,P2P',NCCL_DEBUG_FILE=str(out/'nccl.%h.%p.log'),HF_HUB_DISABLE_XET='1',HF_HUB_DISABLE_PROGRESS_BARS='1',MALLOC_ARENA_MAX='2',MALLOC_TRIM_THRESHOLD_='67108864')
    manifest['allocator_environment']={k:os.environ[k] for k in ('MALLOC_ARENA_MAX','MALLOC_TRIM_THRESHOLD_')};save()
    prep="""import hashlib,json,pathlib,shutil,sys
from concurrent.futures import ThreadPoolExecutor
from huggingface_hub import hf_hub_download
root=pathlib.Path(ROOT_VALUE)
specs=json.loads((root/'semantic_results/pipeline_checkpoint_identity.json').read_text())
if shutil.disk_usage('/tmp').free<sum(s['bytes'] for s in specs.values())+8*2**30:raise RuntimeError('Insufficient disk headroom for four full checkpoints')
def download(item):
    kind,spec=item
    try:path=pathlib.Path(hf_hub_download('Comfy-Org/MiniMax-H3',spec['remote_file'],revision='e5eb578a89295337b8ff433a035929ce0279e0b6',cache_dir='/tmp/pipeline_models'))
    except Exception as e:raise RuntimeError('Checkpoint download failed: '+kind+' '+type(e).__name__) from None
    with path.open('rb') as f:sha=hashlib.file_digest(f,'sha256').hexdigest()
    if path.stat().st_size!=spec['bytes'] or sha!=spec['sha256']:raise RuntimeError('Full checkpoint identity mismatch: '+kind)
    print('VERIFIED full',kind,sha,flush=True);return kind,str(path)
with ThreadPoolExecutor(max_workers=2) as pool:paths=dict(pool.map(download,specs.items()))
(root/'checkpoint_paths.json').write_text(json.dumps(paths))
""".replace('ROOT_VALUE',repr(str(root)))
    command([sys.executable,'-u','-c',prep],setup_end-time.monotonic(),'download.log')
    probe="import json,pathlib,torch,psutil; devices=[dict(name=torch.cuda.get_device_name(i),bytes=torch.cuda.get_device_properties(i).total_memory) for i in range(torch.cuda.device_count())]; assert len(devices)==2 and all('T4' in d['name'] for d in devices); pathlib.Path(OUTPUT).write_text(json.dumps(dict(devices=devices,torch=torch.__version__,cuda=torch.version.cuda,ram=psutil.virtual_memory().total)))".replace('OUTPUT',repr(str(out/'hardware.json')))
    command([sys.executable,'-c',probe],setup_end-time.monotonic(),'hardware.log')
    manifest.update(status='running',setup_seconds=time.monotonic()-start,jobs=[]);save()
    gpu_end=time.monotonic()+2400
    for strategy,budget in [('native',900),('sync',750),('adaptive',750)]:
        args=[sys.executable,'-u',str(root/'benchmarks/pipeline_residency_audit.py'),'--physical','--strategy',strategy,'--steps','4','--checkpoints',str(root/'checkpoint_paths.json'),'--output',str(out)]
        if strategy!='native':args+=['--profile']
        args=[sys.executable,'-u',str(root/'benchmarks/monitor_pipeline_job.py'),
              '--output',str(out/(strategy+'_pcie.json')),'--',*args]
        duration=command(args,min(budget,gpu_end-time.monotonic()),strategy+'.log')
        result=json.loads((out/(strategy+'_result.json')).read_text())
        if not result.get('passed'):raise RuntimeError('Pipeline failed: '+strategy)
        manifest['jobs'].append(dict(strategy=strategy,seconds=duration,passed=True));save()
    manifest.update(status='complete',total_seconds=time.monotonic()-start);save()
except Exception as error:
    manifest.update(status='failed',error=type(error).__name__+': '+str(error),total_seconds=time.monotonic()-start);save();raise
finally:print('AUDIT FINAL',json.dumps(manifest),flush=True)
'''.replace('PAYLOAD',repr(payload))
    if options.reuse_reference:
        reuse="""    import shutil,requests
    references=list(pathlib.Path('/kaggle/input').rglob('native_result.json'))
    if len(references)!=1:raise RuntimeError('Expected exactly one completed Native input')
    previous=json.loads(references[0].read_text())
    if not (previous['passed'] and previous['physical'] and previous['layers']==50 and previous['steps']==4 and previous['width']==512 and previous['height']==512):raise RuntimeError('Native input workload is not qualified')
    updates=json.loads((root/'semantic_results/native_reference_policy_updates.json').read_text())
    for name,sha in previous['source_sha256'].items():
        actual=hashlib.sha256((root/name).read_bytes()).hexdigest()
        if actual!=sha:
            change=updates.get(name,{})
            if not (change.get('before')==sha and change.get('after')==actual and change.get('unchanged_ast_outside_method')):
                raise RuntimeError('Native input source mismatch: '+name)
    identity=json.loads((references[0].parent/'reference_identity.json').read_text())
    links=json.loads((references[0].parent/'reference_links.json').read_text())
    fetch_end=time.monotonic()+300
    (out/'native_reference').mkdir(exist_ok=True)
    for name,spec in identity['files'].items():
        if name=='native_result.json':source=references[0]
        else:
            source=out/'native_reference'/name
            try:
                with requests.get(links[name],stream=True,timeout=(15,30)) as response:
                    if response.status_code!=200:raise RuntimeError('Native reference HTTP '+str(response.status_code))
                    with source.open('wb') as stream:
                        for chunk in response.iter_content(65536):
                            if time.monotonic()>fetch_end:raise RuntimeError('Native reference transport exceeded 300 seconds')
                            stream.write(chunk)
            except requests.RequestException as error:raise RuntimeError('Native reference transport failed: '+type(error).__name__) from None
        if source.stat().st_size!=spec['bytes']:raise RuntimeError('Native artifact size mismatch: '+name)
        with source.open('rb') as stream:sha=hashlib.file_digest(stream,'sha256').hexdigest()
        if sha!=spec['sha256']:raise RuntimeError('Native artifact hash mismatch: '+name)
    shutil.copyfile(references[0],out/'native_result.json')
    shutil.copyfile(references[0].parent/'reference_identity.json',out/'reference_identity.json')
    manifest['reference_provenance']=dict(kernel='chinokiki9/minimax-h3-pipeline-residency-audit',version=3,native_complete=True,source_verified=True,qualified_memory_policy_updates=updates)
    print('AUDIT REUSED Native reference; operator sources unchanged, qualified memory-policy updates recorded',flush=True)
    save()
"""
        # Check the mounted artifact before dependency installs or downloads.
        code=code.replace("try:\n    command", "try:\n"+reuse+"    command",1)
        code=code.replace("    gpu_end=time.monotonic()+2400","    gpu_end=time.monotonic()+2500")
        code=code.replace("gpu_budget_seconds=2400","gpu_budget_seconds=2500")
        code=code.replace("[('native',900),('sync',750),('adaptive',750)]","[('sync',1250),('adaptive',1250)]")
    if options.adaptive_only:
        if not options.reuse_reference:raise RuntimeError('Adaptive-only continuation requires the completed Native reference')
        code=code.replace("[('sync',1250),('adaptive',1250)]","[('adaptive',1250)]")
        code=code.replace('gpu_budget_seconds=2500','gpu_budget_seconds=1250').replace('gpu_end=time.monotonic()+2500','gpu_end=time.monotonic()+1250')
    code=code.replace("'--steps','4','--checkpoints'", "'--steps','4','--cache-gib',"+repr(str(options.cache_gib))+",'--checkpoints'")
    if options.loading_study:
        conversion="""    conversion_start=time.monotonic()
    converter=\"\"\"import gc,json,pathlib,shutil,sys,time,torch
root=pathlib.Path(ROOT_VALUE);sys.path[:0]=[str(root),str(root/'benchmarks')]
import retest
from patches.windowed_tp import CheckpointReader,prepare_tp_block_legacy
from patches.rank_local_int8 import export_block,load_offline_block
torch.set_num_threads(4);torch.set_grad_enabled(False)
paths=json.loads((root/'checkpoint_paths.json').read_text());directory=pathlib.Path('/tmp/rank_local_tp2')
if shutil.disk_usage('/tmp').free<46*2**30:raise RuntimeError('Insufficient disk headroom for offline conversion')
start=time.monotonic()
for kind in ('te','dit'):
    reader=CheckpointReader(paths[kind])
    for index in range(50):
        for rank in (0,1):
            export_block(reader,kind,index,rank,directory,native_comparator=prepare_tp_block_legacy)
            module=load_offline_block(reader,kind,index,rank,directory)
            del module;gc.collect()
        if index%10==9:print('VERIFIED offline',kind,index+1,flush=True)
(root/'offline_conversion.json').write_text(json.dumps(dict(passed=True,cases=200,seconds=time.monotonic()-start,native_tensor_bit_exact=True)))
\"\"\".replace('ROOT_VALUE',repr(str(root)))
    command([sys.executable,'-u','-c',converter],600,'offline_conversion.log')
    manifest['conversion']=json.loads((root/'offline_conversion.json').read_text());save()
"""
        begin=code.index("    gpu_end=time.monotonic()+2400")
        end=code.index("    manifest.update(status='complete'",begin)
        code=code[:begin]+"""    gpu_end=time.monotonic()+LOADING_BUDGET
    reference=out/'native_reference'
    for label,strategy,loader,budget in LOADING_JOBS:
        job_out=out if strategy=='native' else out/label
        job_out.mkdir(exist_ok=True)
        if strategy!='native':(job_out/'native_reference').symlink_to(reference,target_is_directory=True)
        args=[sys.executable,'-u',str(root/'benchmarks/observe_loading_audit.py'),'--physical','--strategy',strategy,'--steps','4','--cache-gib',CACHE_VALUE,'--checkpoints',str(root/'checkpoint_paths.json'),'--output',str(job_out),'--loader',loader]
        if loader=='offline':args+=['--rank-checkpoint-dir','/tmp/rank_local_tp2']
        if strategy!='native':args+=['--profile']
        args=[sys.executable,'-u',str(root/'benchmarks/monitor_pipeline_job.py'),'--output',str(out/(label+'_pcie.json')),'--',*args]
        duration=command(args,min(budget,gpu_end-time.monotonic()),label+'.log')
        result=json.loads((job_out/(strategy+'_result.json')).read_text())
        if not result.get('passed'):raise RuntimeError('Pipeline failed: '+label)
        manifest['jobs'].append(dict(strategy=label,seconds=duration,passed=True));save()
""".replace('CACHE_VALUE',repr(str(options.cache_gib))).replace('LOADING_BUDGET',
            str(900+1200*(3 if options.offline_mode in ('convert','reuse') else 2))).replace('LOADING_JOBS',
            repr([('native','native','direct',900),('direct_sync','sync','direct',1200),
                  ('direct_adaptive','adaptive','direct',1200)]+
                 ([('offline_adaptive','adaptive','offline',1200)] if options.offline_mode in ('convert','reuse') else [])))+code[end:]
        if options.offline_mode=='convert':
            code=code.replace("try:\n    command", "def convert_offline():\n"+conversion+"\ntry:\n    command",1)
            code=code.replace("        job_out=out if strategy=='native' else out/label",
                "        if loader=='offline':\n            before=time.monotonic();convert_offline();gpu_end+=time.monotonic()-before\n"+
                "        job_out=out if strategy=='native' else out/label",1)
        code=code.replace('gpu_budget_seconds=2400',
            'gpu_budget_seconds='+str(900+1200*(3 if options.offline_mode in ('convert','reuse') else 2)))
        code=code.replace("retry=False,", "retry=False,offline_mode="+repr(options.offline_mode)+",")
        code=code.replace("four steps; cold/warm runtime caches'", "four steps; cold/warm runtime caches; direct benchmarks first; optional offline after direct results'")
    if transport:
        # Start while capabilities are fresh, before dependency installation.
        # Bind/reload only after direct measurements; there is no reconversion.
        concurrent="""    offline_error=[]
    ticket=json.loads((root/'offline_download_ticket.json').read_text())
    if min(row['expires'] for row in ticket['objects'].values()) < time.time()+1200:
        raise RuntimeError('Private offline download capabilities too old; renew before launch')
    required=sum(s['bytes'] for s in json.loads((root/'semantic_results/pipeline_checkpoint_identity.json').read_text()).values())+sum(s['bytes'] for s in ticket['objects'].values())+12*2**30
    if shutil.disk_usage('/tmp').free < required:
        raise RuntimeError('Insufficient disk for original/offline checkpoints and installation headroom')
    print('AUDIT OFFLINE PREFETCH START before dependency installation',flush=True)
    offline_args=[sys.executable,'-u',str(root/'benchmarks/restore_offline_bundle.py'),
        '--ticket',str(root/'offline_download_ticket.json'),'--directory','/tmp/rank_local_tp2',
        '--checkpoints',str(root/'checkpoint_paths.json'),'--evidence',str(out/'offline_restore.json')]
    def fetch_offline():
        try:command([*offline_args,'--download-only'],max(1,setup_end-time.monotonic()),'offline_download.log')
        except Exception as error:offline_error.append(type(error).__name__)
    offline_thread=threading.Thread(target=fetch_offline);offline_thread.start()
"""
        code=code.replace('try:\n    command', 'offline_thread=None\ntry:\n'+concurrent+'    command',1)
        code=code.replace("    command([sys.executable,'-u','-c',prep],setup_end-time.monotonic(),'download.log')",
            "    command([sys.executable,'-u','-c',prep],setup_end-time.monotonic(),'download.log')\n"+
            "    offline_thread.join()\n    if offline_error:raise RuntimeError('Offline download failed; see sanitized offline_download.log')",1)
        code=code.replace("finally:print('AUDIT FINAL',json.dumps(manifest),flush=True)",
            "finally:\n    if offline_thread is not None:offline_thread.join()\n    print('AUDIT FINAL',json.dumps(manifest),flush=True)",1)
        code=code.replace("'pip','install','--quiet'", "'pip','install','--no-cache-dir','--quiet'")
        code=code.replace("        job_out=out if strategy=='native' else out/label",
            "        if loader=='offline':\n"+
            "            binding=command([*offline_args,'--bind-only'],1200,'offline_binding.log');gpu_end+=binding\n"+
            "            manifest['offline_restore']=json.loads((out/'offline_restore.json').read_text());save()\n"+
            "        job_out=out if strategy=='native' else out/label",1)
        code=code.replace("start=time.monotonic();setup_end=start+1200","start=time.monotonic();setup_end=start+1800")
        code=code.replace('setup_budget_seconds=1200','setup_budget_seconds=1800')
        code=code.replace("+8*2**30", "+46*2**30")
    if options.p012_study:
        old_jobs=repr([('native','native','direct',900),('direct_sync','sync','direct',1200),
            ('direct_adaptive','adaptive','direct',1200),('offline_adaptive','adaptive','offline',1200)])
        new_jobs=repr([('native','native','direct',900),('direct_sync','sync','direct',1200),
            ('direct_p1','adaptive','direct',1200),('direct_p1_p2','adaptive','direct',1200),
            ('offline_p1_p2','adaptive','offline',1200)])
        code=code.replace(old_jobs,new_jobs,1)
        code=code.replace("        if loader=='offline':args+=['--rank-checkpoint-dir','/tmp/rank_local_tp2']",
            "        args+=['--cross-request-te','off' if label in ('native','direct_sync','direct_p1') else 'on']\n"+
            "        if loader=='offline':args+=['--rank-checkpoint-dir','/tmp/rank_local_tp2']",1)
        code=code.replace('gpu_budget_seconds=4500','gpu_budget_seconds=5700')
        code=code.replace('gpu_end=time.monotonic()+4500','gpu_end=time.monotonic()+5700')
        code=code.replace("    manifest.update(status='complete'",
            "    command([sys.executable,str(root/'analyze_loading_audit.py'),str(out)],300,'analysis.log')\n"+
            "    manifest.update(status='complete'",1)
        code=code.replace("    print('AUDIT FINAL',json.dumps(manifest),flush=True)",
            "    from zipfile import ZipFile,ZIP_DEFLATED\n"+
            "    with ZipFile(out/'summary_evidence.zip','w',ZIP_DEFLATED) as z:\n"+
            "        for p in out.rglob('*'):\n"+
            "            if p.is_file() and p.suffix in ('.json','.log','.md') and not p.name.endswith('_trace.json'):\n"+
            "                z.write(p,p.relative_to(out))\n"+
            "    print('AUDIT FINAL',json.dumps(manifest),flush=True)",1)
    if options.loading_study:
        anchor='    prep="""import hashlib,json,pathlib,shutil,sys'
        code=code.replace(anchor,
            "    command([sys.executable,'-u',str(root/'benchmarks/qualify_loading_spawn.py')],120,'loading_spawn.log')\n"+anchor,1)
    if options.native_kernel_source:
        reuse_kernel="""    previous_files=list(pathlib.Path('/kaggle/input').rglob('native_result.json'))
    if len(previous_files)!=1:raise RuntimeError('Expected one completed Native result in the prior kernel output')
    previous_path=previous_files[0];previous=json.loads(previous_path.read_text())
    if not (previous['passed'] and previous['physical'] and previous['layers']==50 and previous['steps']==4 and previous['width']==512 and previous['height']==512):raise RuntimeError('Prior Native workload is unqualified')
    repair=json.loads((root/'semantic_results/loading_launcher_repair.json').read_text())
    for name,sha in previous['source_sha256'].items():
        current=hashlib.sha256((root/name).read_bytes()).hexdigest()
        if current!=sha and not (name=='benchmarks/run_loading_audit.py' and repair['before']==sha and repair['after']==current and repair['passed']):
            raise RuntimeError('Native reference operator source changed: '+name)
    reference=out/'native_reference';reference.mkdir(exist_ok=True)
    identity={}
    for name in ('conditioning','video','audio','pixels','waveform'):
        source=previous_path.parent/'native_reference'/(name+'.pt')
        if not source.is_file():raise RuntimeError('Native tensor missing: '+name)
        destination=reference/source.name;shutil.copyfile(source,destination)
        with destination.open('rb') as stream:sha=hashlib.file_digest(stream,'sha256').hexdigest()
        identity[source.name]=dict(bytes=destination.stat().st_size,sha256=sha)
    shutil.copyfile(previous_path,out/'native_result.json')
    shutil.copyfile(root/'semantic_results/loading_launcher_repair.json',out/'loading_launcher_repair.json')
    manifest['reference_provenance']=dict(kernel=NATIVE_KERNEL,version=1,native_complete=True,source_verified=True,
        tensor_identity=identity,scope='Prior-session Native numerical reference; Native timing is not a paired same-session baseline')
    print('AUDIT REUSED completed Native tensors; all model operator sources verified',flush=True);save()
""".replace('NATIVE_KERNEL',repr(options.native_kernel_source))
        if options.native_reference_transport:
            fetch="""    import requests
    native_ticket=json.loads((root/'native_reference_download_ticket.json').read_text())
    native_input=root/'downloaded_native_reference';native_input.mkdir(exist_ok=True)
    deadline=time.monotonic()+300
    for name,spec in native_ticket['objects'].items():
        target=native_input/name if name=='native_result.json' else native_input/'native_reference'/name
        target.parent.mkdir(exist_ok=True);temporary=target.with_suffix(target.suffix+'.partial')
        digest=hashlib.sha256();size=0
        try:
            with requests.get(spec['url'],stream=True,timeout=(15,45)) as response:
                if response.status_code!=200:raise RuntimeError('Native output HTTP '+str(response.status_code))
                with temporary.open('wb') as stream:
                    for chunk in response.iter_content(2**20):
                        if time.monotonic()>deadline:raise RuntimeError('Native output download deadline exceeded')
                        stream.write(chunk);digest.update(chunk);size+=len(chunk)
        except requests.RequestException:raise RuntimeError('Native output transport failed; URL omitted') from None
        if size!=spec['bytes'] or digest.hexdigest()!=spec['sha256']:raise RuntimeError('Native output integrity mismatch: '+name)
        temporary.replace(target)
        print('VERIFIED saved Native output',name,flush=True)
    previous_files=[native_input/'native_result.json']
"""
            reuse_kernel=reuse_kernel.replace("    previous_files=list(pathlib.Path('/kaggle/input').rglob('native_result.json'))\n",fetch,1)
        if '    gpu_end=time.monotonic()+5700' not in code:raise ValueError('Unexpected P012 job layout')
        if 'try:\n    offline_error=[]' not in code:raise ValueError('Unexpected offline bootstrap layout')
        # Validate mounted evidence before installation or checkpoint traffic.
        code=code.replace('try:\n    offline_error=[]','try:\n'+reuse_kernel+'    offline_error=[]',1)
        code=code.replace('    gpu_end=time.monotonic()+5700','    gpu_end=time.monotonic()+4800',1)
        code=code.replace("[('native', 'native', 'direct', 900), ","[",1)
        code=code.replace('gpu_budget_seconds=5700','gpu_budget_seconds=4800')
    compile(code,'pipeline_audit','exec')
    if len(code.encode())>=1_000_000:raise ValueError('Kaggle source exceeds 1 MB')
    folder=ROOT/('kaggle_direct_loading_audit' if options.p012_study else
                 'kaggle_loading_audit' if options.loading_study else 'kaggle_pipeline_audit');folder.mkdir(exist_ok=True)
    notebook_metadata=json.loads((ROOT/'kaggle_comfy_sampler/sampler_gate.ipynb').read_text())['metadata']
    (folder/'pipeline_audit.ipynb').write_text(json.dumps(dict(nbformat=4,nbformat_minor=5,metadata=notebook_metadata,
        cells=[dict(id='pipeline-audit',cell_type='code',metadata={},execution_count=None,outputs=[],source=code.splitlines(keepends=True))])))
    metadata=json.loads((ROOT/'kaggle_comfy_sampler/kernel-metadata.json').read_text())
    metadata.update(id='chinokiki9/minimax-h3-pipeline-residency-audit',title='MiniMax H3 Pipeline Residency Audit',code_file='pipeline_audit.ipynb')
    if options.reuse_reference:
        metadata.update(id='chinokiki9/minimax-h3-pipeline-tp-audit',title='MiniMax H3 Pipeline TP Audit',
                        kernel_sources=[],dataset_sources=['chinokiki9/minimax-h3-native-audit-reference-v3'])
    if options.loading_study:
        metadata.update(id='chinokiki9/minimax-h3-rank-local-loading-audit',title='MiniMax H3 Rank-local Loading Audit',
                        kernel_sources=[],dataset_sources=[])
    if options.p012_study:
        metadata.update(id='chinokiki9/minimax-h3-direct-loading-audit',title='MiniMax H3 Direct Loading Audit')
    if options.native_kernel_source:
        metadata.update(id='chinokiki9/minimax-h3-p012-spawn-repaired-audit',
                        title='MiniMax H3 P012 Spawn Repaired Audit',
                        kernel_sources=[] if options.native_reference_transport else [options.native_kernel_source])
    (folder/'kernel-metadata.json').write_text(json.dumps(metadata,indent=2))
    print(folder)


if __name__=='__main__':main()
