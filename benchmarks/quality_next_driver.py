"""Paired DiT quality screen using immutable native conditioning/noise inputs."""
import hashlib
import json
from pathlib import Path
import random
import shutil
import subprocess
import sys


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def prerequisites(root, out):
    schedule = json.loads(Path('/kaggle/working/generation_schedule.json').read_text())
    matches = [p.parent for p in Path('/kaggle/input').rglob('benchmark_override_hashes.json')
               if (p.parent/'native_seed20261013/conditioning.pt').exists()]
    if len(matches) != 1:
        raise RuntimeError('Exactly one complete V4 native input source required')
    prior = matches[0]
    if digest(prior/'status.json') != schedule['v4_status_sha256']:
        raise RuntimeError('Mounted V4 notebook output version changed')
    if json.loads((prior/'benchmark_override_hashes.json').read_text()) != schedule['v4_override_hashes']:
        raise RuntimeError('Mounted V4 source identity changed')
    fixed = json.loads((root/'bench_assets/tokyo_ref2va.json').read_text())
    fields = ['prompt','negative_prompt','width','height','frames','fps','steps','cfg',
              'sampler','scheduler','denoise','turbo_lora','ref_image_size','reference_image']
    inputs = {}
    for seed in schedule['seeds']:
        cfg = json.loads((prior/f'native_seed{seed}.json').read_text())
        if cfg['seed'] != seed or any(cfg[k] != fixed[k] for k in fields):
            raise RuntimeError('Prior native workload differs from fixed paired workload')
        directory = prior/f'native_seed{seed}'
        state = json.loads((directory/'run_state.json').read_text())
        if state['status'] != 'complete' or state['asset_sha256'] != digest(root/fixed['reference_image']):
            raise RuntimeError('Prior native generation/reference identity is invalid')
        inputs[str(seed)] = {'directory':str(directory),'sha256':{
            name:digest(directory/name) for name in ['conditioning.pt','initial_noise.pt','sigmas.pt']}}
    provenance = {'prior_directory':str(prior),'prior_status_sha256':digest(prior/'status.json'),
                  'native_inputs':inputs,'scope':'Cached native TE/reference inputs; no TE speed claim'}
    (out/'paired_input_provenance.json').write_text(json.dumps(provenance,indent=2))
    return schedule, provenance


def run_schedule(root, out, workload, run, state, save):
    schedule, provenance = prerequisites(root,out)
    seeds = schedule['seeds']
    policies = {'native':{},'sage32_dit':{'dit':{'mode':'sage32'}},
                'group32_dit':{'dit':{'mode':'group32_fused','kernel_variant':schedule['group32_variant']}},
                'drop_tau0_dit':{'dit':{'mode':'drop','tau':0.}}}
    nsys = shutil.which('nsys') or next((str(p) for p in Path('/opt/nvidia').rglob('nsys') if p.is_file()),None)
    state.update(status='paired_quality_screen',runs=[],optional_failures=[],nsys=nsys,
                 performance_scope='Instrumented cached-input DiT generation, serial resident 2-device placement; not TP2/request throughput',
                 group32_variant=schedule['group32_variant'])
    save()

    def optional(argv, name, timeout):
        try:
            run(argv,name,timeout)
            return True
        except (RuntimeError,subprocess.TimeoutExpired) as error:
            state['optional_failures'].append({'stage':name,'type':type(error).__name__})
            save()
            return False

    def invoke(config,name,profile=False):
        cfg = out/(name+'.json');cfg.write_text(json.dumps(config,indent=2))
        state.update(current_run=name);save()
        argv = [sys.executable,'-u',str(root/'benchmarks/quality_first_generation.py'),'--config',str(cfg)]
        if profile and nsys:
            trace = out/(name+'_step09')
            argv = [nsys,'profile','--trace=cuda,nvtx,osrt','--sample=none','--cpuctxsw=none',
                    '--capture-range=cudaProfilerApi','--capture-range-end=stop','--force-overwrite=true',
                    '-o',str(trace),*argv]
        run(argv,name,3600)

    with (out/'telemetry.csv').open('w') as log:
        monitor = subprocess.Popen(['nvidia-smi','--query-gpu=timestamp,index,clocks.sm,pstate,utilization.gpu,memory.used,power.draw',
                                    '--format=csv','-lms','500'],stdout=log)
        try:
            # TE/reference execute once per challenging seed, before candidate order is randomized.
            challenge = workload | schedule['challenge']
            common = {}
            for seed in seeds:
                name = f'challenge_inputs_seed{seed}';directory = out/name
                config = challenge | {'seed':seed,'output_directory':str(directory),'attention_policy':{},
                    'prepare_inputs_only':True,'profile_attention':False,'qualify_resident_loading':False}
                invoke(config,name)
                if json.loads((directory/'run_state.json').read_text())['status']!='inputs_prepared':
                    raise RuntimeError('Native challenging inputs were not prepared')
                common[seed] = directory
            # Fixed workload only adds Group32; prior V4 audiovisual baselines remain separately identified.
            jobs = [('fixed',seed,'group32_dit') for seed in seeds]
            for seed in seeds:
                modes = list(policies);random.Random(20261015+seed).shuffle(modes)
                jobs.extend(('challenge',seed,mode) for mode in modes)
            state['execution_order'] = jobs;save()
            for family,seed,mode in jobs:
                name = f'{family}_{mode}_seed{seed}';directory = out/name
                source = Path(provenance['native_inputs'][str(seed)]['directory']) if family=='fixed' else common[seed]
                profile = seed==seeds[0]
                config = (workload if family=='fixed' else challenge) | {
                    'attention_policy':policies[mode],'seed':seed,'output_directory':str(directory),
                    'reuse_native_inputs':str(source),'qualify_resident_loading':False,'qualify_grouped':mode=='group32_dit',
                    'profile_attention':True,'nsys_denoise_step':9 if profile and nsys else None,
                    'cpu_profile_step':9 if profile else None}
                invoke(config,name,profile)
                if json.loads((directory/'run_state.json').read_text())['status']!='complete' or not (directory/'audio.wav').exists():
                    raise RuntimeError('Incomplete audiovisual generation '+name)
                state['runs'].append({'name':name,'workload':family,'mode':mode,'seed':seed,
                    'status':'generated','quality':'unrated','profiled_step':9 if profile and nsys else None})
                save()
                optional([sys.executable,str(root/'benchmarks/quality_first_video_metrics.py'),'--video',str(directory/'native_video.mp4'),
                    '--reference',str(root/workload['reference_image']),'--prompt',config['prompt'],
                    '--output',str(directory/'video_metrics.json')],name+'_flow_metrics',180)
        finally:
            monitor.terminate();monitor.wait(timeout=15)
    if nsys:
        for path in sorted(out.glob('*_step09*.nsys-rep')):
            optional([nsys,'export','--type','sqlite','--force-overwrite=true','-o',str(path.with_suffix('.sqlite')),str(path)],
                     'export_'+path.stem,300)
            optional([nsys,'stats','--force-export=true','--report','cuda_gpu_kern_sum,cuda_api_sum,nvtx_sum','--format','csv',str(path)],
                     'stats_'+path.stem,300)
    state.update(status='screen_generated',quality_status='AV clips generated; metrics and blinded review pending',
                 profiling_status='partial' if state['optional_failures'] or not nsys else 'collected',
                 timing_exclusions='Denoiser call0 includes loading/JIT/qualification; profiled call9 has observer overhead; no independent steady request repetitions')
    save()
