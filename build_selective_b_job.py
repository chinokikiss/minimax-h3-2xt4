"""Source-only isolated Selective B study; never edits production or checkpoints."""
import ast,hashlib,json,sys
from pathlib import Path
from build_h3_reduction_job import plan
ROOT=Path(__file__).resolve().parent


def main():
    proof=json.loads((ROOT/'semantic_results/selective_b_local_final/results.json').read_text(encoding='utf8'))
    if not proof.get('passed') or not proof.get('production_unchanged'):raise RuntimeError('Local feasibility validation must pass first')
    for name,sha in proof['source_sha256'].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=sha:raise RuntimeError('Source changed after local gate: '+name)
    initial=json.loads((ROOT/'semantic_results/selective_b_integrity_before.json').read_text(encoding='utf8'))
    for name,sha in initial['production_source_sha256'].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=sha:raise RuntimeError('Protected production source changed')
    prefixes=('blocks.0.','blocks.25.','blocks.49.')
    plans=dict(dit=plan('D:/minimaxH3/MiniMax_H3_Ref2VA_pruned_int8_convrot.safetensors',
        'Comfy-Org/MiniMax-H3','e5eb578a89295337b8ff433a035929ce0279e0b6',
        'diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors',lambda k:k.startswith(prefixes) or k=='adaln_t_table'),
        lora=plan('D:/minimaxH3/loras/dmad_minimax_h3_4step_full_critic_comfyui.safetensors',
        'ZhengmingYu/DMAD','439136a1dd4f435b8137a084f73d558d2cecda2a',
        'minimax_h3/dmad_minimax_h3_4step_full_critic_comfyui.safetensors',lambda k:k.startswith(tuple('diffusion_model.'+p for p in prefixes))))
    names=['benchmarks/selective_b_prototype.py','benchmarks/selective_b_audit.py','benchmarks/logical_dual_gpu.py',
        'benchmarks/physical_h3_reduction.py','benchmarks/fetch_h3_physical_fixtures.py','benchmarks/checkpoint_ranges.py','benchmarks/fixture_inputs.py','benchmarks/retest.py','benchmarks/merge_h3_lora.py',
        'patches/optimized_tp.py','patches/parallel_int8.py','patches/quantized_collectives.py','patches/h3_lora.py',
        'patches/windowed_tp.py','patches/weight_residency.py','patches/diagnostic_metrics.py','patches/tp_dit.py','patches/tp_qwen3vl.py','patches/rank_local_int8.py',
        'patches/comfy_tp_runtime.py','custom_nodes/minimax_h3_tp2/__init__.py']
    sources={name:(ROOT/name).read_text(encoding='utf8') for name in names}
    for name,text in sources.items():ast.parse(text,filename=name)
    hashes={name:hashlib.sha256(text.encode()).hexdigest() for name,text in sources.items()}
    code='''import pathlib,subprocess,sys,os,time,json,signal,traceback
ROOT=pathlib.Path('/kaggle/working/h3_source');ROOT.mkdir(exist_ok=True)
OUT=pathlib.Path('/kaggle/working/selective_b_results');OUT.mkdir(exist_ok=True)
deadline=time.monotonic()+1740
state=dict(status='setup',source_only=True,scope='Isolated Selective B feasibility, no production integration',commands=[])
def save():(OUT/'status.json').write_text(json.dumps(state,indent=2))
def run(argv,label,cap):
    seconds=min(cap,deadline-time.monotonic())
    if seconds<10:raise TimeoutError('Study budget exhausted')
    print('SELECTIVE_STAGE_START',label,flush=True);start=time.monotonic()
    with (OUT/(label+'.log')).open('w') as stream:
        p=subprocess.Popen(argv,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        try:code=p.wait(timeout=seconds)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid,signal.SIGTERM)
            try:p.wait(timeout=6)
            except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait(timeout=5)
            raise TimeoutError('Stage timed out: '+label)
    state['commands'].append(dict(label=label,exit_code=code,seconds=time.monotonic()-start));save()
    print('SELECTIVE_STAGE_END',label,state['commands'][-1],flush=True)
    if code!=0:
        print((OUT/(label+'.log')).read_text(errors='replace')[-7000:],flush=True)
        raise RuntimeError('Stage failed: '+label)
'''
    for name,text in sources.items():code+=f'p=ROOT/{name!r};p.parent.mkdir(parents=True,exist_ok=True);p.write_text({text!r})\n'
    code+=f"(ROOT/'fixture_plans.json').write_text(json.dumps({plans!r},indent=2))\n"
    code+=f"(OUT/'source_hashes.json').write_text(json.dumps({hashes!r},indent=2))\n"
    code+=f"(OUT/'local_gate.json').write_text(json.dumps({proof!r},indent=2))\n"
    code+='''save()
try:
    os.environ.update(OMP_NUM_THREADS='2',NCCL_P2P_DISABLE='1',NCCL_IB_DISABLE='1',NCCL_DEBUG='WARN',PYTHONUNBUFFERED='1',
                      HF_HUB_DISABLE_XET='1',HF_HUB_DISABLE_PROGRESS_BARS='1',COMFY_ROOT=str(ROOT/'ComfyUI'))
    run([sys.executable,'-m','pip','install','--quiet','--no-cache-dir','comfy-kitchen==0.2.37','comfy-aimdo==0.5.5',
        'safetensors==0.8.0','huggingface-hub==1.12.0','einops','transformers','torchsde','sentencepiece','psutil'],'dependencies',240)
    run([sys.executable,'-m','pip','install','--quiet','--no-cache-dir','--upgrade','torch==2.11.0','torchvision==0.26.0','torchaudio==2.11.0',
        '--index-url','https://download.pytorch.org/whl/cu130'],'runtime',480)
    run(['git','clone','--filter=blob:none','https://github.com/Comfy-Org/ComfyUI.git',str(ROOT/'ComfyUI')],'comfy_clone',120)
    run(['git','-C',str(ROOT/'ComfyUI'),'checkout','d49e888586dd8ae012c0667b33466b815fee07f7'],'comfy_pin',60)
    run([sys.executable,'-u',str(ROOT/'benchmarks/fetch_h3_physical_fixtures.py'),str(ROOT/'fixture_plans.json'),str(ROOT/'fixtures')],'fixtures',360)
    state['status']='physical_study';save()
    run([sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=2',str(ROOT/'benchmarks/selective_b_audit.py'),
        '--physical','--checkpoint',str(ROOT/'fixtures/dit.safetensors'),'--lora',str(ROOT/'fixtures/lora.safetensors'),
        '--output',str(OUT),'--profile'],'dual_t4',600)
    result=json.loads((OUT/'results.json').read_text())
    state.update(status=result['status'],passed=result.get('passed'),production_unchanged=result.get('production_unchanged'))
    for name in ('dit.safetensors','lora.safetensors'):
        path=ROOT/'fixtures'/name
        if path.is_file():path.unlink()
except BaseException as error:
    state.update(status='failed',error_type=type(error).__name__);(OUT/'exception.txt').write_text(traceback.format_exc())
finally:save();print('SELECTIVE_FINAL',state['status'],flush=True)
'''
    ast.parse(code)
    folder=ROOT/'kaggle_h3-selective-b-feasibility';folder.mkdir(exist_ok=True)
    notebook=dict(nbformat=4,nbformat_minor=5,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}},
        cells=[dict(id='selective-b-isolated',cell_type='code',metadata={},execution_count=None,outputs=[],source=code.splitlines(keepends=True))])
    (folder/'experiment.ipynb').write_text(json.dumps(notebook),encoding='utf8')
    (folder/'kernel-metadata.json').write_text(json.dumps(dict(id='chinokiki9/h3-selective-b-feasibility',title='H3 Selective B Feasibility',
        code_file='experiment.ipynb',language='python',kernel_type='notebook',is_private=True,enable_gpu=True,enable_tpu=False,enable_internet=True,
        dataset_sources=[],competition_sources=[],kernel_sources=[],model_sources=[]),indent=2),encoding='utf8')
    (folder/'fixture_plans.json').write_text(json.dumps(plans,indent=2),encoding='utf8')
    print('SOURCE_ONLY_JOB',folder,'fixture_download_bytes',sum(p['bytes'] for p in plans.values()))


if __name__=='__main__':main()
