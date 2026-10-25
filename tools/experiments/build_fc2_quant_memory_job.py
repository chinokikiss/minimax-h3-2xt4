"""Source-only, local-gated FC2 quantizer study on physical dual T4."""
import ast
import argparse
import hashlib
import json
from pathlib import Path
from tools.experiments.build_h3_reduction_job import plan
ROOT=Path(__file__).resolve().parents[2]


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--gate',default='semantic_results/fc2_quant_memory/local_final/results.json')
    parser.add_argument('--lengths',type=int,nargs='+',default=[32768]);args=parser.parse_args()
    proof=json.loads((ROOT/args.gate).read_text(encoding='utf8'))
    assert proof.get('passed') and proof.get('lifetime_exact') and proof.get('owned_exact'),'Local numerical qualification must pass'
    for name,sha in proof['source_sha256'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==sha,'Source changed after local qualification: '+name
    plans=dict(dit=plan('D:/minimaxH3/MiniMax_H3_Ref2VA_pruned_int8_convrot.safetensors',
                       'Comfy-Org/MiniMax-H3','e5eb578a89295337b8ff433a035929ce0279e0b6',
                       'diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors',lambda k:k.startswith('blocks.0.') or k=='adaln_t_table'),
               lora=plan('D:/minimaxH3/loras/dmad_minimax_h3_4step_full_critic_comfyui.safetensors',
                         'ZhengmingYu/DMAD','439136a1dd4f435b8137a084f73d558d2cecda2a',
                         'minimax_h3/dmad_minimax_h3_4step_full_critic_comfyui.safetensors',lambda k:k.startswith('diffusion_model.blocks.0.')))
    names=list(proof['source_sha256'])+['benchmarks/logical_dual_gpu.py','benchmarks/tp2_timing.py',
        'benchmarks/fetch_h3_physical_fixtures.py','benchmarks/checkpoint_ranges.py','benchmarks/retest.py',
        'benchmarks/fixture_inputs.py','benchmarks/merge_h3_lora.py','patches/windowed_tp.py','patches/weight_residency.py',
        'patches/host_weight_cache.py','patches/host_memory_pressure.py','patches/diagnostic_metrics.py',
        'patches/tp_dit.py','patches/tp_qwen3vl.py','patches/parallel_int8.py','patches/quantized_collectives.py',
        'patches/rank_local_int8.py','patches/h3_operator_policy.py']
    sources={name:(ROOT/name).read_text(encoding='utf8') for name in dict.fromkeys(names)}
    for name,source in sources.items():ast.parse(source,filename=name)
    hashes={name:hashlib.sha256(source.encode()).hexdigest() for name,source in sources.items()}
    code='''import pathlib,sys,subprocess,os,signal,json,time,traceback
ROOT=pathlib.Path('/kaggle/working/h3_source');ROOT.mkdir(exist_ok=True)
OUT=pathlib.Path('/kaggle/working/fc2_quant_results');OUT.mkdir(exist_ok=True)
deadline=time.monotonic()+1140
state=dict(status='setup',source_only=True,commands=[])
def save():(OUT/'status.json').write_text(json.dumps(state,indent=2))
def run(argv,label,cap):
    timeout=min(cap,deadline-time.monotonic())
    if timeout<10:raise TimeoutError('Study deadline exhausted')
    print('FC2_STAGE_START',label,flush=True);start=time.monotonic()
    with (OUT/(label+'.log')).open('w') as stream:
        process=subprocess.Popen(argv,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        try:code=process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid,signal.SIGTERM)
            try:process.wait(timeout=6)
            except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait(timeout=5)
            raise TimeoutError(label)
    state['commands'].append(dict(label=label,exit_code=code,seconds=time.monotonic()-start));save()
    print('FC2_STAGE_END',label,state['commands'][-1],flush=True)
    if code:
        print((OUT/(label+'.log')).read_text(errors='replace')[-7000:],flush=True)
        raise RuntimeError(label)
'''
    for name,source in sources.items():code+=f'p=ROOT/{name!r};p.parent.mkdir(parents=True,exist_ok=True);p.write_text({source!r})\n'
    code+=f"(ROOT/'fixture_plans.json').write_text(json.dumps({plans!r},indent=2))\n"
    code+=f"(OUT/'source_hashes.json').write_text(json.dumps({hashes!r},indent=2))\n"
    code+=f"(OUT/'local_gate.json').write_text(json.dumps({proof!r},indent=2))\n"
    code+='''save()
try:
    os.environ.update(OMP_NUM_THREADS='2',NCCL_P2P_DISABLE='1',NCCL_IB_DISABLE='1',NCCL_DEBUG='WARN',
        NCCL_MIN_NCHANNELS='4',NCCL_MAX_NCHANNELS='4',PYTHONUNBUFFERED='1',HF_HUB_DISABLE_XET='1',HF_HUB_DISABLE_PROGRESS_BARS='1')
    os.environ.pop('NCCL_PROTO',None)
    run([sys.executable,'-m','pip','install','--quiet','--no-cache-dir','comfy-kitchen==0.2.37','tilelang==0.1.15',
        'safetensors==0.8.0','huggingface-hub==1.12.0','einops','transformers','torchsde','sentencepiece','psutil'],'dependencies',180)
    run([sys.executable,'-m','pip','install','--quiet','--no-cache-dir','--upgrade','torch==2.11.0','torchvision==0.26.0','torchaudio==2.11.0',
        '--index-url','https://download.pytorch.org/whl/cu130'],'runtime',360)
    run(['git','clone','--filter=blob:none','https://github.com/Comfy-Org/ComfyUI.git',str(ROOT/'ComfyUI')],'comfy_clone',90)
    run(['git','-C',str(ROOT/'ComfyUI'),'checkout','d49e888586dd8ae012c0667b33466b815fee07f7'],'comfy_pin',45)
    run([sys.executable,'-u',str(ROOT/'benchmarks/fetch_h3_physical_fixtures.py'),str(ROOT/'fixture_plans.json'),str(ROOT/'fixtures')],'fixtures',180)
    state['status']='physical_study';save()
    run([sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=2',str(ROOT/'benchmarks/fc2_quant_memory.py'),
        '--physical','--checkpoint',str(ROOT/'fixtures/dit.safetensors'),'--lora',str(ROOT/'fixtures/lora.safetensors'),
        '--output',str(OUT),LENGTH_ARGUMENTS],'dual_t4',600)
    result=json.loads((OUT/'results.json').read_text());state.update(status=result['status'],passed=result.get('passed'),
        lifetime_exact=result.get('lifetime_exact'),owned_exact=result.get('owned_exact'))
finally:
    for name in ('dit.safetensors','lora.safetensors'):
        path=ROOT/'fixtures'/name
        if path.is_file():path.unlink()
    save();print('FC2_FINAL',state['status'],flush=True)
'''
    # Keep exception evidence even when the notebook catches a bootstrap failure.
    code=code.replace('finally:\n    for name',"except BaseException as error:\n    state.update(status='failed',error_type=type(error).__name__);(OUT/'exception.txt').write_text(traceback.format_exc())\nfinally:\n    for name")
    code=code.replace('LENGTH_ARGUMENTS',"'--lengths',"+','.join(repr(str(n)) for n in args.lengths))
    ast.parse(code)
    folder=ROOT/'kaggle_h3-fc2-quant-memory';folder.mkdir(exist_ok=True)
    notebook=dict(nbformat=4,nbformat_minor=5,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}},
                  cells=[dict(id='fc2-quant-memory',cell_type='code',metadata={},execution_count=None,outputs=[],source=code.splitlines(keepends=True))])
    (folder/'experiment.ipynb').write_text(json.dumps(notebook),encoding='utf8')
    (folder/'kernel-metadata.json').write_text(json.dumps(dict(id='ocnsai/h3-fc2-quantizer-memory',title='H3 FC2 Quantizer Memory',code_file='experiment.ipynb',
        language='python',kernel_type='notebook',is_private=True,enable_gpu=True,enable_tpu=False,enable_internet=True,
        dataset_sources=[],competition_sources=[],kernel_sources=[],model_sources=[]),indent=2),encoding='utf8')
    print('SOURCE_ONLY',folder,'REMOTE_FIXTURE_BYTES',sum(p['bytes'] for p in plans.values()))


if __name__=='__main__':main()
