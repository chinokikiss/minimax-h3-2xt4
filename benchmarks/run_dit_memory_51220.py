"""One DiT-only NFE with exact target packing; no TE, VAE, sampler or fixes.

Inputs are deterministic synthetic tensors with the failed L1 shapes/segments.
Trained base checkpoint weights are unchanged (not the offline-LoRA variant).
This profiles allocation mechanics, not the original trajectory or accuracy.
"""
import argparse
from dataclasses import asdict
import gzip
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def boot(source, cpu=False):
    sys.path[:0] = [str(source), str(source/'ComfyUI'), str(source/'benchmarks')]
    sys.argv = ['dit-memory', '--cpu'] if cpu else ['dit-memory']
    import comfy.options
    comfy.options.enable_args_parsing()
    import comfy.cli_args
    from patches.legacy_runtime import validate_packages, validate_source
    validate_packages()
    validate_source(source/'ComfyUI')


def traced_functions():
    from patches import tp_dit, h3_tensor_flow, tilelang_swiglu_quant, tilelang_transport_p2
    from comfy_kitchen.backends.eager import quantization
    functions=dict(DiT=tp_dit.TPDiTBlockINT8.forward,
                KitchenEagerLinear=quantization.int8_linear,
                ModulatedNorm=h3_tensor_flow.modulated_norm,
                ResidualAdd=h3_tensor_flow.gated_add,
                FusedSwiGLUQuant=tilelang_swiglu_quant.prepare,
                TransportEncode=tilelang_transport_p2.encode,
                TransportCollect=tilelang_transport_p2.collect,
                TransportEpilogue=tilelang_transport_p2.finish)
    if hasattr(tp_dit,'run_int8_linear_forward'):
        from importlib.util import find_spec
        if find_spec('patches.dit_column_linear') is not None:
            from patches.dit_column_linear import _eager_column
            functions['DiTColumnLinear']=_eager_column
    return functions


def profile_worker(*args):
    import torch
    import patches.comfy_tp_runtime as rt
    import ref2va_e2e_run as observation
    from dit_memory_lines import LineMarkers, source_bindings, checkpoint_history
    # Keep original forwards, helper mathematics and module ownership intact.
    functions = traced_functions()
    class ProfileRank(observation.ObservedRank):
        def __init__(self, *values):
            super().__init__(*values)
            import comfy_kitchen as ck
            (self.memory_audit.directory/'source_lines.json').write_text(json.dumps(
                dict(functions=source_bindings(functions),
                     kitchen_cuda_extension=ck.backends.cuda._EXT_AVAILABLE,
                     kitchen_cublaslt=ck.backends.cuda._CUBLASLT_AVAILABLE), indent=2))

        def dispatch(self, command, args=(), kwargs=None):
            if command in ('refine', 'denoise'):
                def checkpoint(label):
                    # Persist both ranks before entering FC1, so a peer OOM
                    # and normal TP-service abort cannot erase earlier phases.
                    checkpoint_history(self.memory_audit,label)
                with LineMarkers(self.memory_audit, functions, checkpoint,
                                 representative_only=os.environ.get('H3_LIFETIME_PROFILE')=='1'):
                    return super().dispatch(command, args, kwargs)
            return super().dispatch(command, args, kwargs)
    rt.RankRuntime = ProfileRank
    rt.ResidencyWindow = observation.ObservedWindow
    args = (*args[:-1], observation.MemoryConnection(args[-1]))
    observation.BASE_WORKER(*args)


def cpu_preflight(source, output):
    boot(source, cpu=True)
    import gc
    import weakref
    import torch
    from dit_memory_lines import LineMarkers
    from gpu_memory_overhead import MemoryAudit
    from analyze_gpu_memory_overhead import analyze
    from types import SimpleNamespace
    from comfy.ldm.minimax.model import PackedLayout
    layout = PackedLayout(1182, 107, 32, 56, 603, refs=[
        dict(kind='image', latent_t=1, latent_h=32, latent_w=56, ref_audio_t=0),
        dict(kind='image', latent_t=1, latent_h=32, latent_w=56, ref_audio_t=0)])
    assert layout.seq_len == 51220 and layout.segments[-1] == (3284, 51220, 'video')
    marks = []
    audit = SimpleNamespace(active=True, mark=lambda label, **kw: marks.append(label))
    def example():
        value = torch.ones(7)
        handle = weakref.ref(value)
        del value
        return handle
    with LineMarkers(audit, {'test': example}):
        handle = example()
    gc.collect()
    assert handle() is None and len(marks) > 1 and sys.gettrace() is None
    # The existing collector imports without constructing a CUDA object.
    assert callable(MemoryAudit.begin) and callable(analyze)
    assert all(hasattr(fn,'__code__') for fn in traced_functions().values())
    receipt = dict(status='profile_cpu_ready', tests=['native_51220_layout',
                   'line_observer_retains_no_tensor', 'collector_analyzer_and_all_phase_helpers_import'],
                   cuda_allocation=False, driver_sha256=sha(Path(__file__)),
                   line_observer_sha256=sha(Path(__file__).with_name('dit_memory_lines.py')),
                   analyzer_sha256=sha(Path(__file__).with_name('analyze_gpu_memory_overhead.py')))
    (output/'profile_qualification.json').write_text(json.dumps(receipt, indent=2))
    print('DIT_MEMORY_CPU_PASS', receipt['tests'], flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cpu-preflight', action='store_true')
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--lifetime-scheduling',action='store_true',help='Qualify current lifetime-only changes before the DiT profile')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.cpu_preflight:
        cpu_preflight(args.source, args.output)
        return
    boot(args.source)
    import torch
    from patches import comfy_tp_runtime as rt
    from patches.h3_operator_policy import H3OperatorPolicy
    from patches.h3_tensor_flow import TensorFlowPolicy
    from ref2va_e2e_run import Monitor
    assert torch.cuda.device_count() == 2 and all(torch.cuda.get_device_capability(i) == (7, 5) for i in (0, 1))
    torch.set_grad_enabled(False)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    import shutil,subprocess
    compiler=Path(shutil.which('nvcc')).resolve()
    nvcc_version=subprocess.check_output([str(compiler),'--version'],text=True,timeout=10)
    assert 'release 12.8' in nvcc_version
    os.environ.update(CUDACXX=str(compiler),CUDA_HOME=str(compiler.parent.parent),CUDA_PATH=str(compiler.parent.parent))
    os.environ.update(MINIMAX_H3_GPU_MEMORY_PROFILE='1',
                      MINIMAX_H3_MEMORY_DIR=str(args.output/'gpu_memory'))
    checkpoint = args.checkpoint
    state = dict(status='starting', workload='One DiT-only forward, not an H3 pipeline',
                 expected_tokens=51220, optimizer_changes=args.lifetime_scheduling,
                 input_distribution='Seeded synthetic; exact failed-L1 tensor shapes/packing, not captured real activations',
                 weights='Original trained base INT8 ConvRot; no online/offline LoRA in this diagnostic',
                 stages={}, source_sha256={Path(__file__).name:sha(Path(__file__)),
                    'dit_memory_lines.py':sha(Path(__file__).with_name('dit_memory_lines.py'))})
    state['hardware']=[dict(name=torch.cuda.get_device_name(i),capability=list(torch.cuda.get_device_capability(i)),
                           uuid=str(torch.cuda.get_device_properties(i).uuid)) for i in (0,1)]
    state['torch'],state['cuda'],state['compiler']=torch.__version__,torch.version.cuda,nvcc_version
    def save():
        (args.output/'result.json').write_text(json.dumps(state, indent=2, default=str))
    def stage(name, fn):
        state['status'] = name
        save()
        start = time.monotonic()
        print('DIT_MEMORY_STAGE_START', name, flush=True)
        try:
            value = fn()
            state['stages'][name] = dict(status='complete', seconds=time.monotonic()-start)
            return value
        except BaseException:
            state['stages'][name] = dict(status='failed', seconds=time.monotonic()-start)
            raise
        finally:
            save()
    save()
    if checkpoint is None:
        from huggingface_hub import hf_hub_download
        checkpoint = Path(stage('immutable_checkpoint_download', lambda: hf_hub_download(
            'Comfy-Org/MiniMax-H3', 'diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors',
            revision='e5eb578a89295337b8ff433a035929ce0279e0b6',
            cache_dir='/kaggle/temp/dit_memory_model')))
    state['checkpoint_sha256'] = sha(checkpoint)
    assert state['checkpoint_sha256'] == '9255f52b6677845ad238f20dfaafa94727053694127ab7f255c048f0f9365779'
    if args.lifetime_scheduling:
        os.environ['H3_LIFETIME_PROFILE']='1'
        subprocess.run([sys.executable,'-u',str(args.source/'benchmarks/preview_dit_lifetimes.py'),
            '--checkpoint',str(checkpoint),'--output',str(args.output/'column_gate.json'),
            '--require-t4','--samples','3'],check=True,timeout=180)
        state['column_gate']=json.loads((args.output/'column_gate.json').read_text())
    empty = args.output/'unused_te.safetensors'
    from safetensors.torch import save_file
    save_file({}, str(empty))
    config = rt.RuntimeConfig(str(empty), str(checkpoint), layers=50,
        weight_bytes=6*2**30, pinned_bytes=6*2**30, cache_bytes=3*2**30,
        host_bytes=16*2**30, reserve_bytes=9*2**30, timeout_seconds=300,
        prefetch_lifecycle=True, memory_pressure_aware=True, cross_model_cpu=True,
        idle_residency=False, lora_stack=(), operators=H3OperatorPolicy(dit_attention='tilelang_scalar'),
        tensor_flow=TensorFlowPolicy('fp16'))
    state['runtime'] = asdict(config)
    rt._worker = profile_worker
    service = None
    monitor = Monitor()
    try:
        service = stage('service_start', lambda: rt.PhysicalTPService(config))
        service.call('begin_sample')
        generator = torch.Generator().manual_seed(20261020)
        text = torch.randn(1, 1182, 5120, generator=generator)
        context = stage('refiner', lambda: service.call('refine', text))
        state['after_refine'] = service.call('stats')
        video = torch.randn(1, 24, 107, 32, 56, generator=generator)
        audio = torch.randn(1, 32, 2, 603, generator=generator)
        refs = [dict(kind='image', latent_t=1, latent_h=32, latent_w=56, ref_audio_t=0,
                     latent=torch.randn(1, 24, 1, 32, 56, generator=generator)) for _ in range(2)]
        payload = dict(seed=20261020, audio_scale=1., refs=refs,
                       cond_video_latents=[ref['latent'] for ref in refs], cond_audio_latents=[])
        output = stage('dit_51220', lambda: service.call('denoise', [video, audio],
            torch.tensor([1000.]), context=context, minimax_payload=payload,
            transformer_options={'minimax_h3_sigma_shift_video':12., 'minimax_h3_sigma_shift_audio':2.}))
        del output
        state['ranks'] = service.call('stats')
        state['status'] = 'profile_completed'
    except BaseException as error:
        state.update(status='profile_captured_failure', error=repr(error))
        (args.output/'exception.txt').write_text(traceback.format_exc())
    finally:
        if service:
            try: service.close()
            except Exception as error: state['close_error'] = repr(error)
        monitor.close()
        with gzip.open(args.output/'nvml_samples.json.gz', 'wt') as stream:
            json.dump(monitor.rows, stream)
        save()
    import subprocess
    subprocess.run([sys.executable, str(Path(__file__).with_name('analyze_gpu_memory_overhead.py')),
                    '--directory',str(args.output/'gpu_memory')], check=True, timeout=120)
    indices = list((args.output/'gpu_memory').glob('rank*/index.json'))
    assert len(indices) == 2, 'Missing a rank memory history'
    coverage={}
    for rank in (0,1):
        captures=[]
        for path in (args.output/'gpu_memory'/f'rank{rank}').glob('*.json.gz'):
            with gzip.open(path,'rt') as stream:payload=json.load(stream)
            if payload['command']=='denoise':
                captures.append(dict(file=path.name,checkpoint=payload.get('checkpoint',False),
                    events=len(payload['events']),history_limit_reached=payload['history_limit_reached']))
        coverage[str(rank)]=captures
    state['denoise_history_coverage']=coverage;save()
    assert all(coverage.values()),'Missing denoise history or crash-safe prefix on a rank'
    print('DIT_MEMORY_EVIDENCE_READY', state['status'], flush=True)


if __name__ == '__main__':
    main()
