"""Compare activation flows to the existing TP2 FP32 flow, preserving operators.

No dense-FP32 override, attention replacement, weight dequantization override,
sampler, dependency change, or T4 performance claim. Vision/Video keep their
existing FP16 stages; Audio keeps its existing FP32 stage.
"""
import argparse
from contextlib import contextmanager
from dataclasses import asdict
import gc
import hashlib
import json
from pathlib import Path
import sys
import threading
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'ComfyUI'), str(ROOT/'benchmarks')]
import torch
from patches.h3_tensor_flow import TensorFlowPolicy
from patches.h3_operator_policy import H3OperatorPolicy
from patches.windowed_tp import CheckpointReader, prepare_tp_block
from run_h3_tensor_flow import stats, compare, save, observe, capture_pack


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cleanup():
    gc.collect()
    torch.cuda.empty_cache()


def int8_identity(models):
    result = {}
    for rank, model in enumerate(models):
        for name, value in model.named_parameters():
            value = getattr(value, '_qdata', value)
            if value.dtype == torch.int8:
                result[f'{rank}:{name}'] = hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
    if not result:
        raise ValueError('Expected the production INT8 weights')
    return result


@contextmanager
def int8_calls():
    """Count calls while forwarding unchanged arguments to the installed library."""
    import comfy_kitchen as ck
    original = ck.int8_linear
    native_op = torch.ops.comfy_kitchen.int8_linear
    counts = {}
    lock = threading.Lock()

    def forward(x, weight, *args, **kwargs):
        if weight.dtype != torch.int8:
            raise ValueError('The existing INT8 GEMM must keep INT8 weights')
        with lock:
            key = str(x.dtype)
            counts[key] = counts.get(key, 0) + 1
        return original(x, weight, *args, **kwargs)

    def native_forward(x, weight, *args, **kwargs):
        if weight.dtype != torch.int8:
            raise ValueError('Native INT8 custom op must retain INT8 weights')
        with lock:
            key = 'custom_op:' + str(x.dtype)
            counts[key] = counts.get(key, 0) + 1
        return native_op(x, weight, *args, **kwargs)

    with patch.object(ck, 'int8_linear', forward), patch.object(torch.ops.comfy_kitchen, 'int8_linear', native_forward):
        yield counts


def inputs(args):
    if args.mode == 'dit':
        fixture = torch.load(args.fixture, map_location='cpu', weights_only=False)
        ids = (torch.arange(fixture['input'].shape[0]) if not args.tokens else
               torch.linspace(0, fixture['input'].shape[0]-1, args.tokens).long().unique())
        x = fixture['input'][ids].cuda().float()
        time = fixture['t_emb'].cuda().float()
        rope = fixture['rope'][:, ids].cuda().float()
        segments = []
        for start, end, row in fixture['segments']:
            selected = ((ids >= start) & (ids < end)).nonzero().flatten()
            if selected.numel():
                segments.append((int(selected[0]), int(selected[-1])+1,
                                 row if isinstance(row, int) else row[ids[selected]-start].cuda()))

        def call(model, value):
            return model(value.clone(), time, segments, rope.to(value.dtype), {})

        info = dict(source=str(args.fixture), sha256=digest(args.fixture),
                    provenance=fixture['provenance'], shape=list(x.shape), layer=args.layer,
                    caveat='Retained first-block input. Other layers reuse it as a stress probe, not their actual trajectory.')
    else:
        from fixture_inputs import block_input
        from comfy.text_encoders.llama import precompute_freqs_cis, Qwen3VL_32BConfig
        x = block_input('te', args.tokens or 257, 20261021)['input'].cuda().float()
        config = Qwen3VL_32BConfig()
        rope = precompute_freqs_cis(128, torch.arange(x.shape[1], device=x.device)[None], config.rope_theta,
                                  rope_dims=config.rope_dims, device=x.device, interleaved_mrope=True)

        def call(model, value):
            mask = torch.full((value.shape[1], value.shape[1]), torch.finfo(value.dtype).min/4,
                              device=value.device, dtype=value.dtype).triu_(1)
            return model(value.clone(), mask, rope)

        info = dict(source='Seeded synthetic, half-rounded TE input; not a real TE activation capture',
                    seed=20261021, shape=list(x.shape), layer=args.layer)
    return x, call, info


def blocks(args, report):
    from patches import tp_dit, tp_qwen3vl
    from patches.optimized_tp import enable
    from logical_dual_gpu import LogicalDualGPU, isolate_cublas_workspaces
    reader = CheckpointReader(args.checkpoint)
    if args.lora:
        from patches.h3_lora import H3LoRAStore
        reader.lora_store = H3LoRAStore(((str(args.lora), 1.),), reader, selective_b=True)
    reader.operator_policy = H3OperatorPolicy(row_linear_backend='tilelang_packed',fc2_quantizer='owned_fp32')
    report['operator_policy'] = asdict(reader.operator_policy)
    x, call, report['distribution'] = inputs(args)
    owner = tp_dit if args.mode == 'dit' else tp_qwen3vl
    enable('scalemax')
    isolate_cublas_workspaces()
    comm = LogicalDualGPU(timeout=180)
    configs = [('native_fp32_flow', TensorFlowPolicy()),
               ('bf16_flow', TensorFlowPolicy('bf16_reference', 0)),
               ('fp16_unscaled', TensorFlowPolicy('fp16', 0)),
               ('fp16_fold_c16', TensorFlowPolicy('fp16', 4)),
               ('fp16_fold_c64', TensorFlowPolicy('fp16', 6))]
    gold = None
    identities = None
    for name, policy in configs:
        reader.tensor_flow = policy
        models = [prepare_tp_block(reader, args.mode, args.layer, rank).cuda() for rank in (0, 1)]
        if any(getattr(m, '_full_precision_mm', False) for model in models for m in model.modules()):
            raise ValueError('No full-precision GEMM override is allowed')
        current = int8_identity(models)
        if identities is None:
            identities = current
            report['int8_weight_sha256'] = identities
        elif current != identities:
            raise ValueError('Activation-flow changes must preserve INT8 weight bytes')
        divisor = policy.divisor if policy.enabled else 1.
        dtype = policy.dtype if policy.enabled else torch.float32
        value = (x/divisor).to(dtype)
        with patch.object(owner, 'dist', comm), observe(models[0]) as proof, int8_calls() as counts:
            comm.run([lambda rank=rank: call(models[rank], value) for rank in (0, 1)])
            actual = comm.run([lambda rank=rank: call(models[rank], value) for rank in (0, 1)])
            repeat = comm.run([lambda rank=rank: call(models[rank], value) for rank in (0, 1)])
        physical = (actual[0].float()*divisor).cpu()
        if gold is None:
            if not torch.isfinite(physical).all():
                raise ValueError('Native FP32-flow reference is nonfinite')
            gold = physical
        record = dict(name=name, role='reference' if name == 'native_fp32_flow' else 'candidate',
                      stored_dtype=str(dtype), divisor=divisor, output=stats(physical),
                      vs_native_fp32_flow=compare(physical, gold),
                      rank_equal=torch.equal(actual[0], actual[1]),
                      repeat_equal=torch.equal(actual[0], repeat[0]),
                      int8_weight_bytes_equal=True, int8_linear_calls=counts, operators=proof)
        if name == 'native_fp32_flow' and (not record['rank_equal'] or not record['repeat_equal']):
            raise ValueError('Reference ranks/repeats disagree')
        if not counts:
            raise ValueError('No production INT8 GEMM observed')
        report['cases'].append(record)
        save(args.output, report)
        print(name, record['vs_native_fp32_flow'], flush=True)
        del models, actual, repeat, physical
        cleanup()


def frontend(args, report):
    from patches.comfy_tp_models import native_dit
    source = Path('D:/attention_eval/quality_constrained_20261015/quality_challenge_v1/generation_screen/challenge_inputs_seed20261013')
    context_path = source/'conditioning.pt'
    context = torch.load(context_path, map_location='cpu', weights_only=False)['positive'][0][0].cuda().float()
    report['distribution'] = dict(source=str(context_path), sha256=digest(context_path),
        caveat='Common retained conditioning/noise/reference; no fresh complete TE or denoising trajectory.')
    reader = CheckpointReader(args.checkpoint)
    gold = None
    # Preserve this checkpoint's dense BF16 frontend weights and existing FP32
    # factory; unlike the main DiT blocks, these stages are not INT8.
    frontend_entries = {name: value['dtype'] for name, value in reader.header.items()
                        if name.startswith(('condition_proj.', 'token_refiner.', 'video_patch_proj.', 'audio_patch_proj.'))}
    report['frontend_checkpoint_dtypes'] = sorted(set(frontend_entries.values()))
    if 'I8' in report['frontend_checkpoint_dtypes']:
        raise ValueError('Review frontend dispatch evidence for this different checkpoint')
    for name, policy in [('native_fp32_flow', TensorFlowPolicy()),
                         ('bf16_flow', TensorFlowPolicy('bf16_reference', 0)),
                         ('fp16_fold_c64', TensorFlowPolicy('fp16', 6))]:
        reader.tensor_flow = policy
        model = native_dit(reader).cuda()
        with observe(model) as proof, int8_calls() as counts:
            refined = model.preprocess_text_embeds(context)
            packed = capture_pack(model, refined, source, policy.dtype if policy.enabled else torch.float32)
        physical = [refined.cpu(), packed.float().cpu()*(policy.divisor if policy.enabled else 1.)]
        if gold is None:
            gold = physical
        report['cases'].append(dict(name=name, role='reference' if gold is physical else 'candidate',
            output=[stats(v) for v in physical], vs_native_fp32_flow=[compare(v, g) for v, g in zip(physical, gold)],
            int8_linear_calls=counts, native_frontend_dense_weights=True, operators=proof,
            boundaries=['refined_context', 'packed_input']))
        print(name, report['cases'][-1]['vs_native_fp32_flow'], flush=True)
        del model, refined, packed, physical
        cleanup()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['dit', 'te', 'frontend'])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--tokens', type=int, default=513)
    parser.add_argument('--layer', type=int, default=0)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--lora', type=Path)
    parser.add_argument('--fixture', type=Path, default=Path('D:/attention_eval/fp16_residual_fix_20261020/first_block.pt'))
    args = parser.parse_args()
    sys.argv = sys.argv[:1]
    if args.output.exists():
        raise FileExistsError('Historical results must not be overwritten')
    from patches.legacy_runtime import validate_source, validate_packages
    validate_source(ROOT/'ComfyUI')
    validate_packages()
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError('Local BF16-capable GPU required')
    torch.set_grad_enabled(False)
    if args.checkpoint is None:
        args.checkpoint = Path('D:/minimaxH3')/('qwen3vl_32b_minimax_h3_int8_convrot.safetensors' if args.mode == 'te'
                                             else 'MiniMax_H3_Ref2VA_pruned_int8_convrot.safetensors')
    report = dict(schema='native-fp32-flow-v1', status='running',
        reference='Existing legacy TP2 FP32 tensor flow, existing INT8 GEMMs/attention, normal mixed-precision islands retained',
        unchanged_stages='Vision/Video native FP16, Audio native FP32; no forced all-FP32 conversion',
        scope='Local SM120 component numerics with logical TP2 transport only; no full H3, T4/NCCL or speed claim',
        hardware=torch.cuda.get_device_name(), capability=torch.cuda.get_device_capability(), torch=torch.__version__,
        tf32_matmul=torch.backends.cuda.matmul.allow_tf32, tf32_cudnn=torch.backends.cudnn.allow_tf32,
        configuration=dict(mode=args.mode, tokens=args.tokens, layer=args.layer, checkpoint=str(args.checkpoint),
                           lora=str(args.lora) if args.lora else None),
        runtime_lock_sha256=digest(ROOT/'runtime/local_conda.json'), cases=[])
    try:
        with torch.inference_mode():
            (frontend if args.mode == 'frontend' else blocks)(args, report)
        report['status'] = 'complete'
    except Exception as exc:
        report['status'] = 'failed'
        report['error'] = repr(exc)
        raise
    finally:
        sources = ['benchmarks/run_h3_native_fp32_flow.py', 'benchmarks/run_h3_tensor_flow.py',
                   'benchmarks/logical_dual_gpu.py', 'patches/h3_tensor_flow.py', 'patches/h3_operator_policy.py',
                   'patches/tp_dit.py', 'patches/tp_qwen3vl.py', 'patches/h3_lora.py',
                   'patches/comfy_tp_models.py', 'patches/optimized_tp.py', 'patches/parallel_int8.py']
        report['source_sha256'] = {name: digest(ROOT/name) for name in sources}
        save(args.output, report)


if __name__ == '__main__':
    main()
