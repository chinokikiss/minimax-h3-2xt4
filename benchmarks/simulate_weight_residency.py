"""Real CUDA residency/transport experiments on one physical 5060 Ti.

Local numbers describe diagnostics only. They are never dual-T4 performance.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT/'ComfyUI', ROOT/'.retest-deps', ROOT/'benchmarks'):
    sys.path.insert(0, str(path))

import torch
from torch import nn

import retest
from logical_dual_gpu import LogicalDualGPU, isolate_cublas_workspaces
from patches.weight_residency import ActivationHandoff, ResidencyWindow, WeightSpec, module_bytes, module_tensors
from patches.windowed_tp import CheckpointReader, tp_specs

PATHS = {'te': Path('D:/minimaxH3/qwen3vl_32b_minimax_h3_int8_convrot.safetensors'),
         'dit': Path('D:/minimaxH3/MiniMax_H3_Ref2VA_pruned_int8_convrot.safetensors'),
         'video': Path('D:/minimaxH3/minimax_h3_video_vae_int8_convrot.safetensors')}
GATE = {'rel_l2_max': 1e-5, 'cosine_min': .999999, 'max_abs_max': .005}
TARGET = ROOT/'semantic_results/weight_residency.json'


def persist(rows, **extra):
    files = [ROOT/'patches'/name for name in ['weight_residency.py', 'windowed_tp.py',
             'tp_qwen3vl.py', 'tp_dit.py', 'optimized_tp.py', 'parallel_int8.py']]
    files += [Path(__file__), ROOT/'benchmarks/logical_dual_gpu.py']
    retest.write_json(TARGET, {'passed': False, 'scope': 'One physical RTX 5060 Ti; real CUDA copies/operators; no dual-T4 performance evidence',
        'cases': rows, 'source_sha256': {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}, **extra})


def checked(rows, label, output, reference, **extra):
    score = retest.metrics(output, reference)
    score.update(label=label, passed=retest.passes(score, GATE), **extra)
    rows.append(score)
    persist(rows)
    print('RESIDENCY', label, score['passed'], score.get('is_bit_exact'), flush=True)
    if not score['passed']:
        raise RuntimeError(f'Native mismatch: {label}: {score}')


def transfer_stress(rows):
    producer, consumer = torch.cuda.Stream(), torch.cuda.Stream()
    mover = ActivationHandoff('cuda:0')
    for i in range(12):
        shape = (17 + i % 3, 128)
        with torch.cuda.stream(producer):
            # Queue real producer work so copying on the default stream cannot
            # accidentally substitute for the required event dependency.
            torch.cuda._sleep(1_000_000)
            source = torch.arange(shape[0]*shape[1], device='cuda:0').float().reshape(shape).add_(i)
            value = {'activation': source, 'metadata': (source[:, :3].clone(),)}
        copied = mover.copy(value, producer, consumer)
        assert copied['activation'].data_ptr() != source.data_ptr()
        del value, source
        # Allocate and write while consumer/copy ownership is still active.
        pressure = torch.empty(shape, device='cuda:0').fill_(-1000)
        with torch.cuda.stream(consumer):
            torch.cuda._sleep(1_000_000)
            result = copied['activation'] + copied['metadata'][0].sum(dim=-1, keepdim=True)
        consumer.synchronize()
        expected = torch.arange(shape[0]*shape[1]).float().reshape(shape).add_(i)
        expected = expected + expected[:, :3].sum(dim=-1, keepdim=True)
        checked(rows, f'delayed_copy_reuse_{i}', result.cpu(), expected)
        del copied, result, pressure
    mover.close()
    # Queue consumers without a fence per transfer, with a one-copy source
    # budget. This also exercises backpressure retaining the oldest owner.
    mover = ActivationHandoff('cuda:0', max_pending=1)
    outputs = []
    for i in range(8):
        with torch.cuda.stream(producer):
            torch.cuda._sleep(2_000_000)
            source = torch.full((128, 128), float(i+1), device='cuda:0')
        copied = mover.copy(source, producer, consumer)
        del source
        with torch.cuda.stream(consumer):
            torch.cuda._sleep(2_000_000)
            outputs.append(copied.square())
        del copied
    consumer.synchronize()
    for i, output in enumerate(outputs):
        checked(rows, f'queued_copy_backpressure_{i}', output.cpu(), torch.full((128,128), float((i+1)**2)))
    mover.close()


def small_module(index):
    module = nn.Linear(128, 128)
    with torch.no_grad():
        module.weight.copy_((torch.arange(128*128).reshape(128, 128) % 23 - 11).float() / (200 + index))
        module.bias.fill_(index/100.)
    return module.eval().requires_grad_(False)


def window_stress(rows):
    keys = ['te0', 'te1', 'dit0', 'dit1', 'vae0', 'vae1']
    specs = [WeightSpec(k, lambda i=i: small_module(i), 128*129*4) for i, k in enumerate(keys)]
    uses = ['te0', 'te1'] + ['dit0', 'dit1']*3 + ['vae0', 'vae1']
    x = torch.arange(8*128, device='cuda:0').float().reshape(8, 128)/1000
    ref = x.clone()
    for key in uses:
        model = small_module(keys.index(key)).cuda()
        ref = model(ref)
        del model
    torch.cuda.synchronize()
    for slots, adaptive in [(1, False), (2, False), (2, True)]:
        window = ResidencyWindow(specs, uses, 'cuda:0', max_weight_bytes=slots*specs[0].max_bytes,
                                 pinned_bytes=4*specs[0].max_bytes, lookahead=2, reserve_bytes=64*2**20,
                                 adaptive=adaptive)
        try:
            out = x.clone()
            for key in uses:
                with window.use(key) as model:
                    out = model(out)
                del model
            stats = window.finish()
            checked(rows, f'cross_model_fixture_slots{slots}_adaptive{adaptive}', out.cpu(), ref.cpu(), residency=stats)
            if slots == 2:
                assert stats['loads']['dit0'] == stats['loads']['dit1'] == 1, stats
                assert stats['peak_live_weight_bytes'] <= 2*specs[0].max_bytes
        finally:
            window.close()


def native_stack(kind, path, data, layers=2, steps=1):
    import te_dit_study as study
    x, args = data
    out = x.clone()
    for step in range(steps):
        for index in range(layers):
            block = retest.load_native_block(kind, path, index)
            if kind == 'te':
                out = study.invoke(kind, block, out, args, True)
            else:
                out = block(out.clone(), *args)
            torch.cuda.synchronize()
            del block
    return out.cpu()


def trained_stack(rows, kind, length, steps):
    import patches.tp_qwen3vl as te
    import patches.tp_dit as dit
    from patches.optimized_tp import enable
    import te_dit_study as study
    from ref2va_workloads import capture_inputs
    enable('scalemax')
    import comfy_kitchen as ck
    ck.backends.cuda._max_dynamic_shared_memory_per_block = lambda x: 49152
    isolate_cublas_workspaces()
    comm = LogicalDualGPU(timeout=180)
    te.dist = dit.dist = comm
    if kind == 'te':
        x, args = study.inputs('te', PATHS['te'], length, 1, torch.device('cuda:0'))
    else:
        fixture = ROOT/f'semantic_results/ref2va_inputs/s{length}_0.pt'
        captured = torch.load(fixture, weights_only=True) if fixture.exists() else capture_inputs(PATHS['dit'], length)
        x = captured['input'].cuda()
        t_emb, segments, rope = captured['args']
        args = (t_emb.cuda(), segments, rope.cuda())
        del captured
    reference = native_stack(kind, PATHS[kind], (x, args), steps=steps)
    reader = CheckpointReader(PATHS[kind])
    windows = []
    for rank in range(2):
        specs = tp_specs(reader, kind, rank, layers=2)
        uses = [s.key for s in specs]*steps
        windows.append(ResidencyWindow(specs, uses, 'cuda:0', max_weight_bytes=2*2**30,
            pinned_bytes=2*2**30, lookahead=2, reserve_bytes=256*2**20, adaptive=True))
    def run(rank):
        out = x.clone()
        window = windows[rank]
        for _ in range(steps):
            for key in window.uses[:2]:
                with window.use(key) as block:
                    if kind == 'te':
                        out = block(out, args[0], args[1])
                    else:
                        out = block(out.clone(), *args)
                del block
        return out
    try:
        outputs = comm.run([lambda r=r: run(r) for r in range(2)])
        for rank, output in enumerate(outputs):
            stats = windows[rank].finish()
            checked(rows, f'trained_{kind}_s{length}_layers2_steps{steps}_rank{rank}',
                    output.cpu(), reference, residency=stats)
            if steps > 1:
                assert all(n == 1 for n in stats['loads'].values()), stats
    finally:
        for window in windows:
            window.close()
    del outputs, x, args, reference
    retest.cleanup()


def video_transfer(rows):
    from ref2va_workloads import values
    path = PATHS['video']
    z = values((1, 24, 2, 4, 4), 2718, 'cuda:0', torch.float16)
    # Construct outside inference_mode: combining native QuantizedTensor.to
    # with dtype/device conversion inside that mode can materialize FP16
    # weights on this runtime. A dequantized comparator changes dispatch.
    with torch.inference_mode(False), torch.no_grad():
        native, state = retest.load_video_native(path, 'cuda:0')
    if not hasattr(native.decoder.transformer_blocks[0].attn.to_qkv.weight, '_qdata'):
        raise RuntimeError('Native VAE comparator lost its INT8 weight representation')
    del state
    raw_ref = native._decode_pixels(z).cpu()
    public_ref = native.decode(z).cpu()
    del native
    retest.cleanup()
    def prepare():
        model, state = retest.load_video_native(path, 'cpu')
        del state
        return model
    spec = WeightSpec('trained_video_full36', prepare, path.stat().st_size + 64*2**20)
    window = ResidencyWindow([spec], [spec.key, spec.key], 'cuda:0',
        max_weight_bytes=4*2**30, pinned_bytes=4*2**30, lookahead=1, reserve_bytes=256*2**20)
    try:
        with window.use(spec.key) as model:
            assert all(t.device.type == 'cuda' for t in module_tensors(model))
            raw = model._decode_pixels(z)
        checked(rows, 'trained_video36_payload_metadata_raw', raw.cpu(), raw_ref)
        with window.use(spec.key) as model:
            public = model.decode(z)
        stats = window.finish()
        checked(rows, 'trained_video36_public_temporal_normalization', public.cpu(), public_ref, residency=stats)
        assert stats['loads'][spec.key] == 1
    finally:
        window.close()
    del model, raw, public, z
    retest.cleanup()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--transport-only', action='store_true')
    parser.add_argument('--trained-only', action='store_true')
    parser.add_argument('--video-only', action='store_true')
    parser.add_argument('--long-matrix', action='store_true')
    parser.add_argument('--output', type=Path)
    options = parser.parse_args()
    global TARGET
    if options.output:
        TARGET = options.output
    rows = []
    persist(rows, status='running')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_grad_enabled(False)
    with torch.inference_mode():
        if not options.trained_only and not options.video_only:
            transfer_stress(rows)
            window_stress(rows)
        if not options.transport_only and not options.video_only:
            for length in ([512, 2048, 4096] if options.long_matrix else [512]):
                trained_stack(rows, 'te', length, 1)
            for length in ([8192, 16384, 32768] if options.long_matrix else [8192]):
                trained_stack(rows, 'dit', length, 2)
        if not options.transport_only and not options.trained_only:
            video_transfer(rows)
    persist(rows, passed=True, status='complete', gate=GATE,
            limitations=['Logical ranks share one GPU; physical NCCL/PCIe remains unverified',
                         'Trained TP coverage is two layers, not full TE/DiT or a sampler loop',
                         'Video transfer uses complete 36-layer public/raw decoder at compact geometry',
                         'Cross-model execution fixture checks scheduling; it is not a Ref2VA generation'])
    print('COMPLETE', len(rows), 'local CUDA comparisons', flush=True)


if __name__ == '__main__':
    main()
