"""Targeted physical benchmarks of locally simulated repaired algorithms."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import threading
import time

from retest import (ROOT, GATES, cleanup, digest_tensor, distributed_run, environment,
                    load_native_block, load_video_native, measure, metrics, native_call,
                    validate_runtime, write_json)
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--suite', choices=['te', 'dit', 'video'], required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--fixtures', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--budget-seconds', type=int, default=180)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    args.warmups, args.iterations, args.round, args.allow_non_t4 = 1, 5, 0, False
    args.raw_video = args.suite == 'video'
    args.equivalence_gate = {'rel_l2_max': 1e-5, 'cosine_min': .999999, 'max_abs_max': .005}
    result = {'status': 'preflight', 'suite': args.suite, 'workloads': {},
              'scope': 'full trained 36-block raw video decoder' if args.raw_video else
                       'trained checkpoint block 0; no full-model latency extrapolation',
              'warmups': 1, 'samples_per_mode': 5, 'budget_seconds': args.budget_seconds,
              'equivalence_gate': args.equivalence_gate}
    target = args.output / 'results.json'
    write_json(target, result)
    def expired():
        write_json(args.output / 'budget_exceeded.json', {'seconds': args.budget_seconds})
        os._exit(124)
    timer = threading.Timer(args.budget_seconds, expired)
    timer.daemon = True
    timer.start()
    try:
        validate_runtime()
        if torch.cuda.device_count() != 2 or any('T4' not in torch.cuda.get_device_name(i) for i in range(2)):
            raise RuntimeError('Requires two physical T4 GPUs')
        environment(args.output, args, str(args.checkpoint))
        specs = json.loads((args.fixtures / 'fixture_manifest.json').read_text())
        expected = specs['checkpoints'][args.suite]
        with args.checkpoint.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        if digest != expected['sha256']:
            raise RuntimeError('Checkpoint differs from the locally validated fixture')
        result['checkpoint'] = expected
        result['source_sha256'] = {
            p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted((ROOT / 'patches').glob('*.py'))}
        input_dir = args.output / 'inputs'
        input_dir.mkdir(exist_ok=True)
        references = {}
        model = (load_video_native(args.checkpoint, 'cuda:0')[0] if args.raw_video else
                 load_native_block(args.suite, args.checkpoint))
        for row in specs['inputs'][args.suite]:
            data = torch.load(args.fixtures / row['file'], weights_only=True, map_location='cpu')
            if digest_tensor(data['input']) != row['input_sha256']:
                raise RuntimeError('Input bytes differ from the local simulation')
            name = row['name']
            torch.save(data, input_dir / f'{name}.pt')
            x = data['input'].cuda().to(torch.float32 if args.suite == 'dit' else torch.float16)
            t = data.get('t_emb')
            t = t.cuda() if t is not None else None
            rope = data.get('rope')
            rope = rope.to(device='cuda:0', dtype=x.dtype) if rope is not None else None
            call = ((lambda: model._decode_pixels(x)) if args.raw_video else
                    (lambda: native_call(args.suite, model, x, t, data.get('segments'), rope)))
            measured, ref = measure(call, [0], 1, 5)
            references[name] = ref
            torch.save(ref, args.output / f'native_{name}_output.pt')
            result['workloads'][name] = {'shape': row['shape'], 'input_sha256': row['input_sha256'],
                                         'native': measured}
            write_json(target, result)
            print(f"{args.suite}/native {name}: {measured['median_ms']:.2f}ms", flush=True)
            del x, t, rope, data
        del model
        cleanup()
        result['status'] = 'running'
        for mode in ['tp', 'sp']:
            # Each spawned worker validates native equivalence before warmup,
            # reduces agreement across ranks and checks output again afterward.
            values = distributed_run(mode, args.suite, args.checkpoint, input_dir, args.output, args)
            for name, measured in values.items():
                measured['input_sha256'] = result['workloads'][name]['input_sha256']
                measured['speedup_vs_same_allocation_native'] = (
                    result['workloads'][name]['native']['median_ms'] / measured['median_ms'])
                result['workloads'][name][mode] = measured
            write_json(target, result)
        result['status'] = 'complete'
        write_json(target, result)
    except Exception as exc:
        result.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        write_json(target, result)
        raise
    finally:
        timer.cancel()


if __name__ == '__main__':
    main()
