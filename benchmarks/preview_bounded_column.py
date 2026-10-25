"""Trained FC1/real-input-template memory screen, separate from full H3.

The template is RMS-normalized and repeated to the observed L1 length. It is a
shape/distribution proxy, not a captured FC1 input or denoising trajectory.
Local SM120 results never qualify T4. Use --require-t4 for a future CPU-qualified
deployment experiment; this script neither installs packages nor submits jobs.
"""
import argparse
from contextlib import nullcontext
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
import comfy_kitchen as ck
from safetensors import safe_open
from benchmarks.bounded_column_int8 import bounded_int8_linear, preallocated_eager_int8_linear
from patches.parallel_int8 import packed_rows


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def measure(fn, samples):
    times, peaks = [], []
    torch.cuda.synchronize()
    start = time.perf_counter()
    out = fn()
    torch.cuda.synchronize()
    first = time.perf_counter() - start
    del out
    for _ in range(2):
        out = fn()
        del out
    for _ in range(samples):
        torch.cuda.synchronize()
        before = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        out = fn()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1000)
        peaks.append(torch.cuda.max_memory_allocated() - before)
        del out
    return dict(first_call_seconds=first, warmups=2, samples_ms=times,
                median_ms=statistics.median(times), peak_increment_bytes=peaks)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--assets', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--tokens', type=int, default=51220)
    parser.add_argument('--chunk-rows', type=int, default=4096)
    parser.add_argument('--samples', type=int, default=3)
    parser.add_argument('--backend', choices=('eager', 'auto'), default='eager')
    parser.add_argument('--require-t4', action='store_true')
    args = parser.parse_args()
    assert args.tokens > 0 and args.samples > 0
    torch.set_grad_enabled(False)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    device = dict(name=torch.cuda.get_device_name(), capability=list(torch.cuda.get_device_capability()))
    if args.require_t4:
        assert device['capability'] == [7, 5] and 'T4' in device['name']
    from patches.legacy_runtime import validate_packages
    validate_packages()
    result = dict(status='starting', device=device, torch=torch.__version__,
                  kitchen=version('comfy-kitchen'), tilelang=version('tilelang'),
                  scope='Operator memory/equality screen only; no H3 pipeline or quality acceptance',
                  t4_qualified=False, requested_backend=args.backend,
                  cuda_extension_available=ck.backends.cuda._EXT_AVAILABLE,
                  kitchen_cublaslt_available=ck.backends.cuda._CUBLASLT_AVAILABLE)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    def save():
        args.output.write_text(json.dumps(result, indent=2))
    save()
    try:
        checkpoint = args.assets / 'dit_blocks.safetensors'
        fixture_path = args.assets / 'first_block.pt'
        fixture = torch.load(fixture_path, map_location='cpu', weights_only=False)
        template = torch.nn.functional.rms_norm(fixture['input'].float(), (5376,)).half()
        ids = torch.arange(args.tokens) % template.shape[0]
        x = template[ids].cuda()
        del fixture, template
        with safe_open(checkpoint, framework='pt', device='cpu') as source:
            w, scale = packed_rows(source.get_tensor('blocks.0.mlp.fc1.weight'),
                                   source.get_tensor('blocks.0.mlp.fc1.weight_scale'), 0, 2, 2)
        w, scale = w.cuda(), scale.cuda()
        result.update(shape=[*x.shape, w.shape[0]], input_template_sha256=digest(fixture_path),
                      weight_fixture_sha256=digest(checkpoint),
                      input_scope='RMS-normalized retained first-block input, repeated rows; not actual long-trajectory FC1 input',
                      row_chunk=args.chunk_rows, fp16_output_bytes=x.shape[0] * w.shape[0] * 2,
                      full_int32_result_bytes=x.shape[0] * w.shape[0] * 4,
                      bounded_int32_result_bytes=min(args.chunk_rows, x.shape[0]) * w.shape[0] * 4)
        kwargs = dict(x=x, weight=w, weight_scale=scale, bias=None, out_dtype=torch.float16,
                      convrot=True, convrot_groupsize=256, input_act=None, input_act_weight=None,
                      input_act_eps=0., residual=None, residual_scale=None)
        with ck.registry.use_backend('eager') if args.backend == 'eager' else nullcontext():
            result['selected_backend'] = ck.registry.get_capable_backend('int8_linear', kwargs)
            baseline = lambda: ck.int8_linear(x, w, scale, None, torch.float16, convrot=True, convrot_groupsize=256)
            bounded = lambda: bounded_int8_linear(x, w, scale, chunk_rows=args.chunk_rows)
            torch.cuda.empty_cache()
            result['baseline'] = measure(baseline, args.samples)
            expected = baseline().cpu()
            functions = {'bounded': bounded}
            if result['selected_backend'] == 'eager':
                functions['preallocated_eager'] = lambda: preallocated_eager_int8_linear(x, w, scale)
            for name, fn in functions.items():
                torch.cuda.empty_cache()
                result[name] = measure(fn, args.samples)
                actual = fn()
                torch.cuda.synchronize()
                different = nonfinite = 0
                for start in range(0, args.tokens, args.chunk_rows):
                    value = actual[start:start + args.chunk_rows].cpu()
                    reference = expected[start:start + args.chunk_rows]
                    different += int(torch.count_nonzero(value != reference))
                    nonfinite += int(torch.count_nonzero(~torch.isfinite(value)))
                del actual
                result[name].update(unequal_values=different, nonfinite_values=nonfinite)
                save()
            del expected
        accepted = all(result[n]['unequal_values'] == result[n]['nonfinite_values'] == 0 for n in functions)
        result.update(status='passed' if accepted else 'numerical_gate_failed',
                      t4_qualified=args.require_t4 and accepted)
        save()
        print(json.dumps({k: result[k] for k in ('status', 'device', 'shape', 'selected_backend', 't4_qualified')}))
        assert result['status'] == 'passed'
    except BaseException as error:
        result.update(status='failed', error=repr(error))
        save()
        raise


if __name__ == '__main__':
    main()
