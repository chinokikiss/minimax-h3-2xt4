"""Benchmark-only phase instrumentation around the unchanged TP2 disk baker."""
import argparse
from collections import defaultdict
from contextlib import contextmanager
import functools
import gzip
import inspect
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'ComfyUI'), str(ROOT/'benchmarks')]
import torch
from patches import tp2_disk_bake as baker

ORIGINAL_WORKER = baker._bake_layers


class Recorder:
    def __init__(self, output, owner, device=None):
        self.output, self.owner, self.device = Path(output), owner, device
        self.output.mkdir(parents=True, exist_ok=True)
        self.rows, self.stack, self.restore, self.events = [], [], [], []

    @contextmanager
    def span(self, phase, target=None, cuda=False):
        row = dict(owner=self.owner, pid=os.getpid(), phase=phase,
                   target=target or (self.stack[-1]['target'] if self.stack else None),
                   start_ns=time.monotonic_ns(), children_ns=0)
        cpu = time.thread_time_ns()
        pair = None
        if cuda and self.device is not None:
            pair = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            pair[0].record()
        self.stack.append(row)
        try:
            with torch.profiler.record_function('bake/'+phase):yield
        finally:
            if pair is not None:pair[1].record()
            row.update(end_ns=time.monotonic_ns(), thread_cpu_ns=time.thread_time_ns()-cpu)
            self.stack.pop()
            row['wall_ns'] = row['end_ns']-row['start_ns']
            row['exclusive_wall_ns'] = row['wall_ns']-row.pop('children_ns')
            if self.stack:self.stack[-1]['children_ns'] += row['wall_ns']
            self.rows.append(row)
            if pair is not None:self.events.append((row, pair))

    def wrap(self, module, name, phase, *, cuda=False, target_index=None):
        original = getattr(module, name)
        descriptor = inspect.getattr_static(module, name)
        @functools.wraps(original)
        def wrapped(*args, **kwargs):
            target = args[target_index] if target_index is not None and len(args)>target_index else None
            with self.span(phase, target if isinstance(target, str) else None, cuda):
                return original(*args, **kwargs)
        self.restore.append((module, name, descriptor))
        setattr(module, name, staticmethod(wrapped) if isinstance(module, type) and isinstance(descriptor, (staticmethod, classmethod)) else wrapped)

    def install(self):
        from patches.windowed_tp import CheckpointReader
        from patches.tp2_bundle import BundleReader
        from patches import tp2_bundle
        from patches import offline_lora, offline_lora_reference
        from comfy_kitchen.tensor.int8 import TensorWiseINT8Layout
        from comfy.weight_adapter.lora import LoRAAdapter
        for module, name, phase, cuda, target in (
            (CheckpointReader, 'read_tensor', 'checkpoint_read', False, 1),
            (BundleReader, '_shard', 'base_shard_validation', False, None),
            (baker, 'sha', 'file_sha256', False, None),
            (tp2_bundle, 'sha', 'base_file_sha256', False, None),
            (baker, 'state_proof', 'tensor_sha256', False, None),
            (baker, 'save_file', 'safetensors_serialize_write', False, None),
            (baker.os, 'fsync', 'fsync', False, None),
            (baker, 'split_matrix', 'split_tp2', False, None),
            (baker, 'atomic_json', 'json_publish', False, None),
            (offline_lora, 'merged_weight', 'production_merge', True, 2),
            (offline_lora_reference, 'verify_weight', 'independent_verification', True, 2),
            (TensorWiseINT8Layout, 'dequantize', 'dequantize', True, None),
            (TensorWiseINT8Layout, 'quantize', 'convrot_quantize', True, None),
            (LoRAAdapter, 'calculate_weight', 'native_lora_apply', True, 2),
        ):self.wrap(module, name, phase, cuda=cuda, target_index=target)

    def finish(self):
        self.uninstall()
        if self.device is not None:
            torch.cuda.synchronize(self.device)
            for row, (start, end) in self.events:row['cuda_timeline_ms'] = start.elapsed_time(end)
        phases = defaultdict(lambda:dict(calls=0, wall_seconds=0., exclusive_wall_seconds=0., thread_cpu_seconds=0.))
        for row in self.rows:
            phase = phases[row['phase']]
            phase['calls'] += 1
            for field, source in [('wall_seconds','wall_ns'),('exclusive_wall_seconds','exclusive_wall_ns'),('thread_cpu_seconds','thread_cpu_ns')]:
                phase[field] += row[source]/1e9
        result = dict(owner=self.owner, phases=dict(phases), events=self.rows,
                      cuda_scope='Inclusive timeline spans, including dependencies/idle; not summed kernel execution time',
                      cpu_scope='Exclusive wall subtracts nested instrumented scopes; cross-process rows overlap')
        (self.output/(self.owner+'.json')).write_text(json.dumps(result,indent=2))
    def uninstall(self):
        for module, name, descriptor in reversed(self.restore):setattr(module,name,descriptor)
        self.restore.clear()


def profile_worker(base_path, stack, output, layers, device, recipe_sha):
    torch.cuda.set_device(device)
    recorder = Recorder(os.environ['H3_BAKE_PROFILE_DIR'], 'worker_'+device.replace(':','_'), device)
    recorder.install()
    # Capture only the first production merge. Full-bake phase timings remain
    # available without retaining a many-GB profiler trace for every matrix.
    from patches import offline_lora
    original = offline_lora.merged_weight
    captured = False
    def first_trace(*args, **kwargs):
        nonlocal captured
        if captured:return original(*args, **kwargs)
        captured = True
        trace = recorder.output/(recorder.owner+'_first_merge.json')
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as profile:
            result = original(*args, **kwargs)
            torch.cuda.synchronize(device)
        profile.export_chrome_trace(str(trace))
        data = json.loads(trace.read_text())
        kernels = sum(event.get('cat')=='kernel' for event in data['traceEvents'])
        trace.with_suffix('.json.gz').write_bytes(gzip.compress(trace.read_bytes(),mtime=0))
        trace.unlink()
        (recorder.output/(recorder.owner+'_trace_check.json')).write_text(json.dumps(dict(cuda_kernel_events=kernels,flush_complete=True),indent=2))
        return result
    offline_lora.merged_weight = first_trace
    try:
        with recorder.span('worker_total'):return ORIGINAL_WORKER(base_path,stack,output,layers,device,recipe_sha)
    finally:
        offline_lora.merged_weight = original
        recorder.finish()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--base',type=Path,required=True)
    parser.add_argument('--lora',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--profile',type=Path,required=True)
    parser.add_argument('--expected',type=Path,required=True)
    args=parser.parse_args()
    torch.set_grad_enabled(False);torch.backends.cuda.matmul.allow_tf32=False
    os.environ['H3_BAKE_PROFILE_DIR']=str(args.profile.resolve())
    recorder=Recorder(args.profile,'parent','cuda:0');recorder.install()
    baker._bake_layers=profile_worker
    try:
        with recorder.span('bake_total'):
            descriptor=baker.bake(args.base,((str(args.lora),1.),),args.output,devices=('cuda:0','cuda:1'))
        receipt=json.loads((descriptor.parent/'bake_receipt.json').read_text())
        expected=json.loads(args.expected.read_text())
        exact=receipt['shards'].keys()==expected['shards'].keys() and all(row['state_sha256']==expected['shards'][name] for name,row in receipt['shards'].items())
        assert exact and receipt['verified_native_targets']==208
        assert receipt['auxiliary_sha256']==expected['auxiliary_sha256']
        traces=[json.loads(p.read_text()) for p in args.profile.glob('worker_*_trace_check.json')]
        trace_ok=len(traces)==2 and all(t['cuda_kernel_events']>0 and t['flush_complete'] for t in traces)
        (args.profile/'bake_receipt.json').write_text(json.dumps(receipt,indent=2))
        (args.profile/'result.json').write_text(json.dumps(dict(status='complete' if trace_ok else 'cuda_trace_blocked',all_100_shard_states_match_prior_t4=exact,
            verified_native_targets=208,independent_native_verification=receipt['native_independent_verification'],
            auxiliary_matches_prior_t4=True, cuda_traces_complete=trace_ok,
            profiling_overhead_included=True,comparison_scope='Numerical byte proofs against prior T4 bake; not a speedup comparison'),indent=2))
    finally:
        baker._bake_layers=ORIGINAL_WORKER;recorder.finish()


if __name__=='__main__':main()
