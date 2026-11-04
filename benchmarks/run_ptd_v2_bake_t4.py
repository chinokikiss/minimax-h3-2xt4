"""Full trained PTD v2 single-hop physical dual-T4 Baker qualification.

One hop only. The output is evidence, not a published Dataset or a complete
eight-hop Bank. This process never creates a Kaggle Dataset or invokes inference.
"""
import argparse
import datetime as dt
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'ComfyUI'), str(ROOT / 'benchmarks')]


class T4MemorySampler:
    def __init__(self):
        self.stop = threading.Event()
        self.samples = 0
        self.peaks_mib = [0, 0]
        self.errors = 0
        self.worker = threading.Thread(target=self._sample, daemon=True)

    def _sample(self):
        while not self.stop.is_set():
            try:
                run = subprocess.run(
                    ['nvidia-smi', '--query-gpu=memory.used',
                     '--format=csv,noheader,nounits'], capture_output=True,
                    text=True, timeout=12, check=True)
                values = [int(line.strip()) for line in run.stdout.splitlines() if line.strip()]
                if len(values) == 2:
                    self.peaks_mib = [max(p, v) for p, v in zip(self.peaks_mib, values)]
                    self.samples += 1
                else:
                    self.errors += 1
            except (OSError, ValueError, subprocess.SubprocessError):
                self.errors += 1
            self.stop.wait(1.)

    def __enter__(self):
        self.worker.start()
        return self

    def __exit__(self, *_):
        self.stop.set()
        self.worker.join(timeout=15)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--scratch', type=Path, default=Path('/kaggle/temp/h3_ptd_v2_hop0'))
    args = parser.parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    scratch = args.scratch.resolve()
    if output.exists() or scratch.exists():
        raise FileExistsError('Preserve immutable qualification outputs and scratch')
    output.mkdir(parents=True)
    scratch.mkdir(parents=True)
    state = dict(status='starting', format='h3-ptd-v2-hop0-physical-t4-v1',
                 started_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                 hop=0, layers=50, ptd_source='pinned-full-trained-release',
                 publication='not_performed', dataset_sources=[],
                 normal_pipeline_unchanged=True, stages={}, hardware={},
                 independent_native_reference=(os.environ.get('PTD_V2_INDEPENDENT_VERIFY') == '1'))

    def save():
        (output / 'result.json').write_text(json.dumps(state, indent=2), encoding='utf-8')

    def stage(name, function):
        print('PTD_V2_T4_STAGE_START', name, flush=True)
        start = time.monotonic()
        state['status'] = name
        state['stages'][name] = dict(status='running')
        save()
        try:
            value = function()
            state['stages'][name] = dict(status='complete', seconds=time.monotonic()-start)
            save()
            print('PTD_V2_T4_STAGE_END', name, state['stages'][name]['seconds'], flush=True)
            return value
        except BaseException as exc:
            state['stages'][name] = dict(status='failed', seconds=time.monotonic()-start,
                                         error=repr(exc))
            save()
            raise

    save()
    try:
        import torch
        import importlib.metadata as md
        from patches.legacy_runtime import validate_packages, validate_source
        from patches.ptd.checkpoint import RELEASE, download_on_t4
        from patches.ptd.bake_v2 import bake_shared_part, bake_hop_part
        from patches.ptd.publish_v2 import verify_part
        from patches.tp2_bundle import sha
        from preflight_ptd_v2_bake import inspect_inputs
        from tp2_public_dit_assets import stage as stage_base

        validate_packages()
        validate_source(source / 'ComfyUI')
        if torch.cuda.device_count() != 2:
            raise RuntimeError('Production Baker requires two physical T4 GPUs')
        devices = [
            dict(index=i, name=torch.cuda.get_device_name(i),
                 capability=list(torch.cuda.get_device_capability(i)),
                 uuid=str(torch.cuda.get_device_properties(i).uuid))
            for i in range(2)]
        if any(d['capability'] != [7, 5] or 'T4' not in d['name'] for d in devices):
            raise RuntimeError('GPU hardware differs from two Tesla T4s')
        if len({d['uuid'] for d in devices}) != 2:
            raise RuntimeError('Duplicate physical GPU identity')
        state['hardware'] = dict(devices=devices, torch=str(torch.__version__),
                                 cuda=torch.version.cuda,
                                 tilelang=md.version('tilelang'),
                                 kitchen=md.version('comfy-kitchen'))
        os.environ.update(HF_HUB_DISABLE_XET='1', HF_HUB_DISABLE_PROGRESS_BARS='1',
                          OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', MALLOC_ARENA_MAX='2')
        torch.set_grad_enabled(False)
        torch.set_num_threads(2)
        torch.backends.cuda.matmul.allow_tf32 = False

        with T4MemorySampler() as meter:
            state['scratch_before_download'] = shutil.disk_usage(scratch).free
            base = stage('stage_full_base', lambda: stage_base(source, scratch / 'base'))
            state['base_descriptor_sha256'] = sha(base)
            state['scratch_after_base'] = shutil.disk_usage(scratch).free
            checkpoint = stage('download_pinned_full_ptd', lambda: download_on_t4(scratch / 'adapter'))
            state['ptd_sha256'] = RELEASE['sha256']
            state['ptd_bytes'] = checkpoint.stat().st_size
            parts = scratch / 'parts'
            parts.mkdir()
            admission = stage('production_source_preflight', lambda: inspect_inputs(
                base, checkpoint, parts, layers=50,
                allow_pruned_adaln=True, verify_ptd_sha256=True, check_t4=True))
            (output / 'source_preflight.json').write_text(json.dumps(admission, indent=2),
                                                         encoding='utf-8')
            if not admission['physical_baker_ready']:
                raise ValueError('Production source/T4/scratch preflight failed: '
                                 + '; '.join(admission['issues']))
            state['recipe_inputs'] = dict(base=admission['base'],
                                          release_sha256=admission['checkpoint_sha256'])
            shared_receipt = stage('bake_shared', lambda: bake_shared_part(
                base, checkpoint, parts / 'shared', layers=50, allow_pruned_adaln=True))
            hop_receipt = stage('bake_hop_00', lambda: bake_hop_part(
                base, checkpoint, 0, parts / 'hop_00', layers=50,
                devices=('cuda:0', 'cuda:1'), allow_pruned_adaln=True))
            shutil.copy2(shared_receipt, output / 'shared_receipt.json')
            shutil.copy2(hop_receipt, output / 'hop_00_receipt.json')
            shared = json.loads(shared_receipt.read_text(encoding='utf-8'))
            hop = json.loads(hop_receipt.read_text(encoding='utf-8'))
            if len(hop['shards']) != 100 or len(set(hop['verified_targets'])) != len(
                    hop['verified_targets']):
                raise ValueError('Full 50-layer/two-rank hop or merge proof incomplete')
            if shared['recipe_sha256'] != hop['recipe_sha256']:
                raise ValueError('Shared and hop recipe mismatch')
            state['verified_merge_targets'] = len(hop['verified_targets'])
            state['shards'] = len(hop['shards'])
            state['recipe_sha256'] = hop['recipe_sha256']
            state['local_source_receipt_sha256'] = sha(hop_receipt)
            local = stage('verify_local_part_sha256', lambda: verify_part(
                parts / 'hop_00', 'hop_00', expected_receipt_sha256=sha(hop_receipt)))
            state['local_part_validation'] = local
            state['scratch_after_bake'] = shutil.disk_usage(scratch).free
            state['gpu_memory_peak_sampled_mib'] = meter.peaks_mib
            state['gpu_memory_samples'] = meter.samples
            state['gpu_memory_monitor_errors'] = meter.errors
            if meter.samples < 2:
                raise ValueError('Insufficient physical GPU memory monitor samples')
        state.update(status='complete', finished_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                     precision_scope=('Full-width Native merge; cached FP32-reference output-byte checks '
                                      'and exact local SHA256. Independent A/B verification '
                                      'is performed only with PTD_V2_INDEPENDENT_VERIFY=1; '
                                      'no E2E quality equivalence'),
                     persistence_scope='Local part bytes reside only in disposable GPU scratch; receipts exported')
    except BaseException as error:
        state.update(status='failed', error=repr(error),
                     finished_utc=dt.datetime.now(dt.timezone.utc).isoformat())
        (output / 'exception.txt').write_text(traceback.format_exc(), encoding='utf-8')
        raise
    finally:
        save()
        print('PTD_V2_T4_FINAL', json.dumps(dict(
            status=state['status'], stages=state['stages'],
            shards=state.get('shards'), merge_targets=state.get('verified_merge_targets'),
            gpu_peak_mib=state.get('gpu_memory_peak_sampled_mib'))), flush=True)


if __name__ == '__main__':
    main()
