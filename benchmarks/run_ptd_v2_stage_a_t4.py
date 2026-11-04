"""Stage A: physical dual-T4 PTD Bank v2, two trained blocks and eight tiny hops.

No full checkpoint, TE execution, VAE execution, long-sequence video, Dataset
publication or production acceptance. The Bank is explicitly allow_fixture=True.
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


class MemoryMonitor:
    def __init__(self):
        self.stop = threading.Event()
        self.peaks_mib = [0, 0]
        self.samples = 0
        self.errors = 0
        self.worker = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self.stop.is_set():
            try:
                r = subprocess.run(['nvidia-smi', '--query-gpu=memory.used',
                                    '--format=csv,noheader,nounits'],
                                   capture_output=True, text=True, timeout=10, check=True)
                values = [int(line.strip()) for line in r.stdout.splitlines() if line.strip()]
                if len(values) != 2:
                    raise ValueError('Two GPU memory rows required')
                self.peaks_mib = [max(old, now) for old, now in zip(self.peaks_mib, values)]
                self.samples += 1
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
    parser.add_argument('--scratch', type=Path, default=Path('/kaggle/temp/ptd_v2_stage_a'))
    args = parser.parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    scratch = args.scratch.resolve()
    if output.exists() or scratch.exists():
        raise FileExistsError('Do not overwrite previous Stage A evidence or partial scratch')
    output.mkdir(parents=True)
    scratch.mkdir(parents=True)
    state = dict(format='h3-ptd-v2-stage-a-t4-v1', status='starting',
                 started_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                 source_scope='two real trained PTD blocks, full eight-hop clock',
                 production_ready=False, fixture_only=True,
                 no_video_generation=True, no_te_vae_execution=True,
                 ptd_full_release_sha256_verified=False, hop_count=8,
                 adapted_layers=2, stages={})

    def save():
        (output / 'result.json').write_text(json.dumps(state, indent=2), encoding='utf-8')

    def stage(name, operation):
        print('PTD_V2_STAGE_A_START', name, flush=True)
        begin = time.monotonic()
        state['status'] = name
        state['stages'][name] = dict(status='running')
        save()
        try:
            result = operation()
            state['stages'][name] = dict(status='passed',
                                         seconds=round(time.monotonic()-begin, 3))
            save()
            print('PTD_V2_STAGE_A_END', name, state['stages'][name]['seconds'], flush=True)
            return result
        except BaseException as exc:
            state['stages'][name] = dict(status='failed',
                                         seconds=round(time.monotonic()-begin, 3),
                                         error=repr(exc))
            save()
            raise

    save()
    try:
        # ComfyUI imports rely on the pinned command line and source lock.
        sys.argv = ['ptd-v2-stage-a']
        import comfy.options
        comfy.options.enable_args_parsing()
        import comfy.cli_args
        import torch
        import importlib.metadata as md
        from patches.legacy_runtime import validate_packages, validate_source
        from patches.tp2_bundle import sha
        from patches.ptd.bake_v2 import bake_shared_part, bake_hop_part, assemble_bank_v2
        from patches.ptd.bank_v2 import PTDBankV2
        from patches.ptd.publish_v2 import verify_part
        from run_ptd_tp2_t4 import smoke, resource_plan
        from tp2_public_dit_assets import stage as stage_base
        from ptd_t4_assets import fetch_subset

        validate_packages()
        validate_source(source / 'ComfyUI')
        if torch.cuda.device_count() != 2:
            raise RuntimeError('Stage A requires two physical GPUs')
        devices = [dict(index=i, name=torch.cuda.get_device_name(i),
                        capability=list(torch.cuda.get_device_capability(i)),
                        uuid=str(torch.cuda.get_device_properties(i).uuid))
                   for i in range(2)]
        if any('T4' not in row['name'] or row['capability'] != [7, 5] for row in devices):
            raise RuntimeError('Not physical Tesla T4 devices')
        if len(set(row['uuid'] for row in devices)) != 2:
            raise RuntimeError('GPU physical UUIDs are not distinct')
        state['hardware'] = dict(devices=devices, torch=str(torch.__version__),
                                 cuda=torch.version.cuda, kitchen=md.version('comfy-kitchen'),
                                 tilelang=md.version('tilelang'))
        os.environ.update(HF_HUB_DISABLE_XET='1', HF_HUB_DISABLE_PROGRESS_BARS='1',
                          OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', MALLOC_ARENA_MAX='2')
        torch.set_grad_enabled(False)
        torch.set_num_threads(2)
        torch.backends.cuda.matmul.allow_tf32 = False
        state['scratch_initial_free_bytes'] = shutil.disk_usage(scratch).free
        parts = scratch / 'parts'
        parts.mkdir()
        with MemoryMonitor() as meter:
            base = stage('download_two_base_blocks', lambda:
                         stage_base(source, scratch / 'base', layers=(0, 1)))
            state['base_descriptor_sha256'] = sha(base)
            state['scratch_after_base_free_bytes'] = shutil.disk_usage(scratch).free
            trained = stage('download_trained_ptd_ranges', lambda:
                            fetch_subset(scratch / 'trained_adapter', layers=2))
            state['trained_ptd_subset_sha256'] = sha(trained)
            provenance = json.loads((trained.parent / 'provenance.json').read_text(encoding='utf-8'))
            state['trained_source_provenance'] = provenance
            if provenance['full_release_sha256_verified'] or provenance['deployment_allowed']:
                raise ValueError('Subset may never claim full production-release proof')
            shared_receipt = stage('bake_shared', lambda:
                                   bake_shared_part(base, trained, parts / 'shared',
                                                    layers=2, allow_fixture=True,
                                                    allow_pruned_adaln=True))
            hop_roots = []
            for hop in range(8):
                root = parts / f'hop_{hop:02d}'
                receipt = stage(f'bake_hop_{hop:02d}', lambda hop=hop, root=root:
                                bake_hop_part(base, trained, hop, root, layers=2,
                                              devices=('cuda:0', 'cuda:1'),
                                              allow_pruned_adaln=True,
                                              allow_fixture=True))
                row = json.loads(receipt.read_text(encoding='utf-8'))
                if len(row['shards']) != 4 or row['recipe_sha256'] != json.loads(
                        shared_receipt.read_text(encoding='utf-8'))['recipe_sha256']:
                    raise ValueError('Two-block hop rank/recipe coverage incomplete')
                hop_roots.append(root)
            proofs = stage('verify_local_part_hashes', lambda: [
                verify_part(parts / 'shared', 'shared',
                            expected_receipt_sha256=sha(shared_receipt)),
                *[verify_part(root, f'hop_{hop:02d}',
                              expected_receipt_sha256=sha(root / 'receipt.json'))
                  for hop, root in enumerate(hop_roots)]])
            state['local_part_proofs'] = proofs
            manifest = stage('assemble_fixture_bank', lambda:
                             assemble_bank_v2(parts / 'bank_v2.json', parts / 'shared',
                                              hop_roots, allow_fixture=True))
            bank = PTDBankV2(manifest, allow_fixture=True)
            if bank.layers != 2 or len(bank.hops) != 8:
                raise ValueError('Stage A v2 fixture Bank has incorrect coverage')
            try:
                PTDBankV2(manifest)
            except ValueError:
                state['production_fixture_rejection'] = True
            else:
                raise AssertionError('A two-layer fixture was accepted as production')
            state['bank_manifest_sha256'] = sha(manifest)
            resources = resource_plan(base)
            state['resources'] = resources
            smoke_output = output / 'tiny_latent_smoke'
            smoke_output.mkdir()
            result = stage('eight_hop_tiny_latent_runtime', lambda:
                           smoke(base, manifest, smoke_output, 'queue', resources,
                                 cpu_options=dict(future_cache=False,
                                                  prefetch_refiner=False),
                                 verify_dit_weights=False, bank_v2=True))
            state['tiny_latent'] = dict(
                final_shapes=result['final_shapes'],
                eight_hop_execution=result['hops'] == list(range(8)),
                finite=result['finite'], hop_seconds=result['hop_seconds'],
                aot_wall_seconds=result['aot_wall_seconds'],
                selection_seconds=result['selection_seconds'],
                final_sha256=result['final_sha256'],
                benchmark_backend_counts=[row.get('benchmark', {}).get('executed_backends', {})
                                          for row in result['rank_stats']])
            if not state['tiny_latent']['eight_hop_execution'] or not result['finite']:
                raise ValueError('PTD v2 eight-hop tiny latent execution incomplete')
            state['scratch_final_free_bytes'] = shutil.disk_usage(scratch).free
            state['nvml_peak_used_mib'] = meter.peaks_mib
            state['nvml_samples'] = meter.samples
            state['nvml_errors'] = meter.errors
            if meter.samples < 2:
                raise ValueError('GPU memory observer did not produce enough samples')
        state.update(status='complete',
                     finished_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                     acceptance_scope='Trained two-layer functional fixture only; no E2E quality or 50-layer performance claim')
    except BaseException as exc:
        state.update(status='failed', error=repr(exc),
                     finished_utc=dt.datetime.now(dt.timezone.utc).isoformat())
        (output / 'exception.txt').write_text(traceback.format_exc(), encoding='utf-8')
        raise
    finally:
        save()
        print('PTD_V2_STAGE_A_FINAL',
              json.dumps(dict(status=state['status'], stages=state['stages'],
                              nvml_peak_used_mib=state.get('nvml_peak_used_mib'))), flush=True)


if __name__ == '__main__':
    main()
