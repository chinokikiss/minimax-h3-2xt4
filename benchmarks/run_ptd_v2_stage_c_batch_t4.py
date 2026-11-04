"""Bounded production Stage C independent-T4 Hop pipeline (local job only).

This runner does NOT constitute provider version or remote SHA256 proof.
By default no local payload is deleted. Explicit --retire-after-upload-ack
allows reclaiming reproducible scratch only after the provider acknowledges
a create/update request; remote mounted byte-proof remains an independent job.
Never automatically retry, list provider state or submit Notebook jobs.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, wait, FIRST_COMPLETED
import datetime as dt
import json
import multiprocessing as mp
import os
from pathlib import Path
import shutil
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'ComfyUI'), str(ROOT / 'benchmarks')]

from benchmarks.run_ptd_v2_stage_c_publish_t4 import (
    EXPECTED_RECIPE, _legal_files, dataset_id)
from benchmarks.run_ptd_v2_stage_c_dual_hop_t4 import (
    _bake_worker, _prepared_recipe, _source_seal, _assert_seal,
    _identity, _provider_error_tag, _save)

PARTS = tuple(f'hop_{hop:02d}' for hop in range(8))


def retire_owned_hop(scratch, hop, receipt_sha, published):
    """Only remove exact owned Hop payload + hardlink stage, never source/base."""
    from patches.tp2_bundle import sha
    if type(hop) is not int or not 0 <= hop < 8:
        raise ValueError('Invalid Hop cleanup identity')
    part = f'hop_{hop:02d}'
    if (published.get('dataset_id') != dataset_id(part)
            or published.get('receipt_sha256') != receipt_sha
            or published.get('status') not in (
                'create_request_accepted_processing_unverified',
                'version_request_accepted_processing_unverified')):
        raise ValueError('No matching provider upload ACK for local cleanup')
    scratch = Path(scratch).resolve()
    root = scratch / 'parts' / part
    staging = scratch / 'publish_staging' / part
    if (root.is_symlink() or staging.is_symlink()
            or not root.is_dir() or not staging.is_dir()
            or root.resolve() != root or staging.resolve() != staging
            or sha(root / 'receipt.json') != receipt_sha
            or sha(staging / 'payload' / 'receipt.json') != receipt_sha):
        raise ValueError('Refusing to retire mismatched or symlinked Hop scratch')
    sources = {p.name: p for p in root.iterdir() if p.is_file() and not p.is_symlink()}
    links = {p.name: p for p in (staging / 'payload').iterdir()
             if p.is_file() and not p.is_symlink()}
    if (not sources or sources.keys() != links.keys()
            or len(sources) != len(list(root.iterdir()))
            or len(links) != len(list((staging / 'payload').iterdir()))):
        raise ValueError('Refusing to retire incomplete hardlink staging')
    for name, origin in sources.items():
        a, b = origin.stat(), links[name].stat()
        if (a.st_dev, a.st_ino, a.st_size) != (b.st_dev, b.st_ino, b.st_size):
            raise ValueError('Staging payload is not an owned hardlink: ' + name)
    # Hardlink directory must be unlinked first, otherwise bytes remain pinned.
    shutil.rmtree(staging)
    shutil.rmtree(root)
    return {'part': part, 'retired': True,
            'gate': 'provider_upload_ack_only',
            'remote_sha256': 'pending', 'source_rebake_required_on_remote_failure': True}


def run(args):
    hops = tuple(args.hops)
    if (not hops or len(hops) != len(set(hops))
            or any(type(h) is not int or not 0 <= h < 8 for h in hops)):
        raise ValueError('Hops must be unique integers 0..7')
    if not set(args.replace_hops).issubset(hops):
        raise ValueError('Replacement Hops must be among selected Hops')
    if len(hops) > 2 and not args.retire_after_upload_ack:
        raise ValueError('More than two Hops needs explicit scratch reclamation policy')
    source, scratch, output = (p.resolve() for p in
                               (args.source, args.scratch, args.output))
    if scratch.exists() or output.exists():
        raise FileExistsError('Never reuse an existing Stage C batch scratch/output')
    scratch.mkdir(parents=True)
    output.mkdir(parents=True)
    state = dict(format='h3-ptd-v2-stage-c-bounded-batch-v1',
                 status='starting', hops=list(hops),
                 replace_hops=list(args.replace_hops),
                 max_active_payload_hops=2, parallel_gpu_bakers=2,
                 provider_upload_concurrency=1,
                 retire_after_upload_ack=args.retire_after_upload_ack,
                 cleanup_gate=('provider_ack_remote_unverified' if args.retire_after_upload_ack
                               else 'disabled'),
                 remote_readback='pending',
                 source_recipe_sha256=EXPECTED_RECIPE,
                 started_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                 stages={}, published={}, retained_parts=[], retired_parts=[])
    lock = threading.Lock()
    provider_lock = threading.Lock()
    failed_upload = threading.Event()
    _save(output / 'result.json', state)

    def stage(name, function):
        started = time.monotonic()
        with lock:
            state['stages'][name] = dict(status='running')
            _save(output / 'result.json', state)
        print('STAGE_C_BATCH_START', name, flush=True)
        try:
            value = function()
        except BaseException as error:
            with lock:
                state['stages'][name] = dict(status='failed',
                                             seconds=time.monotonic() - started,
                                             error_type=type(error).__name__)
                _save(output / 'result.json', state)
            raise
        with lock:
            state['stages'][name] = dict(status='complete',
                                         seconds=time.monotonic() - started)
            _save(output / 'result.json', state)
        print('STAGE_C_BATCH_DONE', name, flush=True)
        return value

    def publish(hop, license_path):
        """Admit/stage without full payload rehash; serialize Kaggle API calls."""
        from kaggle.api.kaggle_api_extended import KaggleApi
        from patches.ptd.publish_v2 import admit_baked_part, prepare_private_kaggle_part
        from patches.tp2_bundle import sha

        part = f'hop_{hop:02d}'
        root = scratch / 'parts' / part
        admission = stage('admit_bake_receipt_' + part,
                          lambda: admit_baked_part(root, part))
        portable = {k: v for k, v in admission.items() if k != 'source_snapshot'}
        _save(output / (part + '_local_admission.json'), portable)
        identifier = dataset_id(part)
        staging = scratch / 'publish_staging' / part
        packed = stage('stage_' + part, lambda: prepare_private_kaggle_part(
            root, part, staging, dataset_id=identifier, license_name='other',
            title=f'H3 PTD v2 {part} 20261103', verified_proof=admission))
        if packed['source_receipt_sha256'] != admission['receipt_sha256']:
            raise ValueError('Staging altered baked source receipt')
        shutil.copy2(license_path, staging / 'LICENSE-MiniMax-H3.txt')
        (staging / 'NOTICE.txt').write_text(
            'Private H3 PTD v2 Stage C derived model; '
            'MiniMax H3 Community License Agreement.\n', encoding='utf-8')
        if shutil.disk_usage(staging).free < packed['payload_bytes'] + 2 * 2**30:
            raise OSError('No headroom for Kaggle SDK upload tar archive')

        def request():
            with provider_lock:
                if failed_upload.is_set():
                    raise RuntimeError('Previous private Dataset upload failed; stop cascade')
                api = KaggleApi()
                api.authenticate()
                try:
                    if hop in args.replace_hops:
                        response = api.dataset_create_version(
                            str(staging), version_notes='H3 PTD v2 production replacement',
                            quiet=True, convert_to_csv=False,
                            delete_old_versions=True, dir_mode='tar')
                    else:
                        response = api.dataset_create_new(
                            str(staging), public=False, quiet=True,
                            convert_to_csv=False, dir_mode='tar')
                except BaseException as error:
                    failed_upload.set()
                    raise RuntimeError('Kaggle upload error: '
                                       + _provider_error_tag(error)) from None
                if (str(getattr(response, 'status', '')).lower() != 'ok'
                        or getattr(response, 'error', None)):
                    failed_upload.set()
                    raise RuntimeError('Kaggle Dataset create/update not acknowledged')
            return dict(dataset_id=identifier, part=part, hop=hop,
                        status=('version_request_accepted_processing_unverified'
                                if hop in args.replace_hops else
                                'create_request_accepted_processing_unverified'),
                        receipt_sha256=sha(root / 'receipt.json'),
                        payload_bytes=packed['payload_bytes'],
                        remote_readback='pending', version_pin='pending')

        return stage('upload_' + part, request)

    try:
        import torch
        from kaggle.api.kaggle_api_extended import KaggleApi
        from patches.legacy_runtime import validate_packages, validate_source
        from patches.ptd.checkpoint import RELEASE, download_on_t4
        from benchmarks.preflight_ptd_v2_bake import inspect_inputs
        from benchmarks.tp2_public_dit_assets import stage as stage_base

        validate_packages()
        validate_source(source / 'ComfyUI')
        if (torch.cuda.device_count() != 2
                or any(torch.cuda.get_device_capability(i) != (7, 5)
                       or 'T4' not in torch.cuda.get_device_name(i) for i in (0, 1))
                or len({str(torch.cuda.get_device_properties(i).uuid)
                        for i in (0, 1)}) != 2):
            raise RuntimeError('Requires two separate physical Tesla T4 GPUs')
        torch.set_num_threads(2)
        torch.backends.cuda.matmul.allow_tf32 = False
        os.environ.update(HF_HUB_DISABLE_XET='1', HF_HUB_DISABLE_PROGRESS_BARS='1',
                          OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', MALLOC_ARENA_MAX='2')
        license_path, license_sha, license_rev = stage(
            'license_source', lambda: _legal_files(source, scratch))
        state['license'] = dict(sha256=license_sha, revision=license_rev)
        stage('private_auth', lambda: KaggleApi().authenticate())
        with ThreadPoolExecutor(max_workers=2) as loader:
            future_base = loader.submit(stage, 'stage_full_base',
                                        lambda: stage_base(source, scratch / 'base'))
            future_adapter = loader.submit(stage, 'download_pinned_ptd',
                                           lambda: download_on_t4(scratch / 'adapter'))
            base, checkpoint = future_base.result(), future_adapter.result()
        source_fingerprint = dict(sha256=RELEASE['sha256'],
                                  fingerprint=_identity(checkpoint),
                                  verifier='download_on_t4')
        parts_root = scratch / 'parts'
        parts_root.mkdir()
        (scratch / 'publish_staging').mkdir()
        admission = stage('preflight_full_50_layer', lambda: inspect_inputs(
            base, checkpoint, parts_root, layers=50, allow_pruned_adaln=True,
            verify_ptd_sha256=True, check_t4=True,
            preverified_checkpoint=source_fingerprint))
        _save(output / 'source_preflight.json', admission)
        if not admission['physical_baker_ready']:
            raise ValueError('Source preflight not qualified')
        recipe = stage('reuse_source_recipe',
                       lambda: _prepared_recipe(base, checkpoint, admission))
        seal = _source_seal(base, checkpoint)
        max_slots = min(2, len(hops))
        min_free = (max_slots * admission['scratch']['minimum_free_bytes']
                    + max_slots * 23 * (2**30))
        free = shutil.disk_usage(scratch).free
        if free < min_free:
            raise OSError(f'Insufficient bounded Hop/tar scratch: {free} < {min_free}')
        state['scratch_admission'] = dict(available=free, required=min_free)
        # One fixed spawned process per GPU. A slot is held until upload ACK and
        # optional safe scratch reclamation; at most two unreclaimed Hop trees.
        gpu = [ProcessPoolExecutor(max_workers=1, mp_context=mp.get_context('spawn'))
               for _ in range(max_slots)]
        try:
            with ThreadPoolExecutor(max_workers=2, thread_name_prefix='ptd-stage-c-upload') as uploader:
                remaining = iter(hops)
                active_bakes = {}
                active_uploads = {}
                errors = []

                def launch(slot):
                    try:
                        hop = next(remaining)
                    except StopIteration:
                        return
                    future = gpu[slot].submit(
                        _bake_worker, str(base), str(checkpoint), hop,
                        str(parts_root / f'hop_{hop:02d}'),
                        f'cuda:{slot}', recipe, seal)
                    active_bakes[future] = (hop, slot)
                    print('STAGE_C_BATCH_BAKE_START', hop, slot, flush=True)

                for slot in range(max_slots):
                    launch(slot)
                while active_bakes or active_uploads:
                    done, _ = wait(set(active_bakes) | set(active_uploads),
                                   return_when=FIRST_COMPLETED)
                    for future in done:
                        if future in active_bakes:
                            hop, slot = active_bakes.pop(future)
                            try:
                                baked = future.result()
                                shutil.copy2(baked['receipt'],
                                             output / f'hop_{hop:02d}_receipt.json')
                                with lock:
                                    state['stages'][f'bake_hop_{hop:02d}'] = dict(
                                        status='complete', seconds=baked['seconds'])
                                    _save(output / 'result.json', state)
                                active_uploads[uploader.submit(
                                    publish, hop, license_path)] = (hop, slot)
                            except BaseException as error:
                                errors.append(type(error).__name__)
                                failed_upload.set()
                        else:
                            hop, slot = active_uploads.pop(future)
                            part = f'hop_{hop:02d}'
                            try:
                                published = future.result()
                                with lock:
                                    state['published'][part] = published
                                    _save(output / 'result.json', state)
                                if args.retire_after_upload_ack:
                                    retired = stage('retire_' + part, lambda: retire_owned_hop(
                                        scratch, hop, published['receipt_sha256'], published))
                                    with lock:
                                        state['retired_parts'].append(retired)
                                        _save(output / 'result.json', state)
                                else:
                                    with lock:
                                        state['retained_parts'].append(part)
                                        _save(output / 'result.json', state)
                                if not failed_upload.is_set():
                                    launch(slot)
                            except BaseException as error:
                                errors.append(type(error).__name__)
                                failed_upload.set()
                    # Halt new bakes on an error; preserve remaining unfinished
                    # local payloads for operator-controlled recovery.
                if errors:
                    raise RuntimeError('Stage C pipeline interrupted: ' + ','.join(errors))
        finally:
            for process in gpu:
                process.shutdown(wait=True, cancel_futures=False)
        _assert_seal(seal)
        if args.retire_after_upload_ack:
            # The shared immutable Base/PTD are needed until *all* Hop ACKs.
            shutil.rmtree(scratch / 'base')
            shutil.rmtree(scratch / 'adapter')
        state.update(status='upload_requests_accepted_remote_readback_pending',
                     finished_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                     scratch_after_upload=shutil.disk_usage(scratch).free)
    except BaseException as error:
        state.update(status='failed', error_type=type(error).__name__,
                     finished_utc=dt.datetime.now(dt.timezone.utc).isoformat())
        raise
    finally:
        _save(output / 'result.json', state)
        print('STAGE_C_BATCH_FINAL', state['status'], flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--scratch', required=True, type=Path)
    parser.add_argument('--hops', required=True, nargs='+', type=int)
    parser.add_argument('--replace-hops', nargs='*', type=int, default=[])
    parser.add_argument('--retire-after-upload-ack', action='store_true',
                        help='Delete reproducible local Hop on provider ACK; '
                             'independent remote SHA256 is still PENDING')
    run(parser.parse_args())


if __name__ == '__main__':
    main()
