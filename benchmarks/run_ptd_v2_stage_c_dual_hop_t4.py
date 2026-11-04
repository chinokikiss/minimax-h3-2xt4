"""Opt-in dual-T4 independent Hop01/02 Bake + continuously overlapped I/O upload.

One physical T4 owns each full Hop (both TP2 ranks), with bounded per-layer CPU
prefetch and durable-write overlap. One verified source is downloaded/staged once.
Each Hop reuses its fsynced Baker SHA256 receipt for metadata-only local
admission and hardlink staging; remote independent SHA256 runs separately.
Kaggle creates run serially, Hop01 ACK before Hop02 starts.
Never poll Kaggle jobs.
Experiment uses isolated IOAB Dataset identities; production Hop01/02 are untouched.
Existing IOAB datasets can be replaced only with --replace-existing.
"""
import argparse
from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import datetime as dt
import hashlib
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
    CAMPAIGN, EXPECTED_RECIPE, _legal_files)
from patches.ptd.bake_v2 import FORMAT_V2

HOPS = (1, 2)
OWNER = 'pamcks'
TEST_TAG = 'ioab'
FORMAT = 'h3-ptd-v2-stage-c-independent-dual-hop-ioab-v1'


def dataset_id(hop):
    if hop not in HOPS:
        raise ValueError('Only Hop01/02 are admitted in this isolated experiment')
    return f'{OWNER}/h3-ptd-v2-hop-{hop:02d}-{TEST_TAG}-{CAMPAIGN}'


def _save(path, data):
    # Snapshot is durable for final outcomes; thread locking belongs to caller.
    tmp = path.with_suffix(path.suffix + '.partial')
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    os.replace(tmp, path)


def _identity(path):
    st = Path(path).stat()
    return (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns)


def _source_seal(base, checkpoint):
    from patches.tp2_bundle import BundleReader
    reader = BundleReader(base)
    return {str(p): _identity(p) for p in
            (reader.path, reader.bundle_file, Path(checkpoint).resolve())}


def _assert_seal(seal):
    for path, identity in seal.items():
        if _identity(path) != tuple(identity):
            raise RuntimeError('PTD experiment source identity changed after initial SHA256 gate')


def _prepared_recipe(base, checkpoint, admission):
    """Reuse pinned digest from single shared preflight: no per-Hop 11 GiB rehash."""
    from patches.tp2_bundle import BundleReader, sha
    from patches.ptd.checkpoint import PTDCheckpoint, RELEASE
    reader = BundleReader(base)
    trained = PTDCheckpoint(checkpoint, reader)
    proof = admission['base']
    if (not admission['release_sha256_checked']
            or admission['checkpoint_sha256'] != RELEASE['sha256']
            or reader.bundle_digest != proof['bundle_index_sha256']
            or proof['descriptor_sha256'] != sha(reader.path)):
        raise ValueError('Pinned shared PTD/Base SHA256 qualification missing')
    recipe = dict(format=FORMAT_V2, layers=50, hops=trained.hops, steps=trained.steps,
                  video_shift=12., audio_shift=3., quantization='native-convrot256-int8-tp2',
                  base_bundle_sha256=reader.bundle_digest,
                  base_descriptor_sha256=proof['descriptor_sha256'],
                  ptd_checkpoint_sha256=admission['checkpoint_sha256'],
                  ordinary_loras=[],
                  pruned_adaln_approximation='adaln_t_table' in reader.header)
    digest = hashlib.sha256(json.dumps(recipe, sort_keys=True).encode()).hexdigest()
    if digest != EXPECTED_RECIPE:
        raise ValueError('Independent-Hop recipe differs from qualified Hop00 baseline')
    return recipe


def _bake_worker(base, checkpoint, hop, part_dir, device, recipe, seal):
    import torch
    from patches.ptd.bake_v2 import bake_hop_part
    _assert_seal(seal)
    torch.set_grad_enabled(False)
    torch.set_num_threads(2)
    started = time.monotonic()
    receipt = bake_hop_part(base, checkpoint, hop, part_dir, layers=50,
                            devices=(device,), allow_pruned_adaln=True,
                            independent_hop_gpu=True, prefetch_layers=True,
                            preverified_recipe=recipe)
    _assert_seal(seal)
    value = json.loads(receipt.read_text(encoding='utf-8'))
    if (value['recipe_sha256'] != EXPECTED_RECIPE
            or value['hop'] != hop or len(value['shards']) != 100):
        raise ValueError('Independent-Hop receipt/coverage differs from qualified Bank v2')
    return dict(hop=hop, seconds=time.monotonic() - started,
                receipt=str(receipt), device=device,
                shards=len(value['shards']))


class UploadSkipped(RuntimeError):
    """Hop02 never requests a Dataset create unless Hop01 received an ACK."""


def _provider_error_tag(error):
    """Small, credential-free diagnostic for a Kaggle SDK transport failure."""
    pieces = [type(error).__name__]
    response = getattr(error, 'response', None)
    status = getattr(response, 'status_code', None)
    if isinstance(status, int):
        pieces.append(f'http_{status}')
    errno = getattr(error, 'errno', None)
    if isinstance(errno, int):
        pieces.append(f'errno_{errno}')
    return ':'.join(pieces)


def _publish_one(hop, parts_root, scratch, license_path, output, run_stage,
                 *, previous_upload=None, replace_existing=False):
    """Receipt-admit/stage alongside baking; serialize uploads, no retries."""
    from kaggle.api.kaggle_api_extended import KaggleApi
    from patches.ptd.publish_v2 import admit_baked_part, prepare_private_kaggle_part
    from patches.tp2_bundle import sha

    part = f'hop_{hop:02d}'
    root = parts_root / part
    receipt_sha = sha(root / 'receipt.json')
    proof = run_stage('admit_bake_receipt_' + part,
                      lambda: admit_baked_part(root, part))
    if proof['receipt_sha256'] != receipt_sha:
        raise ValueError('PTD Baker receipt changed during admission')
    # Don't store local inode snapshots as portable readback proofs.
    portable_proof = {key: value for key, value in proof.items()
                      if key != 'source_snapshot'}
    _save(output / (part + '_local_admission.json'), portable_proof)
    identifier = dataset_id(hop)

    def upload():
        staging = scratch / 'publish_staging' / part
        staged = prepare_private_kaggle_part(
            root, part, staging, dataset_id=identifier, license_name='other',
            title=f'H3 PTD v2 Hop {hop:02d} IOAB {CAMPAIGN}',
            verified_proof=proof)
        if staged['source_receipt_sha256'] != receipt_sha:
            raise RuntimeError('PTD staging source receipt proof changed')
        shutil.copy2(license_path, staging / 'LICENSE-MiniMax-H3.txt')
        (staging / 'NOTICE.txt').write_text(
            'MiniMax H3 Community License Agreement; private derivative archive.\n'
            f'Independent-T4 I/O overlap trial: {part}.\n', encoding='utf-8')
        payload_bytes = int(staged['payload_bytes'])
        if shutil.disk_usage(staging).free < payload_bytes + 2 * 2**30:
            raise OSError('Kaggle upload tar scratch headroom insufficient')
        if hop == 2:
            if previous_upload is None:
                raise ValueError('Hop02 requires a Hop01 upload gate')
            print('STAGE_C_IOAB_UPLOAD_WAIT_HOP01', part, flush=True)
            if not previous_upload.result():
                raise UploadSkipped('Hop02 upload skipped: Hop01 did not receive an upload ACK')
        api = KaggleApi()
        api.authenticate()
        print('STAGE_C_IOAB_UPLOAD_START', part, payload_bytes, flush=True)
        began = time.monotonic()
        try:
            if replace_existing:
                response = api.dataset_create_version(
                    str(staging), version_notes='H3 PTD v2 IOAB verified-part replacement',
                    quiet=True, convert_to_csv=False, delete_old_versions=True,
                    dir_mode='tar')
            else:
                response = api.dataset_create_new(str(staging), public=False, quiet=True,
                                                  convert_to_csv=False, dir_mode='tar')
        except BaseException as error:
            tag = _provider_error_tag(error)
            print('STAGE_C_IOAB_PROVIDER_ERROR', part, tag, flush=True)
            raise RuntimeError('Kaggle private upload error: ' + tag) from None
        if str(getattr(response, 'status', '')).lower() != 'ok' or getattr(response, 'error', None):
            # Never log the SDK response body: it may contain signed URLs.
            raise RuntimeError('Kaggle private upload request was not accepted')
        print('STAGE_C_IOAB_UPLOAD_ACK', part, identifier,
              'replace' if replace_existing else 'create', flush=True)
        return dict(status=('version_request_accepted_processing_unverified' if replace_existing
                            else 'create_request_accepted_processing_unverified'),
                    operation='replace_delete_old_versions' if replace_existing else 'create_new',
                    dataset_id=identifier, receipt_sha256=receipt_sha,
                    payload_bytes=payload_bytes, upload_seconds=time.monotonic()-began,
                    remote_readback='pending', version_pin='pending', private=True)
    return run_stage('upload_' + part, upload)


def run(args):
    source = args.source.resolve()
    scratch = args.scratch.resolve()
    output = args.output.resolve()
    if scratch.exists() or output.exists():
        raise FileExistsError('Never overwrite an existing IOAB scratch or notebook result')
    scratch.mkdir(parents=True)
    output.mkdir(parents=True)
    state = dict(format=FORMAT, status='starting', campaign=CAMPAIGN,
                 hops=[1, 2], mapping={'cuda:0': 'hop_01', 'cuda:1': 'hop_02'},
                 source_recipe_sha256=EXPECTED_RECIPE,
                 concurrent_gpu_workers=2, prefetch_depth=1, durable_write_depth=1,
                 dataset_create_concurrency=1, dataset_create_order=[1, 2],
                 publication_mode='replace_delete_old_versions' if args.replace_existing else 'create_new',
                 private_dataset_create_only=not args.replace_existing,
                 remote_readback='pending',
                 started_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                 stages={}, published={}, worker_results={},
                 no_status_list_or_live_log_polling=True)
    lock = threading.Lock()
    _save(output / 'result.json', state)

    def stage(name, function):
        t0 = time.monotonic()
        with lock:
            state['stages'][name] = dict(status='running', started_s=t0)
            _save(output / 'result.json', state)
        print('STAGE_C_IOAB_START', name, flush=True)
        try:
            value = function()
        except BaseException as exc:
            with lock:
                state['stages'][name].update(
                    status='skipped' if isinstance(exc, UploadSkipped) else 'failed',
                    seconds=time.monotonic()-t0, error_type=type(exc).__name__,
                    error_tag=(str(exc) if name.startswith('upload_')
                               and str(exc).startswith(('Kaggle private upload error:',
                                                       'Hop02 upload skipped:')) else None))
                _save(output / 'result.json', state)
            raise
        with lock:
            state['stages'][name].update(status='complete',
                seconds=time.monotonic()-t0)
            _save(output / 'result.json', state)
        print('STAGE_C_IOAB_DONE', name, state['stages'][name]['seconds'], flush=True)
        return value

    try:
        import torch
        from kaggle.api.kaggle_api_extended import KaggleApi
        from patches.legacy_runtime import validate_packages, validate_source
        from patches.ptd.checkpoint import RELEASE, download_on_t4
        from benchmarks.preflight_ptd_v2_bake import inspect_inputs
        from benchmarks.tp2_public_dit_assets import stage as stage_base

        validate_packages()
        validate_source(source / 'ComfyUI')
        if torch.cuda.device_count() != 2:
            raise RuntimeError('IOAB requires exactly two physical Tesla T4s')
        devices = [dict(index=i, name=torch.cuda.get_device_name(i),
                        capability=list(torch.cuda.get_device_capability(i)),
                        uuid=str(torch.cuda.get_device_properties(i).uuid)) for i in range(2)]
        if (any(d['capability'] != [7, 5] or 'T4' not in d['name'] for d in devices)
                or len({d['uuid'] for d in devices}) != 2):
            raise RuntimeError('IOAB requires two distinct physical T4s')
        state['devices'] = devices
        torch.set_num_threads(2)
        torch.backends.cuda.matmul.allow_tf32 = False
        os.environ.update(HF_HUB_DISABLE_XET='1', HF_HUB_DISABLE_PROGRESS_BARS='1',
                          OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', MALLOC_ARENA_MAX='2')
        license_path, license_sha, license_rev = stage(
            'license_source', lambda: _legal_files(source, scratch))
        state['license'] = dict(sha256=license_sha, revision=license_rev)
        stage('private_create_auth', lambda: KaggleApi().authenticate())
        # Independent source trees; overlap network transfer and disk staging.
        # The single full PTD SHA256 admission happens in inspect_inputs below.
        with ThreadPoolExecutor(max_workers=2, thread_name_prefix='ptd-v2-source-io') as loader:
            base_future = loader.submit(stage, 'stage_full_base',
                                        lambda: stage_base(source, scratch / 'base'))
            ptd_future = loader.submit(stage, 'download_pinned_ptd',
                                       lambda: download_on_t4(scratch / 'adapter'))
            base = base_future.result()
            checkpoint = ptd_future.result()
        ptd_verification = dict(sha256=RELEASE['sha256'],
                                fingerprint=_identity(checkpoint),
                                verifier='download_on_t4')
        parts_root = scratch / 'parts'
        parts_root.mkdir()
        (scratch / 'publish_staging').mkdir()
        admission = stage('preflight_full_50_layer',
            lambda: inspect_inputs(base, checkpoint, parts_root, layers=50,
                allow_pruned_adaln=True, verify_ptd_sha256=True, check_t4=True,
                preverified_checkpoint=ptd_verification))
        _save(output / 'source_preflight.json', admission)
        if not admission['physical_baker_ready']:
            raise RuntimeError('Physical 50-layer PTD preflight not qualified')
        recipe = stage('reuse_source_recipe', lambda: _prepared_recipe(base, checkpoint, admission))
        seal = _source_seal(base, checkpoint)
        # Two independent Hop payloads + two Kaggle tar staging archives.
        free = shutil.disk_usage(scratch).free
        need = 2 * admission['scratch']['minimum_free_bytes'] + 2 * 23 * (2**30)
        if free < need:
            raise OSError(f'Insufficient space for two concurrent Hop bakes/uploads: {free} < {need}')
        state['scratch_admission'] = dict(free=free, minimum=need)

        # GPU0 and GPU1 each produce a whole TP2-formatted Hop, not TP2 compute.
        # Publish thread begins as soon as one Hop completes, without waiting
        # for the other Hop's GPU compute/preload/flush to finish.
        upload_futures = {}
        first_upload = Future()
        def release_hop02_gate(future):
            try:
                future.result()
            except BaseException:
                first_upload.set_result(False)
            else:
                first_upload.set_result(True)

        with ThreadPoolExecutor(max_workers=2, thread_name_prefix='ptd-v2-ioab-upload') as uploader, \
             ProcessPoolExecutor(max_workers=2, mp_context=mp.get_context('spawn')) as bakers:
            bake_start = time.monotonic()
            with lock:
                for hop in HOPS:
                    state['stages'][f'bake_hop_{hop:02d}'] = dict(
                        status='running', started_s=bake_start)
                _save(output / 'result.json', state)
            bake_futures = {
                bakers.submit(_bake_worker, str(base), str(checkpoint), hop,
                              str(parts_root / f'hop_{hop:02d}'), f'cuda:{hop-1}',
                              recipe, seal): hop for hop in HOPS}
            errors = []
            for future in as_completed(bake_futures):
                hop = bake_futures[future]
                try:
                    baked = future.result()
                    with lock:
                        state['worker_results'][f'hop_{hop:02d}'] = baked
                        state['stages'][f'bake_hop_{hop:02d}'].update(
                            status='complete', seconds=time.monotonic()-bake_start)
                        _save(output / 'result.json', state)
                    shutil.copy2(baked['receipt'], output/f'hop_{hop:02d}_receipt.json')
                    uploaded = uploader.submit(
                        _publish_one, hop, parts_root, scratch, license_path,
                        output, stage, previous_upload=first_upload if hop == 2 else None,
                        replace_existing=args.replace_existing)
                    upload_futures[uploaded] = hop
                    if hop == 1:
                        uploaded.add_done_callback(release_hop02_gate)
                except BaseException as exc:
                    errors.append(type(exc).__name__)
                    with lock:
                        state['stages'][f'bake_hop_{hop:02d}'].update(
                            status='failed', seconds=time.monotonic()-bake_start,
                            error_type=type(exc).__name__)
                        _save(output / 'result.json', state)
                    print('STAGE_C_IOAB_BAKE_FAILED', hop, type(exc).__name__, flush=True)
            if 1 not in upload_futures.values():
                # Hop01 Bake failed: Hop02 must never wait forever or upload.
                first_upload.set_result(False)
            for future in as_completed(upload_futures):
                hop = upload_futures[future]
                try:
                    published = future.result()
                    with lock:
                        state['published'][f'hop_{hop:02d}'] = published
                        _save(output / 'result.json', state)
                except BaseException as exc:
                    errors.append(type(exc).__name__)
                    print('STAGE_C_IOAB_UPLOAD_FAILED', hop, type(exc).__name__, flush=True)
            if errors:
                raise RuntimeError('Independent dual-Hop experiment failed in '+','.join(errors))
        _assert_seal(seal)
        state.update(status='upload_requests_accepted_remote_readback_pending',
                     scratch_after_upload=shutil.disk_usage(scratch).free)
    except BaseException as error:
        state.update(status='failed', error_type=type(error).__name__)
        raise
    finally:
        state['finished_utc'] = dt.datetime.now(dt.timezone.utc).isoformat()
        with lock:
            _save(output / 'result.json', state)
        print('STAGE_C_IOAB_FINAL', state['status'], flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--scratch', required=True, type=Path)
    parser.add_argument('--replace-existing', action='store_true',
                        help='Replace both IOAB Datasets and delete prior versions (destructive opt-in)')
    run(parser.parse_args())


if __name__ == '__main__':
    main()
