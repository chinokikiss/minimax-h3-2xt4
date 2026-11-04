"""One-hop Stage C private Kaggle publication on physical dual Tesla T4s.

Each invocation creates private Datasets by default. Explicit --replace-existing
instead publishes new versions and requests old-version deletion on Kaggle.
It only acknowledges upload requests; remote mounts and byte readback are separate.
No inference, implicit overwrite, status polling or automatic GPU retry.
"""
import argparse
import datetime as dt
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'ComfyUI'), str(ROOT / 'benchmarks')]

EXPECTED_RECIPE = 'c8b832fb86d00233a7cee437f73c5de02fa9509b58954fef55671a3012bc3716'
LICENSE_RELATIVE = 'docs/data/h3_ptd_v2_stage_c_20261103/license_pin/LICENSE'
LICENSE_SHA256 = '59b99642b95ea21630e311198ddbfffbfe05aadba0c2f5d884cbdf4efcc90f44'
LICENSE_REVISION = '42ed227ee7df40d41602854ae760620d6eb651fe'
CAMPAIGN = '20261103'
OWNER = 'pamcks'
PARTS = ('shared',) + tuple(f'hop_{i:02d}' for i in range(8))


def dataset_id(part):
    if part not in PARTS:
        raise ValueError('Unexpected Stage C part identity')
    slug = part.replace('_', '-')
    return f'{OWNER}/h3-ptd-v2-{slug}-{CAMPAIGN}'


def _save(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')


def _stage(name, state, output, function):
    started = time.monotonic()
    state['stages'][name] = dict(status='running')
    state['active_stage'] = name
    _save(output / 'result.json', state)
    print('STAGE_C_START', name, flush=True)
    try:
        value = function()
        state['stages'][name] = dict(status='complete', seconds=time.monotonic() - started)
        _save(output / 'result.json', state)
        print('STAGE_C_DONE', name, state['stages'][name]['seconds'], flush=True)
        return value
    except BaseException as error:
        state['stages'][name] = dict(status='failed', seconds=time.monotonic() - started,
                                    error_type=type(error).__name__)
        _save(output / 'result.json', state)
        raise


def _legal_files(source, scratch):
    """Use the independently pinned official H3 license shipped in source overlay."""
    from patches.tp2_bundle import sha
    license_path = source / LICENSE_RELATIVE
    content = license_path.read_bytes()
    if (b'MiniMax H3 COMMUNITY LICENSE AGREEMENT' not in content
            or len(content) != 17604 or sha(license_path) != LICENSE_SHA256):
        raise ValueError('Source-pinned official H3 license hash/bytes mismatch')
    return license_path, LICENSE_SHA256, LICENSE_REVISION


def _upload_one(part, root, scratch, license_path, state, output, *,
                replace_existing=False, admitted_proof=None):
    """One explicit private create or replacement; no auto-fallback/retry."""
    from kaggle.api.kaggle_api_extended import KaggleApi
    from patches.ptd.publish_v2 import prepare_private_kaggle_part
    from patches.tp2_bundle import sha

    if part not in PARTS:
        raise ValueError('Unknown part')
    receipt = root / 'receipt.json'
    original_receipt_sha = sha(receipt)
    identifier = dataset_id(part)
    staging = scratch / 'publish_staging' / part
    staged = prepare_private_kaggle_part(root, part, staging,
        dataset_id=identifier, license_name='other',
        title=f'H3 PTD v2 {part.replace("_", " ")} {CAMPAIGN}',
        verified_proof=admitted_proof)
    if staged['source_receipt_sha256'] != original_receipt_sha:
        raise ValueError('PTD staging changed source receipt')
    shutil.copy2(license_path, staging / 'LICENSE-MiniMax-H3.txt')
    (staging / 'NOTICE.txt').write_text(
        'MiniMax H3 is licensed under the MiniMax H3 Community License Agreement, '
        'Copyright © 2026 MiniMax. All Rights Reserved.\n'
        'Modified derivative: PTD/RefAsymOPD and optional ordinary LoRA '
        'were merged into the Native TP2 INT8 ConvRot256 weight format. '
        'Part identifier: ' + part + '.\n'
        'This is a private archival model-weight dataset, not a public release.\n',
        encoding='utf-8')
    # The uncompressed tar is created by Kaggle SDK for payload/, and can
    # transiently occupy another complete part's worth of scratch space.
    payload_bytes = int(staged['payload_bytes'])
    free = shutil.disk_usage(staging).free
    required = payload_bytes + 2 * (2**30)
    if free < required:
        raise OSError(f'Upload tar headroom insufficient: need {required} bytes, have {free}')
    api = KaggleApi()
    api.authenticate()
    state['published'][part] = dict(status='upload_in_progress', identifier=identifier,
        source_receipt_sha256=original_receipt_sha, payload_bytes=payload_bytes,
        license_file='LICENSE-MiniMax-H3.txt', notice_file='NOTICE.txt',
        dir_mode='tar', private=True,
        operation='replace_delete_old_versions' if replace_existing else 'create_new')
    _save(output / 'result.json', state)
    print('STAGE_C_PRIVATE_UPLOAD_START', part, payload_bytes, flush=True)
    started = time.monotonic()
    try:
        if replace_existing:
            response = api.dataset_create_version(
                str(staging), version_notes='H3 PTD v2 verified-part replacement',
                quiet=True, convert_to_csv=False, delete_old_versions=True,
                dir_mode='tar')
        else:
            response = api.dataset_create_new(str(staging), public=False, quiet=True,
                convert_to_csv=False, dir_mode='tar')
    except BaseException as error:
        # SDK exception messages or URLs may contain secrets. Keep type only.
        raise RuntimeError('Kaggle private upload failed: ' + type(error).__name__) from None
    status = str(getattr(response, 'status', '')).lower()
    if status != 'ok' or getattr(response, 'error', None):
        raise RuntimeError('Kaggle private upload request was not accepted')
    entry = state['published'][part]
    entry.update(status=('version_request_accepted_processing_unverified' if replace_existing
                         else 'create_request_accepted_processing_unverified'),
        upload_seconds=time.monotonic() - started,
        url=str(getattr(response, 'url', '') or ''),
        remote_readback='pending', version_pin='pending')
    _save(output / 'result.json', state)
    print('STAGE_C_PRIVATE_UPLOAD_ACK', part, identifier,
          'replace' if replace_existing else 'create', flush=True)
    return entry


def run(args):
    source = args.source.resolve()
    output = args.output.resolve()
    scratch = args.scratch.resolve()
    if output.exists() or scratch.exists():
        raise FileExistsError('Preserve previous Stage C outputs and scratch')
    if args.hop not in range(8):
        raise ValueError('Stage C hop must be 0..7')
    output.mkdir(parents=True)
    scratch.mkdir(parents=True)
    state = dict(format='h3-ptd-v2-stage-c-one-hop-publish-v1',
        status='starting', started_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        hop=args.hop, layers=50, owner=OWNER, campaign=CAMPAIGN,
        publication_mode='replace_delete_old_versions' if args.replace_existing else 'create_new',
        private_dataset_create_only=not args.replace_existing,
        external_publication_authorized=True,
        no_status_list_or_live_log_polling=True, no_model_e2e=True,
        stages={}, published={}, source_recipe_sha256=EXPECTED_RECIPE)
    _save(output / 'result.json', state)

    try:
        import torch
        from patches.legacy_runtime import validate_packages, validate_source
        from patches.ptd.checkpoint import RELEASE, download_on_t4
        from patches.ptd.bake_v2 import bake_shared_part, bake_hop_part
        from patches.ptd.publish_v2 import admit_baked_part
        from patches.tp2_bundle import sha
        from preflight_ptd_v2_bake import inspect_inputs
        from tp2_public_dit_assets import stage as stage_base

        validate_packages()
        validate_source(source / 'ComfyUI')
        if torch.cuda.device_count() != 2:
            raise RuntimeError('Stage C Baker requires exactly 2 physical T4 GPUs')
        devices = [dict(index=i, name=torch.cuda.get_device_name(i),
            capability=list(torch.cuda.get_device_capability(i)),
            uuid=str(torch.cuda.get_device_properties(i).uuid)) for i in range(2)]
        if any(d['capability'] != [7, 5] or 'T4' not in d['name'] for d in devices):
            raise RuntimeError('Stage C Baker requires two Tesla T4 GPUs')
        if len({d['uuid'] for d in devices}) != 2:
            raise RuntimeError('Physical T4 UUIDs must be distinct')
        state['devices'] = devices
        torch.set_grad_enabled(False)
        torch.set_num_threads(2)
        torch.backends.cuda.matmul.allow_tf32 = False
        os.environ.update(HF_HUB_DISABLE_XET='1', HF_HUB_DISABLE_PROGRESS_BARS='1',
                          OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', MALLOC_ARENA_MAX='2')

        license_path, license_sha, license_revision = _stage('license_source',
            state, output, lambda: _legal_files(source, scratch))
        state['license'] = dict(source_revision=license_revision, sha256=license_sha)
        # Check native Kaggle authentication BEFORE spending T4 time on Bake.
        def auth_gate():
            from kaggle.api.kaggle_api_extended import KaggleApi
            api=KaggleApi()
            api.authenticate()
            return True
        _stage('private_create_auth', state, output, auth_gate)
        state['scratch_before_download'] = shutil.disk_usage(scratch).free
        base = _stage('stage_full_base', state, output,
            lambda: stage_base(source, scratch / 'base'))
        checkpoint = _stage('download_pinned_ptd', state, output,
            lambda: download_on_t4(scratch / 'adapter'))
        if sha(checkpoint) != RELEASE['sha256']:
            raise ValueError('PTD full release checksum differs from pinned source')
        parts_root = scratch / 'parts'
        parts_root.mkdir()
        admission = _stage('preflight_full_50_layer', state, output,
            lambda: inspect_inputs(base, checkpoint, parts_root, layers=50,
                allow_pruned_adaln=True, verify_ptd_sha256=True, check_t4=True))
        _save(output / 'source_preflight.json', admission)
        if not admission['physical_baker_ready']:
            raise ValueError('Physical production PTD preflight failed')
        if args.hop == 0:
            shared = _stage('bake_shared', state, output, lambda: bake_shared_part(
                base, checkpoint, parts_root / 'shared', layers=50,
                allow_pruned_adaln=True))
            shutil.copy2(shared, output / 'shared_receipt.json')
        hop_id = f'hop_{args.hop:02d}'
        hop_receipt = _stage('bake_' + hop_id, state, output,
            lambda: bake_hop_part(base, checkpoint, args.hop, parts_root / hop_id,
                layers=50, devices=('cuda:0', 'cuda:1'),
                allow_pruned_adaln=True))
        shutil.copy2(hop_receipt, output / (hop_id + '_receipt.json'))
        recipe_sha = json.loads(hop_receipt.read_text())['recipe_sha256']
        if recipe_sha != EXPECTED_RECIPE:
            raise ValueError('Stage C bake recipe differs from Stage B source qualification')
        if args.hop == 0:
            if json.loads(shared.read_text())['recipe_sha256'] != recipe_sha:
                raise ValueError('Shared/first Hop recipe mismatch')
        produced = ['shared', hop_id] if args.hop == 0 else [hop_id]
        local_admissions = {}
        for part in produced:
            root = parts_root / part
            proof = _stage('admit_bake_receipt_' + part, state, output,
                lambda root=root,part=part: admit_baked_part(root, part))
            local_admissions[part] = proof
            _save(output / (part + '_local_admission.json'), proof)
        (scratch / 'publish_staging').mkdir()
        for part in produced:
            root = parts_root / part
            _stage('upload_' + part, state, output,
                lambda root=root,part=part: _upload_one(
                    part, root, scratch, license_path, state, output,
                    replace_existing=args.replace_existing,
                    admitted_proof=local_admissions[part]))
        state.update(status='upload_requests_accepted_remote_readback_pending',
            finished_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
            scratch_after_upload=shutil.disk_usage(scratch).free,
            note='No remote processing, provider visibility/version, mount bytes or E2E qualified')
    except BaseException as error:
        state.update(status='failed', error_type=type(error).__name__,
            finished_utc=dt.datetime.now(dt.timezone.utc).isoformat())
        # Avoid persisting potentially credential-bearing exception text.
        (output / 'exception_type.txt').write_text(type(error).__name__ + '\n',
            encoding='utf-8')
        raise
    finally:
        _save(output / 'result.json', state)
        print('STAGE_C_FINAL', state['status'], flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', required=True, type=Path)
    p.add_argument('--hop', required=True, type=int)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--scratch', required=True, type=Path)
    p.add_argument('--replace-existing', action='store_true',
                   help='Publish a new version and delete old versions (destructive; explicit opt-in)')
    args = p.parse_args()
    run(args)


if __name__ == '__main__':
    main()
