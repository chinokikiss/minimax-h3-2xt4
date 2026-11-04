"""Stage C source-bound, no-side-effect planning from physical Stage B evidence.

This never runs GPU, downloads weights, publishes datasets, assumes remote
quota, or treats Kaggle scratch as durable storage.
"""
import argparse
import hashlib
import json
from pathlib import Path

PARTS = ('shared',) + tuple(f'hop_{i:02d}' for i in range(8))
EXPECTED_PTD_SHA256 = '6796802b9150110a3cd7f66891c9e6190502a432c30b0fa1aa299724cfc15cc3'


def _read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def build_plan(evidence_root):
    root = Path(evidence_root).resolve()
    result = _read(root / 'result.json')
    preflight = _read(root / 'source_preflight.json')
    shared = _read(root / 'shared_receipt.json')
    hop0 = _read(root / 'hop_00_receipt.json')
    if (result.get('status') != 'complete'
            or any(stage.get('status') != 'complete'
                   for stage in result.get('stages', {}).values())
            or len(result.get('stages', {})) != 6):
        raise ValueError('Stage B physical result is not 6/6 complete')
    if (result.get('layers') != 50 or result.get('hop') != 0
            or result.get('shards') != 100
            or result.get('verified_merge_targets') != 208):
        raise ValueError('Stage B is not the full 50-layer single-hop qualification')
    if (preflight.get('status') != 'passed'
            or not preflight.get('release_sha256_checked')
            or not preflight.get('physical_baker_ready')
            or preflight.get('issues')):
        raise ValueError('Stage B production-source preflight was not passed')
    if (result.get('ptd_sha256') != EXPECTED_PTD_SHA256
            or preflight.get('checkpoint_sha256') != EXPECTED_PTD_SHA256):
        raise ValueError('Pinned complete PTD source hash mismatch')
    recipe = shared.get('recipe', {})
    recipe_sha = hashlib.sha256(json.dumps(recipe, sort_keys=True).encode()).hexdigest()
    if (recipe.get('layers') != 50 or recipe.get('hops') != 8
            or recipe.get('steps') != 32
            or recipe.get('ptd_checkpoint_sha256') != EXPECTED_PTD_SHA256
            or result.get('recipe_sha256') != recipe_sha
            or shared.get('recipe_sha256') != recipe_sha
            or hop0.get('recipe_sha256') != recipe_sha):
        raise ValueError('Full 8-hop source recipe is not anchored to Stage B')
    names = {f'dit.block{i:02d}.rank{r}' for i in range(50) for r in (0, 1)}
    if set(hop0.get('shards', {})) != names:
        raise ValueError('Stage B Hop 0 coverage incomplete')
    if len(hop0.get('verified_targets', [])) != 208:
        raise ValueError('Stage B merged-target coverage incomplete')
    proof = result.get('local_part_validation', {})
    if (proof.get('status') != 'verified' or proof.get('part') != 'hop_00'
            or proof.get('payload_files') != 202
            or proof.get('layers') != 50 or proof.get('hop') != 0
            or proof.get('recipe_sha256') != recipe_sha):
        raise ValueError('Stage B source-local hash proof unavailable')
    hop_bytes = int(proof['payload_bytes'])
    # Shared "payload" excludes receipt.json; Hop-0 payload similarly excludes
    # receipt, so this is a minimum expected published model-data size.
    shared_bytes = sum(int(shared[key]) for key in
                       ('aux_bytes', 'native_header_bytes', 'block_plan_bytes'))
    if hop_bytes <= 0 or shared_bytes <= 0:
        raise ValueError('Invalid Stage B payload accounting')
    return {
        'format': 'h3-ptd-v2-stage-c-plan-v1',
        'status': 'source_qualified_publication_not_started',
        'stage_b_result': str(root / 'result.json'),
        'stage_b_scope': 'physical dual T4 full 50-layer Hop 0 local bake',
        'source_recipe_sha256': recipe_sha,
        'source_ptd_sha256': EXPECTED_PTD_SHA256,
        'steps': 32,
        'hops': 8,
        'layers': 50,
        'part_order': list(PARTS),
        'nominal_hop_payload_bytes': hop_bytes,
        'nominal_shared_payload_bytes': shared_bytes,
        'estimated_eight_hop_plus_shared_bytes': 8*hop_bytes + shared_bytes,
        'estimate_excludes': ['Dataset metadata, receipt JSON, upload archives',
                              'other datasets, processing overhead, and retained versions'],
        'release_requirements': {
            'dataset_license_and_derivative_redistribution_review': 'not_verified',
            'account_private_quota_and_dataset_size_limits': 'not_verified',
            'private_dataset_create_auth': 'small_synthetic_probe_only',
            'large_model_upload_and_remote_byte_readback': 'not_verified',
            'published_parts': 0,
            'remote_readback_proofs': 0,
            'bank_assembly': 'blocked_until_nine_version_pinned_remote_proofs'
        },
        'execution_protocol': [
            'Reserve separate immutable private Dataset identities for shared and all eight hops; never replace versions in place.',
            'Re-bake Shared and Hop 00 from full pinned PTD source: previous Stage B scratch payload was not preserved.',
            'For each new part: local verify_part -> prepare private zero-copy staging -> private create -> confirm processed version.',
            'Verify remote mounted dataset bytes against separately preserved source receipt SHA256, record version-pinned readback proof.',
            'Only after remote SHA256 proof and durable publication identity may local scratch for that part be released.',
            'Repeat serially for Hops 01..07 under checked private quota and per-part scratch headroom.',
            'Mount all nine immutable part versions and verify nine remote readback proofs; assemble production Bank v2 manifest last.',
            'No Stage C completion or production/PTD E2E quality claims until all nine remote parts have been verified.'
        ],
        'safe_to_upload_now': False,
        'remarks': 'Plan is source-admitted; remote quota, license, and publication/remote readback remain separate gates.'
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    path = args.output.resolve()
    if path.exists():
        raise FileExistsError('Do not overwrite Stage C plan evidence')
    if not path.parent.is_dir():
        raise FileNotFoundError(path.parent)
    plan = build_plan(args.evidence_root)
    path.write_text(json.dumps(plan, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({
        'status': plan['status'], 'recipe': plan['source_recipe_sha256'],
        'estimate_bytes': plan['estimated_eight_hop_plus_shared_bytes'],
        'output': str(path)
    }, indent=2), flush=True)


if __name__ == '__main__':
    main()
