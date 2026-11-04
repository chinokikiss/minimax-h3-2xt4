"""Compare measured dual-Hop01/02 IOAB result to actual Hop00 Stage C baselines.

A two-Hop sequential expectation is an extrapolation from Hop00's measured
one-Hop stages, not a historical measurement of two sequential Hops.
Do not treat a Kaggle Kernel push as executed or an upload ACK as remote proof.
"""
import argparse
import datetime as dt
import json
from pathlib import Path


def _utc(value):
    return dt.datetime.fromisoformat(value.replace('Z', '+00:00'))


def _seconds(row, name):
    stage = row.get('stages', {}).get(name)
    if stage is None or stage.get('status') != 'complete':
        raise ValueError('Missing completed stage: ' + name)
    return float(stage['seconds'])


def compare(baseline_path, trial_path):
    baseline = json.loads(Path(baseline_path).read_text(encoding='utf-8'))
    trial = json.loads(Path(trial_path).read_text(encoding='utf-8'))
    if baseline.get('status') != 'upload_requests_accepted_remote_readback_pending':
        raise ValueError('Hop00 baseline has no completed private create acknowledgments')
    if trial.get('status') != 'upload_requests_accepted_remote_readback_pending':
        raise ValueError('Dual-Hop result is not complete; no speedup claim is permitted')
    if trial.get('hops') != [1, 2] or set(trial.get('published', {})) != {'hop_01', 'hop_02'}:
        raise ValueError('Trial does not contain both complete private create requests')
    for part in ('hop_01', 'hop_02'):
        if trial['published'][part]['status'] != 'create_request_accepted_processing_unverified':
            raise ValueError('Trial lacks Kaggle private-create acknowledgment: ' + part)
    baseline_repeated = ('license_source', 'private_create_auth', 'stage_full_base',
                         'download_pinned_ptd', 'preflight_full_50_layer',
                         'bake_hop_00', 'verify_local_hop_00', 'upload_hop_00')
    one_hop = sum(_seconds(baseline, stage) for stage in baseline_repeated)
    sequential_two = 2 * one_hop
    actual = (_utc(trial['finished_utc']) - _utc(trial['started_utc'])).total_seconds()
    if actual <= 0:
        raise ValueError('Trial wall clock is nonpositive')
    hops = {}
    for hop in (1, 2):
        part = f'hop_{hop:02d}'
        admission_name = ('admit_bake_receipt_' + part if
                          'admit_bake_receipt_' + part in trial.get('stages', {}) else
                          'verify_local_' + part)
        hops[part] = dict(device=f'cuda:{hop-1}',
                          bake_seconds=_seconds(trial, 'bake_' + part),
                          local_admission_seconds=_seconds(trial, admission_name),
                          local_admission_method=('bake_write_time_receipt' if
                                                  admission_name.startswith('admit_') else
                                                  'independent_full_sha256'),
                          publish_staging_and_upload_seconds=_seconds(trial, 'upload_' + part),
                          upload_ack_only=True)
    return dict(format='h3-ptd-v2-stage-c-ioab-comparison-v1',
                qualification='measured_trial_vs_hop00_based_extrapolation',
                status='compared_private_create_acks_only_remote_sha256_pending',
                hop00_measured_one_hop_seconds=one_hop,
                baseline_two_hop_sequential_extrapolated_seconds=sequential_two,
                new_dual_hop_measured_wall_seconds=actual,
                effective_hops_per_hour=2 * 3600 / actual,
                extrapolated_speedup_vs_hop00_sequential=sequential_two / actual,
                baseline_stage_seconds={name: _seconds(baseline, name)
                                        for name in baseline_repeated},
                dual_hop_stage_seconds=hops,
                max_concurrent_bake_seconds=max(x['bake_seconds'] for x in hops.values()),
                baseline_single_hop_bake_seconds=_seconds(baseline, 'bake_hop_00'),
                excludes_shared_part=True,
                remote_provider_version_and_sha256_verified=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', required=True, type=Path)
    parser.add_argument('--trial', required=True, type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    value = compare(args.baseline, args.trial)
    if args.output:
        if args.output.exists():
            raise FileExistsError('Refusing to overwrite an earlier comparison')
        args.output.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(value, indent=2))


if __name__ == '__main__':
    main()
