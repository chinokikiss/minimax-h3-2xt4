"""Independently check completeness and arithmetic of downloaded benchmark artifacts."""
import argparse
import hashlib
import io
import json
import math
from pathlib import Path
import statistics
import tarfile

from tools.experiments.summarize_retest import rows_for

SUITES = ['audio', 'video', 'video_tiling', 'te', 'dit', 'te_stack', 'dit_stack']
EXPECTED = {
    'audio': ({'5s','10s','20s'}, {'native','channel_2gpu','tiled_1gpu','tiled_2gpu'}),
    'video': ({'512x512_1F','512x512_2F','768x768_1F','768x1344_1F','768x1344_2F'}, {'native','tp','sp'}),
    'video_tiling': ({'512x512_2F','768x768_1F','768x1344_2F'}, {'native','tiled_1gpu_256','tiled_2gpu_256','tiled_1gpu_384','tiled_2gpu_384'}),
    'te': ({'S64','S128','S256','S512','S1024','S2048'}, {'native','tp','sp'}),
    'dit': ({'S1024','S2048','S4096','S8192'}, {'native','tp','sp'}),
    'te_stack': ({'S64','S128','S256','S512','S1024','S2048'}, {'PP=2 50 blocks'}),
    'dit_stack': ({'S1024','S2048','S4096','S8192'}, {'PP=2 50 blocks'}),
}
GATES = {
    'block': {'rel_l2_max':.01, 'cosine_min':.999},
    'video': {'rel_l2_max':.01, 'cosine_min':.999, 'max_abs_max':.05},
    'audio_channel': {'rel_l2_max':1e-5, 'cosine_min':.999999},
}

def passes(accuracy, gate):
    return bool(accuracy and accuracy.get('valid')
                and accuracy['rel_l2'] <= gate['rel_l2_max']
                and accuracy['cosine'] >= gate['cosine_min']
                and accuracy.get('max_abs',0) <= gate.get('max_abs_max',math.inf))

def audit(root, rounds, suites=None, include_profile=True):
    suites = SUITES if suites is None else suites
    issues, execution_failures, correctness_failures = [], [], []
    timing_rows = sample_count = 0
    source_hash = None
    recorded_inputs = {}
    archive_path = root / 'retest_source.tar.gz'
    if archive_path.exists():
        with tarfile.open(fileobj=io.BytesIO(archive_path.read_bytes()),mode='r:gz') as archive:
            source_hash = hashlib.sha256(archive.extractfile('benchmarks/retest.py').read()).hexdigest()
    else:
        issues.append('Exact source archive missing')
    manifest_path = root / 'retest_manifest.json'
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    jobs = {(j['round'],j['suite']):j for j in manifest.get('jobs',[])}
    if 'completed_utc' not in manifest:
        issues.append('Allocation manifest has no completion timestamp')
    if manifest.get('comfy_sha') != 'd49e888586dd8ae012c0667b33466b815fee07f7':
        issues.append('ComfyUI revision differs')
    profile = root / 'retest_results' / 'round_0' / 'video_profile'
    profile_job = jobs.get((0,'video_profile'))
    if include_profile and (profile_job is None or profile_job.get('returncode') != 0):
        execution_failures.append('video_profile: job missing or failed')
    elif include_profile:
        for name in ['native','tp_rank0','tp_rank1','sp_rank0','sp_rank1']:
            path = profile / f'{name}_profile.json'
            if not path.exists() or not (profile / f'{name}_trace.json').exists():
                issues.append(f'video_profile/{name}: trace or summary missing')
                continue
            record = json.loads(path.read_text())
            expected_shape = [1,14115 if name.startswith('sp') else 28229,2048]
            if record.get('first_block_input_shapes') != [expected_shape] or not record.get('kernels'):
                issues.append(f'video_profile/{name}: unexpected token shape or no CUDA events')
        if not (profile/'accuracy.json').exists():
            issues.append('video_profile: output comparison missing')
    for round_index in range(rounds):
        for suite in suites:
            label = f'round_{round_index}/{suite}'
            directory = root / 'retest_results' / f'round_{round_index}' / suite
            job = jobs.get((round_index,suite))
            if job is None:
                issues.append(f'{label}: job missing')
            elif job.get('returncode') != 0:
                execution_failures.append(f'{label}: process returncode {job.get("returncode")}')
            env_path = directory / 'environment.json'
            if not env_path.exists():
                issues.append(f'{label}: environment missing')
                continue
            env = json.loads(env_path.read_text())
            backend = env.get('kitchen_backends',{}).get('cuda',{})
            if env.get('gpus') != ['Tesla T4','Tesla T4'] or env.get('cuda') != '13.0' or not backend.get('available') or backend.get('disabled'):
                issues.append(f'{label}: incorrect hardware or native CUDA backend')
            if env.get('source_sha256') != source_hash:
                issues.append(f'{label}: source hash differs from archived source')
            expected_packages = {'torch':'2.11.0+cu130','comfy-kitchen':'0.2.37','comfy-aimdo':'0.5.5'}
            if any(env.get('packages',{}).get(k) != v for k,v in expected_packages.items()):
                issues.append(f'{label}: dependency pins differ')
            if env.get('checkpoint_revision') != 'e5eb578a89295337b8ff433a035929ce0279e0b6':
                issues.append(f'{label}: checkpoint revision differs')
            if suite.startswith('dit') and env.get('native_compute_dtype') != 'float32':
                issues.append(f'{label}: DiT baseline must use supported FP32 computation on T4')
            if env.get('warmups') != 3 or env.get('iterations') != 10:
                issues.append(f'{label}: timing protocol differs')
            results_path = directory / 'results.json'
            if not results_path.exists():
                execution_failures.append(f'{label}: no results JSON')
                continue
            seen, mode_failures = set(), set()
            data = json.loads(results_path.read_text())
            gate = GATES['video' if suite == 'video' else 'block'] if suite not in ('audio','video_tiling') else None
            if gate and data.get('gates') != gate:
                issues.append(f'{label}: declared correctness gate differs')
            input_path = directory / 'input_manifest.json'
            inputs = json.loads(input_path.read_text()) if input_path.exists() else {}
            if suite not in ('audio','video_tiling') and set(inputs) != EXPECTED[suite][0]:
                issues.append(f'{label}: saved-input manifest incomplete')
            input_hashes = {name:entry.get('sha256') for name,entry in inputs.items()}
            input_hashes.update({name:entry.get('input_sha256') for name,entry in data.get('workloads',{}).items()
                                 if entry.get('input_sha256')})
            for name,input_hash in input_hashes.items():
                key = (suite,name)
                if not input_hash:
                    issues.append(f'{label}/{name}: input hash missing')
                elif key in recorded_inputs and recorded_inputs[key] != input_hash:
                    issues.append(f'{label}/{name}: input differs between rounds')
                recorded_inputs[key] = input_hash
            for name,mode,value,native in rows_for(suite,data):
                if name == '*':
                    mode_failures.add(mode)
                    execution_failures.append(f'{label}/{mode}: {value.get("status")}')
                    continue
                seen.add((name,mode))
                row_label = f'{label}/{name}/{mode}'
                if 'median_ms' not in value:
                    execution_failures.append(f'{row_label}: no timing ({value.get("status")})')
                    continue
                samples = value.get('samples_ms',[])
                if len(samples) != 10 or value.get('n') != 10 or any(not math.isfinite(s) or s <= 0 for s in samples):
                    issues.append(f'{row_label}: invalid raw timing samples')
                    continue
                timing_rows += 1
                sample_count += len(samples)
                if not math.isclose(statistics.median(samples),value['median_ms'],rel_tol=1e-10):
                    issues.append(f'{row_label}: median does not match raw samples')
                ordered = sorted(samples)
                position = .95 * (len(ordered)-1)
                lower = math.floor(position)
                p95 = ordered[lower] + (ordered[min(lower+1,len(ordered)-1)]-ordered[lower])*(position-lower)
                if not math.isclose(p95,value['p95_ms'],rel_tol=1e-10):
                    issues.append(f'{row_label}: p95 does not match raw samples')
                if 'correctness_passed' in value and not value['correctness_passed']:
                    correctness_failures.append(row_label)
                accuracy = value.get('accuracy',value.get('accuracy_vs_normal'))
                if accuracy and accuracy.get('valid'):
                    if any(not math.isfinite(accuracy[k]) for k in ('cosine','rel_l2','max_abs','mean_abs')) or not (-1 <= accuracy['cosine'] <= 1) or accuracy['rel_l2'] < 0 or accuracy['max_abs'] < 0 or accuracy['mean_abs'] < 0:
                        issues.append(f'{row_label}: impossible metric')
                if mode == 'native' and accuracy and not accuracy.get('valid'):
                    issues.append(f'{row_label}: invalid native output')
                if 'correctness_passed' in value:
                    row_gate = GATES['audio_channel'] if suite == 'audio' else gate
                    if row_gate is None or value['correctness_passed'] != passes(accuracy,row_gate):
                        issues.append(f'{row_label}: correctness decision differs from declared gate')
                if 'matching_tiles_passed' in value and value['matching_tiles_passed'] != passes(value.get('accuracy_vs_matching_tiles'),GATES['audio_channel']):
                    issues.append(f'{row_label}: matching-tile decision differs from declared gate')
                if value.get('accepted_speedup') is not None and not value.get('correctness_passed'):
                    issues.append(f'{row_label}: failed output received a speedup')
                input_hash = value.get('input_sha256')
                if input_hash and name in inputs and input_hash != inputs[name]['sha256']:
                    issues.append(f'{row_label}: input hash differs from saved-input manifest')
            workloads, modes = EXPECTED[suite]
            for name in workloads:
                for mode in modes:
                    if (name,mode) not in seen and mode not in mode_failures:
                        issues.append(f'{label}/{name}/{mode}: comparison missing')
    report = {'artifact_integrity_passed':not issues,'expected_rounds':rounds,
              'selected_suites':suites,
              'source_sha256':source_hash,'comfy_sha':manifest.get('comfy_sha'),
              'input_sha256':{f'{suite}/{name}':value for (suite,name),value in sorted(recorded_inputs.items())},
              'timing_rows':timing_rows,'raw_sample_count':sample_count,
              'integrity_issues':issues,'execution_failures':execution_failures,
              'correctness_failure_count':len(correctness_failures),
              'correctness_failures':correctness_failures}
    (root/'artifact_audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))
    return not issues

if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('root',type=Path)
    parser.add_argument('--expected-rounds',type=int,default=3)
    parser.add_argument('--suites',default=','.join(SUITES))
    parser.add_argument('--skip-profile',action='store_true')
    args=parser.parse_args()
    raise SystemExit(0 if audit(args.root,args.expected_rounds,args.suites.split(','),not args.skip_profile) else 1)
