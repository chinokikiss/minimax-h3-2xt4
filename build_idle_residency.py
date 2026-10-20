"""Freeze qualified opt-in idle scheduling; upload source only, no model data."""
import hashlib
import json
from pathlib import Path
from freeze_pipeline_job import read_bundle,refresh_references,write_job

ROOT=Path(__file__).resolve().parent


def main():
    notebook,files=read_bundle(ROOT/'kaggle_residency_p01/residency_p01.ipynb')
    changed=('patches/host_weight_cache.py','patches/weight_residency.py','patches/comfy_tp_runtime.py',
        'patches/comfy_sampler.py','patches/comfy_idle_lifecycle.py')
    helpers=('benchmarks/idle_residency_pipeline.py','benchmarks/idle_residency_matrix.py',
        'benchmarks/test_idle_window.py','benchmarks/test_idle_lifecycle.py','benchmarks/causal_trace.py',
        'benchmarks/residency_experiment_matrix.py')
    for name in (*changed,*helpers):files[name]=(ROOT/name).read_bytes()
    for policy in ('p0_idle','p0_p1_idle'):
        proof=json.loads((ROOT/f'semantic_results/residency_qualification/{policy}/result.json').read_text())
        if not proof['passed'] or not proof['all_outputs_bit_exact_with_uninstrumented']:raise RuntimeError('Native gate missing')
        for name,sha in proof['source_sha256'].items():
            if (name.startswith('patches/') or name=='benchmarks/causal_trace.py') and name in files and hashlib.sha256(files[name]).hexdigest()!=sha:
                raise RuntimeError('Frozen inference differs from qualification: '+name)
        files[f'semantic_results/{policy}_qualification.json']=json.dumps(proof).encode()
    proof=json.loads((ROOT/'semantic_results/idle_trained_te_reuse.json').read_text())
    if not proof['passed']:raise RuntimeError('Trained TE reuse gate missing')
    for name,sha in proof['source_sha256'].items():
        if name in files and hashlib.sha256(files[name]).hexdigest()!=sha:raise RuntimeError('Trained reuse source changed: '+name)
    files['semantic_results/idle_trained_te_reuse.json']=json.dumps(proof).encode()
    plan=json.loads(files['diagnostic_plan.json'])
    plan['resume']['source_sha256']={n:hashlib.sha256(files[n]).hexdigest() for n in plan['resume']['source_sha256']}
    plan['resume']['scope']='Preserved physical Native outputs; qualified opt-in idle residency, unchanged operators'
    files['diagnostic_plan.json']=json.dumps(plan).encode()
    refresh_references(files)
    bootstrap=files['benchmarks/residency_kaggle_bootstrap.py'].decode().replace('residency_results','idle_residency_results')
    start=bootstrap.index("        command([sys.executable,str(root/'benchmarks/test_residency_policy.py')]")
    end=bootstrap.index('    except BaseException as error:',start)
    bootstrap=bootstrap[:start]+'''        for name in ('test_residency_policy.py','test_residency_spawn.py','test_idle_window.py','test_idle_lifecycle.py'):
            command([sys.executable,str(root/'benchmarks'/name)],120,name+'.log')
        manifest.update(status='idle_pilot',diagnostic_only=False,optimization_changes=True,
            matrix=['p0','p0_idle'],scope='Counterbalanced full physical pilot and zero-gap regression; idle remains off by default');save()
        command([sys.executable,'-u',str(root/'benchmarks/idle_residency_matrix.py'),
            '--output',str(out),'--phase','pilot'],12000,'idle_pilot.log')
        manifest.update(status='complete',seconds=time.monotonic()-start);save()
'''+bootstrap[end:]
    files['benchmarks/idle_kaggle_bootstrap.py']=bootstrap.encode()
    write_job(ROOT,'kaggle_idle_residency',notebook,files,'minimax-h3-idle-residency-pilot',
        'MiniMax H3 Idle Residency Pilot','benchmarks/idle_kaggle_bootstrap.py')


if __name__=='__main__':main()
