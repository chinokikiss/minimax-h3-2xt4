"""Physical P0 diagnostic only, frozen before idle-residency implementation."""
import hashlib
import json
from pathlib import Path
from freeze_pipeline_job import read_bundle,refresh_references,write_job

ROOT=Path(__file__).resolve().parent


def main():
    notebook,files=read_bundle(ROOT/'kaggle_residency_p01/residency_p01.ipynb')
    proof=json.loads((ROOT/'semantic_results/residency_qualification/p0_block_profile/result.json').read_text())
    if not proof['passed'] or not proof['all_outputs_bit_exact_with_uninstrumented']:raise RuntimeError('Observer qualification required')
    for name,sha in proof['observer_source_sha256'].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=sha:raise RuntimeError('Observer changed after qualification')
    for name,sha in proof['source_sha256'].items():
        if name.startswith('patches/') and name in files and hashlib.sha256(files[name]).hexdigest()!=sha:
            raise RuntimeError('Frozen inference differs from qualified local source: '+name)
    for name in ('benchmarks/te_block_telemetry.py','benchmarks/te_causal_pipeline.py',
                 'analyze_te_causal_profile.py','benchmarks/test_te_causal_profile.py'):
        files[name]=(ROOT/name).read_bytes()
    refresh_references(files)
    bootstrap=files['benchmarks/residency_kaggle_bootstrap.py'].decode().replace('residency_results','te_causal_results')
    start=bootstrap.index("        command([sys.executable,str(root/'benchmarks/test_residency_policy.py')]")
    end=bootstrap.index('    except BaseException as error:',start)
    bootstrap=bootstrap[:start]+'''        manifest.update(status='te_block_causality',diagnostic_only=True,optimization_changes=False,
            scope='Two actual full P0 requests; selected warm TE Kineto and whole-block OS counters');save()
        directory=out/'pipeline';directory.mkdir(exist_ok=True)
        command([sys.executable,'-u',str(root/'benchmarks/te_causal_pipeline.py'),
            '--case','offline_p1_p2','--capture','kineto','--output',str(directory),
            '--checkpoints',str(root/'checkpoint_paths.json'),'--reference',str(out/'native/native_reference'),
            '--rank-checkpoint-dir','/tmp/rank_local_tp2','--requests','2','--warmup','0','--steps','4'],2400,'te_causal.log')
        command([sys.executable,str(root/'analyze_te_causal_profile.py'),str(directory)],300,'te_analysis.log')
        # Preserve selected full traces once, compressed with an integrity receipt.
        import gzip
        receipts=[]
        for path in sorted(directory.glob('batch_r*_*.json')):
            sha=hashlib.sha256(path.read_bytes()).hexdigest();size=path.stat().st_size
            with path.open('rb') as source,gzip.open(str(path)+'.gz','wb',compresslevel=3) as target:shutil.copyfileobj(source,target)
            with gzip.open(str(path)+'.gz','rb') as source:
                if hashlib.file_digest(source,'sha256').hexdigest()!=sha:raise RuntimeError('Trace archival mismatch')
            receipts.append(dict(file=path.name,bytes=size,sha256=sha));path.unlink()
        (directory/'trace_archive.json').write_text(json.dumps(receipts,indent=2))
        manifest.update(status='complete',seconds=time.monotonic()-start);save()
'''+bootstrap[end:]
    bootstrap=bootstrap.replace('deadline=start+19800','deadline=start+7200').replace('budget_seconds=19800','budget_seconds=7200')
    files['benchmarks/te_causal_kaggle_bootstrap.py']=bootstrap.encode()
    files['semantic_results/te_block_observer_qualification.json']=json.dumps(proof).encode()
    write_job(ROOT,'kaggle_te_causal_profile',notebook,files,'minimax-h3-te-block-critical-path',
        'MiniMax H3 TE Block Critical Path','benchmarks/te_causal_kaggle_bootstrap.py')


if __name__=='__main__':main()
