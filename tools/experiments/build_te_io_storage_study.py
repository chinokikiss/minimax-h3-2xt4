"""Small source-only physical host I/O job; no full-model downloads or uploads."""
import hashlib
import json
from pathlib import Path
from tools.experiments.freeze_pipeline_job import read_bundle, write_job

ROOT=Path(__file__).resolve().parents[2]


def main():
    notebook,old=read_bundle(ROOT/'kaggle_residency_p01/residency_p01.ipynb')
    keep=('semantic_results/offline_conversion_plan.json',
        'semantic_results/kaggle_hf_publication_v6/publication.json',
        'semantic_results/kaggle_hf_publication_v6/upload_receipts.json')
    files={k:v for k,v in old.items() if k in keep or
        (k.startswith('semantic_results/kaggle_hf_publication_v6/te.block') and k.endswith('.json'))}
    for name in ('benchmarks/te_shard_io.py','benchmarks/te_io_storage_study.py','benchmarks/te_io_telemetry.py',
        'benchmarks/te_io_probe.py','patches/windowed_tp.py','patches/weight_residency.py','patches/diagnostic_metrics.py'):
        files[name]=(ROOT/name).read_bytes()
    proof=json.loads((ROOT/'semantic_results/te_shard_io_qualification.json').read_text())
    if not proof['passed'] or hashlib.sha256(files['benchmarks/te_shard_io.py']).hexdigest()!=proof['source_sha256']['benchmarks/te_shard_io.py']:
        raise RuntimeError('I/O candidate byte gate required')
    files['semantic_results/te_shard_io_qualification.json']=json.dumps(proof).encode()
    files['benchmarks/te_storage_bootstrap.py']=b'''import json,subprocess,sys,time
from pathlib import Path
root=Path(__file__).resolve().parents[1]
out=Path('/kaggle/working/te_storage_results');out.mkdir(exist_ok=True)
manifest=dict(status='setup',physical=True,scope='Large-working-set I/O only; no inference speedup claim')
def save():(out/'manifest.json').write_text(json.dumps(manifest,indent=2))
save()
try:
 subprocess.run([sys.executable,'-m','pip','install','--no-cache-dir','--quiet','huggingface-hub==1.12.0','psutil','requests'],check=True,timeout=300)
 manifest['status']='storage_controls';save()
 subprocess.run([sys.executable,'-u',str(root/'benchmarks/te_io_storage_study.py'),'--download','--output',str(out)],check=True,timeout=5400)
 manifest['status']='complete';save()
except BaseException as error:
 manifest.update(status='failed',error_type=type(error).__name__);save();raise
'''
    write_job(ROOT,'kaggle_te_io_storage',notebook,files,'minimax-h3-te-storage-controls',
        'MiniMax H3 TE Storage Controls','benchmarks/te_storage_bootstrap.py')


if __name__=='__main__':main()
