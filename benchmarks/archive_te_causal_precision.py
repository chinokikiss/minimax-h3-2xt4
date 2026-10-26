"""Preserve the one collected TE audit and its exact submitted sources."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

from benchmarks.analyze_te_causal_precision import summarize


def archive(evidence,output):
    evidence=Path(evidence);output=Path(output);output.mkdir(parents=True,exist_ok=False)
    submission=json.loads((evidence/'gpu_submission.json').read_text())
    assert submission['final_collection_count']==1
    assert hashlib.sha256((evidence/'gpu_job/experiment.ipynb').read_bytes()).hexdigest()==submission['notebook_sha256']
    raw=evidence/'gpu_outputs/000_te_causal_precision.json';batch=json.loads(raw.read_text())
    assert batch['status']=='completed' and len(batch['ranks'])==2
    assert batch['supplemental_sources']==submission['source_binding']['supplemental_sources']
    assert batch['delivery_binding']['capsule_sha256']==submission['source_binding']['publication']['capsule_sha256']
    files={'raw_batch.json':raw,'cpu_validation.json':evidence/'cpu_validation.json','cpu_tests.log':evidence/'cpu_tests.log',
        'local_SM120_smoke.json':evidence/'local_smoke.json','gpu_submission.json':evidence/'gpu_submission.json',
        'gpu_collection.json':evidence/'gpu_collection.json','source_binding.json':evidence/'gpu_job/source_binding.json'}
    repo=Path(__file__).resolve().parents[1]
    paths={'te_precision.py':repo/'experiments/kitchen_attention/te_precision.py',
           'kitchen_te_attention.py':repo/'patches/kitchen_te_attention.py'}
    for name,path in paths.items():
        assert hashlib.sha256(path.read_bytes()).hexdigest()==batch['supplemental_sources'][name]
        files['source_'+name+'.txt']=path
    for name,path in files.items():shutil.copyfile(path,output/name)
    result=summarize(batch);result['input_sha256']=hashlib.sha256(raw.read_bytes()).hexdigest()
    (output/'summary.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
    manifest={p.name:dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sorted(output.iterdir())}
    (output/'manifest.json').write_text(json.dumps(dict(scope='Synthetic T4 TE operator audit, no model/captured activations or credentials',
        files=manifest),indent=2)+'\n')
    print(json.dumps(dict(files=len(manifest),bytes=sum(x['bytes'] for x in manifest.values()),output=str(output))))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--evidence',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();archive(args.evidence,args.output)
