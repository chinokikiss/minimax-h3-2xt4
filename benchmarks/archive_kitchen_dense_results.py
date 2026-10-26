"""Preserve final local receipts and raw samples; never contacts Kaggle."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

from benchmarks.analyze_native_attention_matrix import summarize


def archive(root,output,*,graph=None):
    root=Path(root);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    sources={
        'dense_raw.json':root/'gpu_outputs/attention_matrix/batch_results.json',
        'dense_build_receipt.json':root/'gpu_outputs/attention_matrix/native_dense_build/build_receipt.json',
        'dense_compile.log':root/'gpu_outputs/attention_matrix/native_dense_build/compile.log',
        'cpu_setup_receipt.json':root/'cpu_outputs/light_setup/setup_receipt.json',
        'cpu_operator_qualification.json':root/'cpu_outputs/native_cpu/result.json',
        'dense_source_publication.json':root/'publication.json',
        'dense_cpu_submission.json':root/'cpu_submission.json',
        'dense_gpu_submission.json':root/'gpu_submission.json',
        'dense_cpu_collection.json':root/'cpu_submission_collection_status.json',
        'dense_gpu_collection.json':root/'gpu_submission_collection_status.json',
    }
    # Preserve exactly the build associated with the final operator output.
    for key,suffix in [('dense_build_receipt.json','build_receipt.json'),('dense_compile.log','compile.log')]:
        matches=list((root/'gpu_outputs/attention_matrix').rglob(suffix))
        assert len(matches)==1,(suffix,matches)
        sources[key]=matches[0]
    for name in ('local_workflow_update.json','local_workflow_before_te_default.json',
                 'final_local_validation.json','final_local_operator_controls.json','final_local_preserved.log'):
        if (root/name).is_file():sources[name]=root/name
    if graph:
        graph=Path(graph)
        sources.update({
            'graph_raw.json':graph/'gpu_outputs/graph_matrix/batch_results.json',
            'graph_source_hashes.json':graph/'recovered_submission_sources/source_hashes.json',
            'graph_submission.json':graph/'gpu_submission.json',
            'graph_collection.json':graph/'gpu_submission_collection_status.json',
            'graph_setup_receipt.json':graph/'gpu_outputs/light_setup/setup_receipt.json',
            'graph_evidence_verification.json':graph/'final_evidence_verification.json',
            'graph_execution_supervision.json':graph/'gpu_outputs/graph_supervision/supervision.json',
            'graph_download_manifest.json':graph/'gpu_outputs/download_manifest.json',
        })
        for name in ('graph.py','graph_bench.py','kitchen_te_attention.py'):
            sources['graph_source_'+name+'.txt']=graph/'recovered_submission_sources'/name
    for name,path in sources.items():
        assert path.is_file(),path
        shutil.copyfile(path,output/name)
    for prefix in ('dense', 'graph') if graph else ('dense',):
        batch=json.loads((output/(prefix+'_raw.json')).read_text())
        assert batch['status']=='completed'
        (output/(prefix+'_summary.json')).write_text(json.dumps(summarize(batch),indent=2)+'\n')
    index={p.name:dict(sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bytes=p.stat().st_size)
           for p in sorted(output.iterdir()) if p.is_file() and p.name!='manifest.json'}
    (output/'manifest.json').write_text(json.dumps(dict(scope='Final synthetic T4 operator evidence; no activations or credentials',
        files=index),indent=2)+'\n')
    print(json.dumps(dict(output=str(output),files=len(index),bytes=sum(p['bytes'] for p in index.values()))))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--dense',type=Path,required=True)
    parser.add_argument('--graph',type=Path);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();archive(args.dense,args.output,graph=args.graph)
