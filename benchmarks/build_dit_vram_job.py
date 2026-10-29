"""Immutable experiment-only overlay and fresh CPU/GPU jobs; no polling."""
import argparse
import ast
import datetime as dt
import hashlib
import io
import json
from pathlib import Path
import re
import subprocess
import sys
import tarfile

from build_kaggle_light_delivery import payload_from_distribution,write_distribution
from run_dit_vram_batch import FILES
from t4_notebook_watchdog import supervise_worker

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--mode',choices=('delivery','cpu','gpu'),required=True)
    parser.add_argument('--base',type=Path);parser.add_argument('--evidence',type=Path,required=True)
    parser.add_argument('--publication',type=Path);parser.add_argument('--qualification',type=Path)
    parser.add_argument('--candidate-qualification',type=Path);parser.add_argument('--submit',action='store_true')
    args=parser.parse_args();args.evidence.mkdir(parents=True,exist_ok=True)
    if args.mode=='delivery':
        raw=args.base.read_bytes();digest=hashlib.sha256(raw).hexdigest();assert args.base.name==digest+'.tar.gz'
        with tarfile.open(fileobj=io.BytesIO(raw),mode='r:gz') as archive:
            distribution={m.name:archive.extractfile(m).read() for m in archive}
        original=payload_from_distribution(distribution);files=dict(original)
        for name in FILES:
            value=(ROOT/'benchmarks'/name).read_bytes();ast.parse(value)
            files['benchmarks/'+name]=value
        assert all(files[name]==value for name,value in original.items() if name not in {'benchmarks/'+n for n in FILES})
        assert not any(re.search(rb'(?:hf_[A-Za-z0-9]{25,}|gh[pousr]_[A-Za-z0-9]{30,})',value) for value in files.values())
        write_distribution(args.evidence/'delivery',distribution,files,parent=digest)
        (args.evidence/'preservation.json').write_text(json.dumps(dict(parent_sha256=digest,production_runtime_byte_exact=True,
            native_and_core_pins_preserved=True,benchmark_only_sources=list(FILES),no_model_weights=True,
            no_private_inputs=True,no_credential=True),indent=2))
        return
    publication=json.loads(args.publication.read_text())
    job=args.evidence/(args.mode+'_job')
    argv=[sys.executable,str(ROOT/'benchmarks/build_kaggle_light_job.py'),'--mode','cpu' if args.mode=='cpu' else 'e2e',
          '--publication',str(args.publication),'--output',str(job),'--setup-budget','600']
    if args.mode=='cpu':argv+=['--review-suite']
    else:
        setup=json.loads(args.qualification.read_text());gate=json.loads(args.candidate_qualification.read_text())
        assert setup['status']=='cpu_ready' and setup['review_suite']['passed'] and gate['status']=='vram_cpu_ready'
        assert gate['capsule_sha256']==setup['capsule_sha256']==publication['capsule_sha256']
        argv+=['--qualification',str(args.qualification)]
    subprocess.run(argv,check=True)
    notebook_path=job/'experiment.ipynb';notebook=json.loads(notebook_path.read_text());code=''.join(notebook['cells'][0]['source'])
    if args.mode=='gpu':
        marker="assert receipt['status']=='ready_for_device_gate'\n";assert code.count(marker)==1
        code=code[:code.index(marker)+len(marker)]
    execution=f"""import json,os,pathlib,sys
source=pathlib.Path('/kaggle/temp/h3_light_source')
env=dict(os.environ);env.update(json.loads(pathlib.Path('/kaggle/working/light_setup/execution_env.json').read_text()))
argv=[sys.executable,'-u',str(source/'benchmarks/run_dit_vram_batch.py'),'--source',str(source),'--output','/kaggle/working/dit_vram']
if {args.mode=='cpu'!r}:argv+=['--cpu-preflight']
os.execve(sys.executable,argv,env)
"""
    code+=supervise_worker(execution,budget=180 if args.mode=='cpu' else 1800,output='/kaggle/working/dit_vram_supervision')
    ast.parse(code);notebook['cells'][0]['source']=code.splitlines(keepends=True);notebook_path.write_text(json.dumps(notebook))
    metadata=json.loads((job/'kernel-metadata.json').read_text());label='cpu' if args.mode=='cpu' else 't4'
    metadata.update(id=f'pamcks/h3-dit-vram-fusion-{label}-20261029',title=f'H3 DiT VRAM Fusion {label.upper()} 20261029')
    assert metadata['is_private'] and not metadata['dataset_sources'] and not metadata['kernel_sources']
    (job/'kernel-metadata.json').write_text(json.dumps(metadata,indent=2))
    binding=json.loads((job/'source_binding.json').read_text());binding.update(full_pipeline=False,
        new_cpu_gate=True,model_scope='Two trained base block0 TP2 shards only; generated 59134-row inputs',
        candidate_qualification_sha256=hashlib.sha256(args.candidate_qualification.read_bytes()).hexdigest() if args.candidate_qualification else None)
    (job/'source_binding.json').write_text(json.dumps(binding,indent=2))
    if not args.submit:return
    receipt=args.evidence/(args.mode+'_submission.json');timeout=900 if args.mode=='cpu' else 2500
    state=dict(kernel=metadata['id'],status='submitting',submitted_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        source_binding=binding,final_collection_count=0,run_timeout_seconds=timeout,earliest_final_collection_seconds=timeout,
        result_relative='dit_vram/result.json',notebook_sha256=hashlib.sha256(notebook_path.read_bytes()).hexdigest())
    with receipt.open('x') as stream:json.dump(state,stream,indent=2)
    argv=[sys.executable,'-m','kaggle','kernels','push','-p',str(job),'-t',str(timeout)]
    if args.mode=='gpu':argv+=['--accelerator','NvidiaTeslaT4']
    try:
        result=subprocess.run(argv,capture_output=True,text=True,timeout=180);text=result.stdout+result.stderr
        state.update(returncode=result.returncode,status='accepted' if result.returncode==0 else 'acceptance_uncertain')
        version=re.search(r'Kernel version (\d+)',text)
        if version:state['version']=int(version.group(1))
    except subprocess.TimeoutExpired:
        state.update(returncode=-1,status='acceptance_uncertain');text='Acknowledgement timeout; do not duplicate automatically.'
    receipt.write_text(json.dumps(state,indent=2));(args.evidence/(args.mode+'_submission_cli.log')).write_text(text)
    print(text[-1500:],flush=True)


if __name__=='__main__':main()
