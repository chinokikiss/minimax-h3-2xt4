"""One final CLI artifact collection after a bounded run; never status polling.

A durable local helper can wait without holding the chat open. An exclusive
claim file prevents a later manual invocation and the timer from both making
requests. A cancel marker stops the timer without any Kaggle interaction.
"""
import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time


def read(path):return json.loads(path.read_text(encoding='utf-8'))


def write(path,value):path.write_text(json.dumps(value,indent=2),encoding='utf-8')


def status_path(submission):return submission.with_name(submission.stem+'_collection_status.json')


def collect(submission,output,*,run=subprocess.run):
    submission=submission.resolve();output=output.resolve();root=submission.parent
    if not output.is_relative_to(root):raise ValueError('Collection output must stay inside its evidence directory')
    state=read(submission)
    if state['returncode']!=0:raise ValueError('Submission was not accepted')
    if state.get('final_collection_count',0):return dict(status='already_claimed',kaggle_requests=0)
    claim=submission.with_suffix('.collection-claimed')
    try:
        with claim.open('x') as stream:stream.write(dt.datetime.now(dt.timezone.utc).isoformat())
    except FileExistsError:return dict(status='already_claimed',kaggle_requests=0)
    status=dict(status='collecting',kaggle_requests=1,kernel=state['kernel'],output=str(output))
    write(status_path(submission),status)
    state['final_collection_count']=1;write(submission,state)
    if output.exists():raise FileExistsError('Preserve earlier collected artifacts')
    output.mkdir()
    result=run([sys.executable,'-m','kaggle','kernels','output',state['kernel'],'-p',str(output)],
               capture_output=True,text=True,timeout=300)
    submission.with_name(submission.stem+'_collection_cli.log').write_text(result.stdout+result.stderr,encoding='utf-8')
    manifest={str(p.relative_to(output)):dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
              for p in output.rglob('*') if p.is_file()}
    write(output/'download_manifest.json',manifest)
    status.update(returncode=result.returncode,files=len(manifest),
                  completed_utc=dt.datetime.now(dt.timezone.utc).isoformat())
    if result.returncode or not manifest:status['status']='final_outputs_unavailable'
    else:
        if state.get('experiment_kind') in ('dataset_read','dataset_upload'):
            kind=state['experiment_kind'];binding=output/kind/'source_binding.json'
            expected=state['source_binding']
            if not binding.exists() or read(binding)!=expected:
                status['status']='source_binding_mismatch'
            else:
                status['status']='collected'
                result=read(output/state['result_relative'])
                status['experiment']=dict(status=result['status'],scope=expected['scope'],
                    completed_trials=len(result.get('trials',[])),upload_methods=result.get('methods',[]),
                    acceptance='CPU study only; inspect cache/authentication/privacy limitations')
            write(status_path(submission),status)
            return status
        binding=output/'light_setup/delivery_binding.json'
        expected=state['source_binding']['publication']['capsule_sha256']
        if not binding.exists() or read(binding).get('capsule_sha256')!=expected:
            status['status']='source_binding_mismatch'
        else:
            status['status']='collected'
            relative=Path(state.get('result_relative','light_execution/e2e/L1/result.json'))
            if relative.is_absolute() or '..' in relative.parts:raise ValueError('Result must stay inside collected outputs')
            path=output/relative
            if path.exists():
                result=read(path)
                comparison=state['source_binding'].get('seed_comparison')
                if comparison:
                    generation=path.parent/'generation/result.json'
                    if generation.exists():
                        result=read(generation)
                        actual=result.get('inputs',{})
                        prior=comparison['baseline_inputs']
                        status['seed_comparison']=dict(seed=result.get('config',{}).get('seed'),
                            expected_seed=comparison['seed'],baseline_seed=comparison['baseline_seed'],
                            conditioning_hash_matches=actual.get('conditioning',{}).get('sha256')==prior['conditioning']['sha256'],
                            noise_hashes_changed={key:actual.get(key,{}).get('sha256')!=value['sha256']
                                for key,value in prior.items() if key.startswith('noise_') and key in actual},
                            acceptance='Seed repeat only; generation quality requires review and does not isolate TE')
                if state.get('experiment_kind')=='review_operators':
                    status['experiment']=dict(status=result['status'],error=result.get('error'),
                        all_outputs_byte_exact=result.get('all_outputs_byte_exact'),
                        cases_by_rank=[len(r.get('cases',[])) for r in result.get('ranks',[])],
                        profiling=result.get('profiling'),
                        acceptance='Operator fixtures only; no generation-quality or full-pipeline acceptance')
                elif state.get('experiment_kind')=='review_operators_cpu':
                    status['experiment']=dict(status=result['status'],error=result.get('error'),
                        review_suite=result.get('review_suite'),
                        acceptance='CPU setup/contracts only; no physical T4 execution')
                elif state.get('experiment_kind')=='dit_single':
                    status['experiment']=dict(status=result['status'],error=result.get('error'),
                        calls_by_rank=[r.get('calls') for r in result.get('ranks',[])],
                        sample_wrapper_seconds=result.get('sample_wrapper_seconds'),
                        quality_acceptance='One DiT call only; no generation-quality experiment')
                else:status['experiment']=dict(status=result['status'],error=result.get('error'),
                    stages=result.get('stages'),completed_sampler_steps=len(result.get('sampler_steps',[])),
                    completed_denoiser_calls=len(result.get('denoiser_outputs',[])),media=result.get('media'),
                    quality_acceptance='Pending review; collection does not establish generation quality')
                prior=state.get('expected_prior_inputs',{})
                status['prior_input_hash_matches']={k:result.get('inputs',{}).get(k,{}).get('sha256')==v['sha256']
                                                   for k,v in prior.items()}
    write(status_path(submission),status)
    return status


def main():
    p=argparse.ArgumentParser();p.add_argument('--submission',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--now',action='store_true')
    args=p.parse_args();root=args.submission.resolve().parent;state=read(args.submission)
    deadline=dt.datetime.fromisoformat(state['submitted_utc']).timestamp()+state['earliest_final_collection_seconds']
    while not args.now:
        if args.submission.with_suffix('.collection-claimed').exists():return
        if (root/'final_collection.cancel').exists():
            write(status_path(args.submission),dict(status='cancelled',kaggle_requests=0));return
        remaining=deadline-time.time()
        if remaining<=0:break
        write(status_path(args.submission),dict(status='waiting_for_final_window',kaggle_requests=0,
            not_before_utc=dt.datetime.fromtimestamp(deadline,dt.timezone.utc).isoformat()))
        time.sleep(min(60,remaining))
    try:result=collect(args.submission,args.output)
    except BaseException as error:
        write(status_path(args.submission),dict(status='collection_error',error=repr(error)))
        raise
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
