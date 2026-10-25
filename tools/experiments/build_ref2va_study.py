"""Freeze the prescribed token matrix into one bounded physical dual-T4 run."""
import ast
import base64
import hashlib
import io
import json
from pathlib import Path
import tarfile
from tools.experiments.build_study_notebook import ROOT


def main():
    proof=ROOT/'semantic_results/ref2va_fixed_matrix.json'
    value=json.loads(proof.read_text())
    if not value['passed']:raise RuntimeError('Local prescribed-matrix semantic proof incomplete')
    for name,sha in value['source_sha256'].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=sha:raise RuntimeError('Changed after local validation: '+name)
    paths=list((ROOT/'patches').glob('*.py'))
    paths += [ROOT/'benchmarks'/n for n in ['retest.py','te_dit_study.py','fixture_inputs.py','full_models.py','full_study.py','ref2va_workloads.py','ref2va_study.py']]
    paths += [proof,ROOT/'semantic_results/full_checkpoint_identity.json']
    # Archive the executed local checks as well as the merged proof.
    paths += list((ROOT/'semantic_results').glob('ref2va_part*.json'))
    paths += [ROOT/name for name in value['source_sha256'] if name not in {p.relative_to(ROOT).as_posix() for p in paths}]
    stream=io.BytesIO()
    with tarfile.open(fileobj=stream,mode='w:gz') as tar:
        for p in sorted(set(paths)):tar.add(p,arcname=p.relative_to(ROOT).as_posix())
    payload=base64.b64encode(stream.getvalue()).decode()
    # Reuse the verified environment/download/process teardown implementation.
    tree=ast.parse((ROOT/'tools/experiments/build_optimized_study.py').read_text())
    template=next(n.value for n in ast.walk(tree) if isinstance(n,ast.Constant) and isinstance(n.value,str) and n.value.startswith('import base64, io, json, math, os'))
    code=template.replace('PAYLOAD',repr(payload))
    code=code.replace('import base64, io, json, math, os, pathlib, signal, subprocess, sys, tarfile, time','import base64, io, json, math, os, pathlib, signal, subprocess, sys, tarfile, time, threading')
    code=code.replace("def save(): (out/'manifest.json').write_text(json.dumps(manifest,indent=2))", "def save():\n    manifest['files']=[p.relative_to(pathlib.Path('/kaggle/working')).as_posix() for p in out.rglob('*') if p.is_file() and p.suffix!='.pt']\n    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))")
    code=code.replace("proc=subprocess.Popen(args,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)", "proc=subprocess.Popen(args,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,start_new_session=True,text=True,bufsize=1)\n            def pump():\n                for line in proc.stdout:\n                    log.write(line);log.flush()\n                    if line.startswith(('LOADED','FULL','THROUGHPUT','MEASURED','CAPTURED','Traceback','RuntimeError','torch.OutOfMemoryError')):print(line,end='',flush=True)\n            reader=threading.Thread(target=pump,daemon=True);reader.start()")
    code=code.replace("        print(logpath.read_text(),flush=True)","            reader.join(5)\n        print('FINISHED',name,'returncode',rc,'seconds',round(time.monotonic()-begin,2),flush=True)")
    begin=code.index('    selected={}\n');end=code.index('    manifest.update(status=',begin)
    orchestration='''    entry=str(root/'benchmarks/ref2va_study.py')
    inputs=pathlib.Path('/tmp/ref2va_inputs')
    def invocation(kind,scope,mode='native',length=None,batch=1,reuse=False,throughput=False):
        folder=out/(('full_' if scope=='full' else '')+kind)
        args=[sys.executable,'-u',entry,'--kind',kind,'--scope',scope,'--checkpoint',checkpoints[kind],'--output',str(folder),'--inputs',str(inputs),'--mode',mode]
        if length:args+=['--length',str(length)]
        if batch!=1:args+=['--batch',str(batch)]
        if reuse:args+=['--reuse-placement']
        if scope=='full':args+=['--throughput-only' if throughput else '--latency-only']
        return args
    if job('dit/prepare',invocation('dit','prepare'),600):raise RuntimeError('Native Ref2VA preparation failed')
    manifest['token_matrix']={'te':[512,2048,4096],'dit':[8192,16384,32768]}
    manifest['fixture_scope']='Native image/video reference packing at prescribed totals; synthetic conditioning/latents; no workflow token extraction'
    # Establish complete full-model comparisons at the practical smallest case
    # before spending time on the larger capacity/time probes.
    for kind,length in [('te',512),('dit',8192)]:
        full=out/('full_'+kind);full.mkdir(exist_ok=True)
        modes=['native','tp_optimized','pp_overlap']
        for mode in modes:
            if mode!='native' and not (full/f'reference_s{length}_b1.pt').exists():
                manifest['jobs'].append({'name':f'full_{kind}/{mode}_s{length}','skipped':'Native reference unavailable'});save();continue
            if mode=='pp_overlap' and not (full/'balance.json').exists():
                manifest['jobs'].append({'name':f'full_{kind}/{mode}_s{length}','skipped':'Balance profile unavailable'});save();continue
            job(f'full_{kind}/{mode}_s{length}',invocation(kind,'full',mode,length),900 if kind=='dit' else 450)
    # Targeted legacy-vs-optimized ablation, isolated per shape. A capacity or
    # trace failure never prevents measurements of other shapes/strategies.
    for kind,shapes in [('te',[(512,1),(512,2),(2048,1),(4096,1)]),('dit',[(8192,1),(8192,2),(16384,1),(32768,1)])]:
        for length,batch in shapes:
            folder=out/kind
            for mode in ['native','tp_legacy','tp_scalemax']:
                if mode!='native' and not (folder/f'native_s{length}_b{batch}.pt').exists():
                    manifest['jobs'].append({'name':f'{kind}/{mode}_s{length}_b{batch}','skipped':'Native reference unavailable'});save();continue
                job(f'{kind}/{mode}_s{length}_b{batch}',invocation(kind,'block',mode,length,batch),300)
    for kind,lengths in [('te',[2048,4096])]:
        full=out/('full_'+kind)
        for length in lengths:
            for mode in ['native','tp_optimized','pp_overlap']:
                if mode!='native' and not (full/f'reference_s{length}_b1.pt').exists():
                    manifest['jobs'].append({'name':f'full_{kind}/{mode}_s{length}','skipped':'Native capacity/time limit left no equivalent reference'});save();continue
                job(f'full_{kind}/{mode}_s{length}',invocation(kind,'full',mode,length,reuse=True),900 if kind=='dit' else 450)
    for kind,length in [('te',512),('dit',8192)]:
        full=out/('full_'+kind)
        for mode in ['native','tp_optimized','pp_overlap']:
            if not (full/f'reference_s{length}_b1.pt').exists() or not (full/'balance.json').exists():
                manifest['jobs'].append({'name':f'full_{kind}/{mode}_throughput','skipped':'Native reference or balance unavailable'});save();continue
            job(f'full_{kind}/{mode}_throughput',invocation(kind,'full',mode,length,reuse=True,throughput=True),1800 if kind=='dit' else 450)
    # Larger full-model cases are capacity/time probes after the complete
    # primary matrix. Never infer their latency from a block result.
    kind='dit';full=out/'full_dit'
    for length in [16384,32768]:
        for mode in ['native','tp_optimized','pp_overlap']:
            if mode!='native' and not (full/f'reference_s{length}_b1.pt').exists():
                manifest['jobs'].append({'name':f'full_{kind}/{mode}_s{length}','skipped':'Native capacity/time limit left no equivalent reference'});save();continue
            job(f'full_{kind}/{mode}_s{length}',invocation(kind,'full',mode,length,reuse=True),900)
'''
    code=code[:begin]+orchestration+code[end:]
    code=code.replace("'phase':'optimized_blocks_and_full_models'","'phase':'prescribed_ref2va_matrix'")
    code=code.replace("'gpu_budget_seconds':2400","'gpu_budget_seconds':5400")
    code=code.replace('gpu_end=time.monotonic()+2400','gpu_end=time.monotonic()+5400')
    code=code.replace("'INIT,GRAPH,P2P'","'INIT,GRAPH,COLL,P2P'")
    code=code.replace("all(j['returncode']==0 for j in manifest['jobs'])","all(j.get('returncode')==0 for j in manifest['jobs'])")
    compile(code,'prescribed_ref2va_study','exec')
    folder=ROOT/'kaggle_study';folder.mkdir(exist_ok=True)
    notebook={'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}},'cells':[{'id':'run','cell_type':'code','metadata':{},'execution_count':None,'outputs':[],'source':code.splitlines(keepends=True)}]}
    (folder/'study.ipynb').write_text(json.dumps(notebook),encoding='utf-8')
    print('Frozen prescribed-matrix notebook bytes',(folder/'study.ipynb').stat().st_size)


if __name__=='__main__':main()
