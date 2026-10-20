"""Bounded independent P2 jobs; immutable historical sources are only read."""
import argparse
import ast
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent
OUT=Path('D:/attention_eval/tp2_p2_20261018')
BASE=Path('D:/attention_eval/operator_optimization_tp2_20261018/tradeoffs')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('kind',choices=['transport','attention']);parser.add_argument('--restore-only',action='store_true');parser.add_argument('--confirm-transport',action='store_true');parser.add_argument('--scalar-profile-only',action='store_true');args=parser.parse_args()
    manifest=json.loads((OUT/'tp2_operator_manifest.json').read_text())
    previous=json.loads((BASE/'tp2_operator_manifest.json').read_text())
    assert manifest['GEMMs']==previous['GEMMs'] and manifest['attention']==previous['attention'],'Current TP2 targets changed'
    # Other tasks can edit this shared checkout. Historical dependencies are
    # taken from the completed T4 run's immutable submitted snapshots below.
    current_audit=manifest
    manifest=previous
    tree=ast.parse((ROOT/'build_tp2_tradeoffs_job.py').read_text(encoding='utf8'))
    names=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='names' for t in n.targets))
    names+=['patches/tilelang_transport_p2.py','benchmarks/test_tp2_p2_epilogue.py','benchmarks/tp2_p2_transport.py',
            'benchmarks/materialize_tp2_p2_seed.py','benchmarks/fetch_h3_physical_fixtures.py','benchmarks/checkpoint_ranges.py'] if args.kind=='transport' else ['patches/tilelang_attention_p2.py','benchmarks/tp2_p2_attention.py']
    if args.confirm_transport:names+=['benchmarks/tp2_p2_transport_confirm.py']
    frozen=BASE/'attention_repair'/'submitted_source'
    sources={n:((frozen/n) if (frozen/n).exists() else ROOT/n).read_text(encoding='utf8') for n in names}
    for name,source in sources.items():ast.parse(source,filename=name)
    hashes={n:hashlib.sha256(s.encode()).hexdigest() for n,s in sources.items()}
    constants=[ast.literal_eval(n.value) for n in tree.body if isinstance(n,(ast.Assign,ast.AugAssign)) and (isinstance(n,ast.AugAssign) and isinstance(n.target,ast.Name) and n.target.id=='code' or isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='code' for t in n.targets)) and isinstance(n.value,ast.Constant) and isinstance(n.value.value,str)]
    budget=480 if args.scalar_profile_only else 540 if args.confirm_transport else 480 if args.restore_only else 1080 if args.kind=='transport' else 840
    code=constants[0].replace('780',str(budget)).replace("/kaggle/working/tradeoffs","/kaggle/working/p2")
    code+="state['scope']="+repr('Real H3 TP2 transport and exact fused epilogues' if args.kind=='transport' else 'Group32 QK fixed-route attention controlled kernel ablations and scalar profiling')+'\n'
    for name,source in sources.items():code+=f'p=SRC/{name!r};p.parent.mkdir(parents=True,exist_ok=True);p.write_text({source!r})\n'
    code+=f"(OUT/'source_hashes.json').write_text(json.dumps({hashes!r},indent=2))\n(OUT/'tp2_operator_manifest.json').write_text(json.dumps({manifest!r},indent=2))\n"
    runtime=constants[-1].split('    try:\n        comm_ok=')[0]
    runtime=runtime.replace("data=list(pathlib.Path('/kaggle/input').rglob('capture_manifest.json'));assert len(data)==1", "data=list(pathlib.Path('/kaggle/input').rglob("+repr('real_transport_manifest.json' if args.kind=='transport' else 'capture_manifest.json')+"));assert len(data)==1")
    runtime=runtime.replace("benchmarks/tp2_tradeoffs.py",'benchmarks/tp2_p2_'+args.kind+'.py')
    if args.confirm_transport:runtime=runtime.replace('benchmarks/tp2_p2_transport.py','benchmarks/tp2_p2_transport_confirm.py')
    if args.restore_only:
        runtime=runtime.replace("    monitor=subprocess.Popen", "    base+=['--restore-only']\n    monitor=subprocess.Popen")
    if args.scalar_profile_only:
        runtime=runtime.replace("    monitor=subprocess.Popen", "    base+=['--scalar-profile-only']\n    temporary=OUT/'nsys_temp';temporary.mkdir(exist_ok=True);os.environ['TMPDIR']=str(temporary)\n    monitor=subprocess.Popen")
    if args.kind=='transport':
        runtime=runtime.replace("data=list(pathlib.Path('/kaggle/input').rglob('real_transport_manifest.json'));assert len(data)==1",'''seed=list(pathlib.Path('/kaggle/input').rglob('seed_manifest.json'));assert len(seed)==1
    captures=list(pathlib.Path('/kaggle/input').rglob('dit_layer25_call09.pt'));assert len(captures)==1
    assert run([sys.executable,str(SRC/'benchmarks/materialize_tp2_p2_seed.py'),'--seed',str(seed[0].parent),'--captures',str(captures[0].parent),'--output','/kaggle/working/fixtures'],'materialize_real_fixtures',120)
    data=[pathlib.Path('/kaggle/working/fixtures/real_transport_manifest.json')]''')
    code+=runtime
    code+='''    try:
        state['screen_ok']=run([*base,'--output',str(OUT/'screen')],'screen',SCREEN_CAP)
        state['repeat_ok']=state['screen_ok'] and run([*base,'--round','1','--output',str(OUT/'repeat')],'repeat',REPEAT_CAP)
        nsys=shutil.which('nsys') or next((str(p) for p in pathlib.Path('/opt/nvidia').rglob('nsys') if p.is_file()),None)
        state['nsys']=nsys;state['profile_ok']=False;save()
        if nsys and state['screen_ok'] and remaining()>45:
            state['profile_ok']=run([nsys,'profile','--trace=cuda,nvtx,osrt','--sample=none','--cpuctxsw=none',
                '--capture-range=cudaProfilerApi','--capture-range-end=repeat','--force-overwrite=true','-o',str(OUT/'profile'),
                *base,'--profile','--output',str(OUT/'profiled')],'nsys',PROFILE_CAP)
        if not nsys:state['profiling_blocker']='Nsight Systems unavailable'
        state['status']='complete' if all(state.get(k) for k in ('screen_ok','repeat_ok','profile_ok')) else 'partial'
    finally:monitor.terminate();monitor.wait(timeout=10)
except Exception as e:
    state['status']='blocked';state['error']=type(e).__name__+': '+str(e)
    (OUT/'exception.txt').write_text(traceback.format_exc())
finally:save();print('TP2_P2_FINAL',state['status'],flush=True)
'''.replace('SCREEN_CAP','150' if args.confirm_transport else '120' if args.restore_only else '470' if args.kind=='transport' else '350').replace('REPEAT_CAP','80' if args.confirm_transport else '65' if args.restore_only else '280' if args.kind=='transport' else '180').replace('PROFILE_CAP','100' if args.confirm_transport else '65' if args.restore_only else '120')
    ast.parse(code)
    if args.scalar_profile_only:
        code=code.replace("state['screen_ok']=run([*base,'--output',str(OUT/'screen')],'screen',350)","state['screen_ok']=None  # Existing V3 screen is reused; this job only captures selected profiles.")
        code=code.replace("state['repeat_ok']=state['screen_ok'] and run([*base,'--round','1','--output',str(OUT/'repeat')],'repeat',180)","state['repeat_ok']=None")
        code=code.replace("if nsys and state['screen_ok'] and remaining()>45:","if nsys and remaining()>45:").replace("],'nsys',120)","],'nsys',180)")
        code=code.replace("all(state.get(k) for k in ('screen_ok','repeat_ok','profile_ok'))","state.get('profile_ok')")
        ast.parse(code)
    label='scalar-profile' if args.scalar_profile_only else 'transport-confirm' if args.confirm_transport else 'restoration' if args.restore_only else args.kind
    slug='h3-tp2-p2-'+label;folder=ROOT/('kaggle_'+slug);folder.mkdir(exist_ok=True)
    notebook=dict(nbformat=4,nbformat_minor=5,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}},cells=[dict(id='tp2-p2',cell_type='code',metadata={},execution_count=None,outputs=[],source=code.splitlines(keepends=True))])
    (folder/'experiment.ipynb').write_text(json.dumps(notebook),encoding='utf8')
    (folder/'kernel-metadata.json').write_text(json.dumps(dict(id='chinokiki9/'+slug,title='H3 TP2 P2 '+label.replace('-',' ').title(),
        code_file='experiment.ipynb',language='python',kernel_type='notebook',is_private=True,enable_gpu=True,enable_tpu=False,enable_internet=True,
        dataset_sources=['chinokiki9/h3-tp2-p2-transport-seed','chinokiki9/h3-tp2-operator-fixtures'] if args.kind=='transport' else ['chinokiki9/h3-tp2-operator-fixtures'],competition_sources=[],kernel_sources=[],model_sources=[]),indent=2))
    evidence=OUT/label;evidence.mkdir(exist_ok=True)
    (evidence/'submitted_sources.json').write_text(json.dumps(hashes,indent=2));(evidence/'tp2_operator_manifest.json').write_text(json.dumps(manifest,indent=2))
    (evidence/'current_shape_audit.json').write_text(json.dumps(current_audit,indent=2))
    for name,source in sources.items():
        p=evidence/'submitted_source'/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(source,encoding='utf8')
    print(folder)


if __name__=='__main__':main()
