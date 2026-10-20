"""Train-weight full compact request equivalence; no performance claims."""
import argparse
import functools
import hashlib
import json
from pathlib import Path
import sys
import torch
import qualify_causal_audit as original
from patches.host_weight_cache import HostBudget


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--policy',choices=('baseline','p0','p0_p1'),required=True)
    parser.add_argument('--block-telemetry',action='store_true')
    parser.add_argument('--idle',action='store_true')
    args=parser.parse_args();root=Path(__file__).resolve().parents[1]
    source=Path(original.__file__).read_text()
    if args.idle and (args.policy=='baseline' or args.block_telemetry):parser.error('Idle qualification requires P0 and separate instrumentation')
    name=args.policy+('_block_profile' if args.block_telemetry else '_idle' if args.idle else '')
    source=source.replace("output=root/'semantic_results/causal_qualification'",
        "output=root/'semantic_results/residency_qualification'/"+repr(name))
    source=source.replace('output.mkdir(exist_ok=True)','output.mkdir(parents=True,exist_ok=True)')
    source=source.replace("local_reference=output/'native/native_reference'",
        "local_reference=root/'semantic_results/causal_qualification/native/native_reference'")
    source=source.replace("cross_request_te=strategy=='adaptive',cross_model_cpu=strategy=='adaptive')",
        "cross_request_te=True,cross_model_cpu=True,rank_checkpoint_dir='D:/minimaxH3/rank_local_tp2',"+
        f"prefetch_lifecycle={args.policy!='baseline'},memory_pressure_aware={args.policy=='p0_p1'},"+
        f"idle_residency={args.idle},idle_weight_bytes=256*2**20)")
    if args.idle:
        source=source.replace('tracing.ACTIVE=recorder','tracing.install(recorder)')
        marker='                retained.append(traced);records.append(record)'
        source=source.replace(marker,marker+'''
                service.call('idle_begin')
                import time
                deadline=time.monotonic()+15
                while time.monotonic()<deadline:
                    with torch.cuda.device(0),torch.inference_mode():
                        for rank in service.ranks:rank.idle_tick()
                    if all(rank.idle_window and rank.idle_window.entries for rank in service.ranks):break
                    time.sleep(.02)
                assert all(rank.idle_window and rank.idle_window.entries for rank in service.ranks)
''')
    source=source.replace("for name,tensor in values.items():torch.save(tensor,baseline_dir/f'{index}_{name}.pt')",
        "assert all(torch.equal(torch.load(root/'semantic_results/cpu_qualification/baseline_adaptive_auto'/f'0_{k}.pt',weights_only=True),v) for k,v in values.items())")
    original.qualify.HostBudget=functools.partial(HostBudget,pressure_aware=args.policy=='p0_p1')
    if args.block_telemetry:
        if args.policy!='p0':parser.error('The causal diagnostic must use unchanged P0')
        sys.argv=[__file__,'--capture','kineto']
        import te_causal_pipeline
        source=source.replace("baseline_dir=output/('baseline_'+strategy+'_'+mode)",
            "baseline_dir=root/'semantic_results/cpu_qualification/baseline_adaptive_auto'")
    namespace=dict(__name__='residency_qualification',__file__=original.__file__)
    exec(compile(source,original.__file__,'exec'),namespace)
    capture='traced' if args.block_telemetry else 'baseline'
    observer_hashes={n:hashlib.sha256((root/n).read_bytes()).hexdigest() for n in
        ('benchmarks/te_block_telemetry.py','benchmarks/te_causal_pipeline.py')} if args.block_telemetry else {}
    sys.argv=[__file__,'--strategy','adaptive','--capture',capture,'--vae-mode','auto','--requests','1']
    namespace['main']()
    folder=root/'semantic_results/residency_qualification'/name
    result=json.loads((folder/f'adaptive_auto_{capture}_qualification.json').read_text())
    result.update(policy=args.policy,all_outputs_bit_exact_with_uninstrumented=True,
        physical=False,scope='One compact trained full Native Ref2VA request, logical ranks; semantics only',
        continuous_limitation='Two-request Windows run reached host admission guard at second TE startup; no memory guard relaxed. Physical consecutive requests remain required.')
    if args.idle:
        result.update(scope='One compact full Native request plus trained rank-local TE GPU idle prefill; repeated trained TE reuse is a separate gate; no performance claim',
            continuous_limitation='Repeated full requests reached Windows host admission guard; no guard relaxed. Physical sustained capacity remains separate.')
    if args.block_telemetry:
        if any(hashlib.sha256((root/n).read_bytes()).hexdigest()!=h for n,h in observer_hashes.items()):
            raise RuntimeError('Observer changed during qualification')
        result['observer_source_sha256']=observer_hashes
    (folder/'result.json').write_text(json.dumps(result,indent=2))
    print('VERIFIED residency policy',args.policy,flush=True)


if __name__=='__main__':main()
