"""Qualify opt-in loading policies on the existing logical-rank Native path."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import qualify_causal_audit as original
from benchmarks.historical_te_loading import install


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--mode',choices=('baseline','coalesced','coalesced_parallel','coalesced_parallel_priority'),required=True)
    parser.add_argument('--profile',action='store_true')
    parser.add_argument('--io',action='store_true',help='Qualify unchanged requests with opt-in file-read observers')
    args=parser.parse_args();root=Path(__file__).resolve().parents[1]
    install(args.mode)
    source=Path(original.__file__).read_text()
    if args.io and args.mode!='baseline':parser.error('Read diagnosis must use baseline inference')
    profile=args.profile or args.io
    name=args.mode+('_io' if args.io else '_profile' if args.profile else '')
    source=source.replace("output=root/'semantic_results/causal_qualification'",
        "output=root/'semantic_results/te_loading_qualification'/"+repr(name))
    source=source.replace('output.mkdir(exist_ok=True)','output.mkdir(parents=True,exist_ok=True)')
    source=source.replace("local_reference=output/'native/native_reference'",
        "local_reference=root/'semantic_results/causal_qualification/native/native_reference'")
    source=source.replace("cross_request_te=strategy=='adaptive',cross_model_cpu=strategy=='adaptive')",
        "cross_request_te=strategy=='adaptive',cross_model_cpu="+str(args.mode!='coalesced_parallel_priority')+
        ",rank_checkpoint_dir='D:/minimaxH3/rank_local_tp2')")
    # Numerical comparisons use the preserved Native reference; this gate
    # emits small verdicts, not another set of generated output tensors.
    source=source.replace('for name,tensor in values.items():torch.save(tensor,baseline_dir/f\'{index}_{name}.pt\')','pass')
    if profile:
        import te_pipeline_ablation as trial
        import cpu_causal_trace as cpu
        original.diagnostic.tracing.Recorder=trial.TERecorder
        original.diagnostic.tracing.install=cpu.install
        if args.io:
            import te_io_pipeline as io_trial
            original.diagnostic.tracing.Recorder=io_trial.IORecorder
            original.diagnostic.tracing.install=io_trial.install
        source=source.replace("baseline_dir=output/('baseline_'+strategy+'_'+mode)",
            "baseline_dir=root/'semantic_results/cpu_qualification'/('baseline_'+strategy+'_'+mode)")
    namespace=dict(__name__='te_loading_qualification',__file__=original.__file__)
    exec(compile(source,original.__file__,'exec'),namespace)
    capture='traced' if profile else 'baseline'
    sys.argv=[__file__,'--strategy','adaptive','--capture',capture,'--vae-mode','auto','--requests','1']
    namespace['main']()
    folder=root/'semantic_results/te_loading_qualification'/name
    result=json.loads((folder/f'adaptive_auto_{capture}_qualification.json').read_text())
    result.update(mode=args.mode,physical=False,scope='Compact trained Native full request; scheduling/loader equivalence, not physical performance',
        candidate_sha256=hashlib.sha256((root/'benchmarks/historical_te_loading.py').read_bytes()).hexdigest(),
        observer_sha256=hashlib.sha256((root/'benchmarks/te_pipeline_ablation.py').read_bytes()).hexdigest(),
        io_source_sha256={name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in
            ('benchmarks/te_io_pipeline.py','benchmarks/te_io_telemetry.py')} if args.io else {})
    (folder/'result.json').write_text(json.dumps(result,indent=2))


if __name__=='__main__':main()
