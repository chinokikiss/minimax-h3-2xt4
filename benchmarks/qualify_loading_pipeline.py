"""Current loader/scheduler full trained local pipeline against saved Native.

The Native forward functions and inputs are unchanged; the loader's exact
trained tensor equality is separately proved by qualify_rank_local_loader.
One NFE cold/warm at 512x512/124 frames; local timings are not T4 performance.
"""
import argparse
from dataclasses import replace
from pathlib import Path
import shutil
import sys
import pipeline_residency_audit as audit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--strategy',choices=('sync','adaptive'),default='adaptive')
    parser.add_argument('--offline',action='store_true')
    args = parser.parse_args()
    original = audit.AuditConfig
    def config(*positional,**kwargs):
        value = original(*positional,**kwargs)
        return replace(value,host_bytes=24*audit.GiB,
            rank_checkpoint_dir='D:/minimaxH3/rank_local_tp2' if args.offline else None,
            cross_request_te=args.strategy=='adaptive',cross_model_cpu=args.strategy=='adaptive')
    audit.AuditConfig = config
    output = audit.ROOT/('semantic_results/loading_pipeline_offline' if args.offline else
                       'semantic_results/loading_pipeline_direct')
    output.mkdir(exist_ok=True); (output/'native_reference').mkdir(exist_ok=True)
    reference = audit.ROOT/'semantic_results/pipeline_audit_full_local/native_reference'
    for name in ('conditioning','video','audio','pixels','waveform'):
        target = output/'native_reference'/(name+'.pt')
        if not target.exists():shutil.copyfile(reference/(name+'.pt'),target)
    sys.argv = [sys.argv[0],'--strategy',args.strategy,'--layers','50','--steps','1',
                '--output',str(output)]
    audit.main()


if __name__ == '__main__':main()
