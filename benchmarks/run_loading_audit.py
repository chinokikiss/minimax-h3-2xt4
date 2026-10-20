"""Physical ablations of the locally qualified loading policies."""
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys

import pipeline_residency_audit as audit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--loader', choices=('direct', 'offline'), default='direct')
    parser.add_argument('--rank-checkpoint-dir')
    parser.add_argument('--cross-request-te', choices=('on','off'), default='on',
                        help='Separate next-request CPU lookahead from P1 scheduling')
    options, arguments = parser.parse_known_args()
    strategy = arguments[arguments.index('--strategy')+1]
    if options.loader == 'offline' and not options.rank_checkpoint_dir:
        raise ValueError('Offline loading requires a target-host converted checkpoint')
    # Keep AuditConfig importable under its original module name: spawn
    # pickles this dataclass before either GPU worker can start.
    def configured_service(original):
        def construct(config):
            config = replace(config,
            host_bytes=24*audit.GiB,
            rank_checkpoint_dir=options.rank_checkpoint_dir,
            cross_request_te=strategy == 'adaptive' and options.cross_request_te == 'on',
            cross_model_cpu=strategy == 'adaptive')
            return original(config)
        return construct
    import qualify_comfy_sampler as qualify
    physical, logical = audit.runtime.PhysicalTPService, qualify.LogicalService
    audit.runtime.PhysicalTPService = configured_service(physical)
    qualify.LogicalService = configured_service(logical)
    sys.argv = [sys.argv[0], *arguments]
    try:
        audit.main()
    finally:
        audit.runtime.PhysicalTPService = physical
        qualify.LogicalService = logical
    output = Path(arguments[arguments.index('--output')+1])
    path = output/f'{strategy}_result.json'
    result = json.loads(path.read_text())
    result['loading_policy'] = dict(loader=options.loader, cpu_horizon=16 if strategy == 'adaptive' else 1,
        gpu_horizon=2 if strategy == 'adaptive' else 1,
        cross_request_te=strategy == 'adaptive' and options.cross_request_te == 'on',
        cross_model_cpu=strategy == 'adaptive')
    result['source_sha256']['benchmarks/run_loading_audit.py'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    path.write_text(json.dumps(result, indent=2))


if __name__ == '__main__': main()
