"""Exercise the actual loading wrapper's two-process spawn without weights."""
from dataclasses import asdict
import json
import multiprocessing as mp
from pathlib import Path
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[1]
for folder in (ROOT,ROOT/'benchmarks',ROOT/'ComfyUI',ROOT/'.retest-deps'):
    sys.path.insert(0,str(folder))
import pipeline_residency_audit as audit
import run_loading_audit as launcher


def child(config, queue):
    queue.put(asdict(config))


def main():
    cases=[]
    for strategy,loader,p2 in [('sync','direct','off'),('adaptive','direct','off'),
                               ('adaptive','direct','on'),('adaptive','offline','on')]:
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory)
            def service(config):
                assert audit.AuditConfig is type(config)
                context=mp.get_context('spawn');queue=context.Queue()
                processes=[context.Process(target=child,args=(config,queue)) for _ in range(2)]
                try:
                    for process in processes:process.start()
                    received=[queue.get(timeout=60) for _ in processes]
                    for process in processes:
                        process.join(60)
                        assert process.exitcode==0
                    assert all(item==asdict(config) for item in received)
                    assert config.host_bytes==24*audit.GiB
                    assert config.cross_request_te==(strategy=='adaptive' and p2=='on')
                    assert config.cross_model_cpu==(strategy=='adaptive')
                    assert config.rank_checkpoint_dir==('unused-offline-path' if loader=='offline' else None)
                finally:
                    for process in processes:
                        if process.is_alive():process.terminate();process.join()
                    queue.close()
            def exercise():
                audit.runtime.PhysicalTPService(audit.AuditConfig('unused-te','unused-dit',strategy=strategy))
                (output/(strategy+'_result.json')).write_text(json.dumps(dict(source_sha256={})))
            original_main,original_service=audit.main,audit.runtime.PhysicalTPService
            audit.main=exercise;audit.runtime.PhysicalTPService=service
            sys.argv=['run_loading_audit.py','--strategy',strategy,'--output',str(output),
                      '--loader',loader,'--cross-request-te',p2]
            if loader=='offline':sys.argv+=['--rank-checkpoint-dir','unused-offline-path']
            try:launcher.main()
            finally:audit.main=original_main;audit.runtime.PhysicalTPService=original_service
            cases.append(dict(strategy=strategy,loader=loader,p2=p2,ranks=2,passed=True))
            print('VERIFIED loading spawn',strategy,loader,p2,flush=True)
    (ROOT/'semantic_results/loading_spawn_qualification.json').write_text(json.dumps(
        dict(passed=True,cases=cases,scope='CPU configuration transport only; no GPU performance evidence'),indent=2))


if __name__=='__main__':main()
