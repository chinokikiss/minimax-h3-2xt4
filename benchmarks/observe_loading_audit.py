"""CPU progress diagnostics around unchanged Native/TP audit execution."""
import faulthandler
from pathlib import Path
import runpy
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
for folder in (ROOT,ROOT/'ComfyUI',ROOT/'.retest-deps',ROOT/'benchmarks'):
    sys.path.insert(0,str(folder))


def main():
    # Capture Python's blocked call sites when a stage produces no progress.
    # This adds no CUDA fence, tensor operation or attention replacement.
    faulthandler.enable()
    print('AUDIT DIAGNOSTICS Python stack snapshots every 120s; their Timeout headings do not abort execution',flush=True)
    faulthandler.dump_traceback_later(120,repeat=True)
    from native_resident_reference import ResidentNativeDiT
    original=ResidentNativeDiT.forward
    def observed(self,*args,**kwargs):
        self._audit_evaluation=getattr(self,'_audit_evaluation',0)+1
        evaluation=self._audit_evaluation;start=time.monotonic()
        print('AUDIT NATIVE EVAL BEGIN',evaluation,flush=True)
        try:
            return original(self,*args,**kwargs)
        finally:
            print('AUDIT NATIVE EVAL HOST_RETURN',evaluation,
                  'seconds',round(time.monotonic()-start,3),
                  'not_a_synchronized_gpu_latency',flush=True)
    ResidentNativeDiT.forward=observed
    try:
        runpy.run_path(str(ROOT/'benchmarks/run_loading_audit.py'),run_name='__main__')
    finally:
        faulthandler.cancel_dump_traceback_later()
        ResidentNativeDiT.forward=original


if __name__=='__main__':main()
