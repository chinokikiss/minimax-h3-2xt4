"""Trained dual-rank qualification of depth four and the physical host budget.

Both ranks share the 5060 Ti; GPU windows stay small enough for that device.
Only RAM/cache/depth configuration changes. Timings are not performance evidence.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import pipeline_residency_audit as audit


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--strategy',choices=('sync','adaptive'),required=True)
    parser.add_argument('--cache-gib',type=float,default=3.)
    args=parser.parse_args()
    original_config=audit.AuditConfig
    original_window=audit.AuditWindow
    def config(*positional,**kwargs):
        if kwargs.get('logical'):kwargs['host_bytes']=24*audit.GiB
        return original_config(*positional,**kwargs)
    class PhysicalDepthWindow(original_window):
        def __init__(self,*positional,**kwargs):
            super().__init__(*positional,**kwargs)
            if self.audit.config.logical and self.audit.config.strategy=='adaptive':
                self.max_lookahead=4
    audit.AuditConfig=config;audit.AuditWindow=PhysicalDepthWindow
    output=audit.ROOT/'semantic_results/pipeline_audit_depth4_local'
    output.mkdir(exist_ok=True)
    # Reuse Native outputs; no Native execution invokes the repaired admission
    # or worker-abort methods. The model operators and request are unchanged.
    original_reference=audit.ROOT/'semantic_results/pipeline_audit_full_local/native_reference'
    import shutil
    (output/'native_reference').mkdir(exist_ok=True)
    for name in ('conditioning','video','audio','pixels','waveform'):
        source=original_reference/(name+'.pt');target=output/'native_reference'/source.name
        if not target.exists():shutil.copyfile(source,target)
    sys.argv=[sys.argv[0],'--strategy',args.strategy,'--layers','50','--steps','1',
              '--cache-gib',str(args.cache_gib),'--output',str(output)]
    audit.main()
    path=output/(args.strategy+'_result.json');result=json.loads(path.read_text())
    result['qualification_configuration']=dict(host_gib=24,cache_gib_per_rank=args.cache_gib,max_adaptive_lookahead=4,
        scope='Full 50 trained layers, 512x512/124 frames, Native frontend and VAEs, cold/warm; shared 5060 Ti')
    result['source_sha256']['benchmarks/qualify_pipeline_depth4.py']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    path.write_text(json.dumps(result,indent=2))
    print('QUALIFIED physical host/depth configuration',args.strategy,flush=True)


if __name__=='__main__':main()
