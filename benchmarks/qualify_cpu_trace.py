"""Reuse the qualified compact Native path; isolate all new proof artifacts."""
import hashlib
import json
from pathlib import Path
import shutil
import sys

import causal_pipeline_audit as pipeline
import cpu_causal_trace as cpu
import qualify_causal_audit as previous


def main():
    root=Path(__file__).resolve().parents[1]
    old=root/'semantic_results/causal_qualification'
    out=root/'semantic_results/cpu_qualification';out.mkdir(exist_ok=True)
    for name in ('native/native_reference','baseline_adaptive_auto'):
        shutil.copytree(old/name,out/name,dirs_exist_ok=True)
    pipeline.tracing.Recorder=cpu.CPURecorder;pipeline.tracing.install=cpu.install
    source=Path(previous.__file__).read_text(encoding='utf-8').replace(
        "output=root/'semantic_results/causal_qualification'", "output=root/'semantic_results/cpu_qualification'")
    namespace=dict(__name__='qualification_reused',__file__=previous.__file__)
    exec(compile(source,previous.__file__,'exec'),namespace)
    sys.argv=[__file__,'--strategy','adaptive','--capture','traced','--vae-mode','auto','--requests','1']
    try:namespace['main']()
    except BaseException:
        import traceback
        (out/'failure.txt').write_text(traceback.format_exc(),encoding='utf-8')
        raise
    result=json.loads((out/'adaptive_auto_traced_qualification.json').read_text())
    result['scope']='New CPU observers: logical-rank compact full trained Native components, four actual NFEs; equivalence only'
    paths=[root/'benchmarks'/n for n in ('cpu_causal_trace.py','cpu_pipeline_audit.py','cpu_os_sampler.py','qualify_cpu_trace.py')]
    result['cpu_source_sha256']={p.relative_to(root).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    (out/'result.json').write_text(json.dumps(result,indent=2))


if __name__=='__main__':main()
