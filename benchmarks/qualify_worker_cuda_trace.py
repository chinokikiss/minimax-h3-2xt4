"""Test capture, flush and process-exit separately before any model download."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from worker_trace_coverage import check
from worker_cuda_checkpoint import PROTOCOL


def run_checkpoint_restart(nsys,output):
    """Both chunks must survive stop/start, including a subsequent forced exit."""
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    result=dict(passed=False,protocol=PROTOCOL,cases=[])
    def command(argv,path,timeout=90):
        with path.open('w') as stream:return subprocess.run(argv,stdout=stream,stderr=subprocess.STDOUT,timeout=timeout).returncode
    for exit_mode in ('terminate','graceful'):
        name='checkpoint_'+exit_mode;root=output/name;root.mkdir()
        report=root/'capture';start=time.monotonic()
        code=command([nsys,'profile','--trace=cuda,nvtx','--sample=none','--cpuctxsw=none',
            '--trace-fork-before-exec=true','--flush-on-cudaprofilerstop=true','--wait=all','-o',str(report),
            sys.executable,str(Path(__file__).with_name('worker_trace_probe.py')),'--output',str(root),
            '--checkpoint-restart','--flush','no','--exit',exit_mode],root/'capture.log')
        row=dict(name=name,exit_code=code,seconds=time.monotonic()-start,exit=exit_mode,passed=False)
        if report.with_suffix('.nsys-rep').exists():
            export=command([nsys,'export','--type=sqlite','--output',str(report.with_suffix('.sqlite')),
                str(report.with_suffix('.nsys-rep'))],root/'export.log',45)
            row['export_exit_code']=export
            if export==0:
                row['coverage']=check(report.with_suffix('.sqlite'),minimum=8)
                row['chunks']=check(report.with_suffix('.sqlite'),prefix='H3_CAPTURE_CHUNK/',minimum=8,minimum_per_label=4)
                receipts=[json.loads((root/f'rank{rank}_cuda_checkpoints.json').read_text()) for rank in (0,1)]
                row['checkpoint_receipts_valid']=all(r['protocol']==PROTOCOL and len(r['records'])==2 and
                    all(v['status']=='complete' and v['stopped'] for v in r['records']) and r['records'][0]['restarted'] and
                    r['records'][1]['restarted']==(exit_mode=='terminate') for r in receipts)
                row['passed']=code==0 and row['coverage']['passed'] and row['chunks']['passed'] and row['checkpoint_receipts_valid']
        result['cases'].append(row)
        (output/'capture_gate.json').write_text(json.dumps(result,indent=2))
    passed=all(r['passed'] for r in result['cases'])
    result.update(passed=passed,selected='checkpoint_graceful' if passed else None,
        checkpoint_restart_verified=passed,
        helper_sha256=hashlib.sha256(Path(__file__).with_name('worker_cuda_checkpoint.py').read_bytes()).hexdigest(),
        blocker=None if passed else 'Explicit stop/start checkpoint coverage failed; do not download models')
    (output/'capture_gate.json').write_text(json.dumps(result,indent=2))
    return result


def run(nsys,output,*,retry=False):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    result=dict(passed=False,cases=[])
    def command(argv,path,timeout=120):
        with path.open('w') as stream:return subprocess.run(argv,stdout=stream,stderr=subprocess.STDOUT,timeout=timeout).returncode
    # The unflushed/terminated case reproduces the previous lifecycle. All
    # alternatives use the exact same four kernels in each rank process.
    cases=[('terminate',False),('graceful',True)] if retry else [('terminate',False),('terminate',True),('graceful',False),('graceful',True)]
    for exit_mode,flush in cases:
        name=exit_mode+('_flush' if flush else '_no_flush');root=output/name;root.mkdir()
        report=root/'capture';start=time.monotonic()
        periodic=['--cuda-flush-interval=1000'] if retry else []
        code=command([nsys,'profile',*periodic,'--trace=cuda,nvtx','--sample=none','--cpuctxsw=none',
            '--trace-fork-before-exec=true','--flush-on-cudaprofilerstop=true','--wait=all','-o',str(report),
            sys.executable,str(Path(__file__).with_name('worker_trace_probe.py')),
            '--output',str(root),'--flush','yes' if flush else 'no','--exit',exit_mode,
            '--dwell-seconds','3' if retry and exit_mode=='terminate' else '0'],root/'capture.log')
        row=dict(name=name,exit_code=code,seconds=time.monotonic()-start,flush=flush,exit=exit_mode)
        if report.with_suffix('.nsys-rep').exists():
            export=command([nsys,'export','--type=sqlite','--output',str(report.with_suffix('.sqlite')),
                str(report.with_suffix('.nsys-rep'))],root/'export.log',45)
            row['export_exit_code']=export
            if export==0:row['coverage']=check(report.with_suffix('.sqlite'))
        result['cases'].append(row)
        (output/'capture_gate.json').write_text(json.dumps(result,indent=2))
    selected=next((r for r in reversed(result['cases']) if r['exit']=='graceful' and r['flush'] and r.get('coverage',{}).get('passed')),None)
    periodic_passed=not retry or all(r.get('coverage',{}).get('passed') for r in result['cases'])
    result.update(passed=selected is not None and periodic_passed,selected=selected['name'] if selected else None,
        periodic_flush_ms=1000 if retry else None,
        blocker=None if selected is not None and periodic_passed else 'Periodic/graceful worker CUDA capture did not pass; do not run DiT attribution')
    (output/'capture_gate.json').write_text(json.dumps(result,indent=2))
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--nsys',required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--retry',action='store_true',help='Only qualify periodic failure-path flush and graceful flush; prior four-case evidence retained')
    p.add_argument('--checkpoint-restart',action='store_true')
    args=p.parse_args()
    if args.retry and args.checkpoint_restart:p.error('Select one capture protocol')
    result=run_checkpoint_restart(args.nsys,args.output) if args.checkpoint_restart else run(args.nsys,args.output,retry=args.retry)
    print(json.dumps(result),flush=True)
    if not result['passed']:raise SystemExit(2)


if __name__=='__main__':main()
