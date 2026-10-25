"""One NFE after a passed worker gate; attribute only a complete CUDA trace."""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from worker_trace_coverage import check
from worker_cuda_checkpoint import PROTOCOL


def find_tool(name):
    found=shutil.which(name)
    if found:return found
    root=Path('/opt/nvidia')
    return next((str(p) for p in sorted(root.rglob(name)) if p.is_file() and os.access(p,os.X_OK)),None) if root.exists() else None


def tools():
    result={}
    for name in ('nsys','ncu'):
        path=find_tool(name);result[name]=dict(path=path,available=bool(path))
        if path:
            probe=subprocess.run([path,'--version'],capture_output=True,text=True,timeout=15)
            result[name].update(version=(probe.stdout+probe.stderr).strip(),version_exit_code=probe.returncode)
    return result


def claim_profile(output,config,inputs):
    """A backend case can be executed at most once, even after interruption."""
    value=dict(config_sha256=hashlib.sha256(config.read_bytes()).hexdigest())
    with inputs.open('rb') as stream:value['inputs_sha256']=hashlib.file_digest(stream,'sha256').hexdigest()
    with (output/'profile.claimed').open('x',encoding='utf-8') as stream:json.dump(value,stream)


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path);p.add_argument('--inputs',type=Path)
    p.add_argument('--capture-gate',type=Path)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--tools-only',action='store_true')
    args=p.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=True)
    if not args.tools_only:
        if not args.config or not args.inputs or not args.capture_gate:p.error('Bound config, inputs and passed worker gate required')
        claim_profile(out,args.config,args.inputs)
    state=dict(status='starting',tools=tools(),scope='One DiT NFE after worker capture qualification',runs=[])
    def save():(out/'profilers.json').write_text(json.dumps(state,indent=2))
    def run(argv,name,timeout,env=None):
        start=time.monotonic()
        with (out/(name+'.log')).open('w') as log:
            proc=subprocess.run(argv,stdout=log,stderr=subprocess.STDOUT,timeout=timeout,env=env)
        row=dict(name=name,exit_code=proc.returncode,seconds=time.monotonic()-start)
        state['runs'].append(row);save();return row
    save()
    if args.tools_only:return
    gate=json.loads(args.capture_gate.read_text())
    nsys=state['tools']['nsys']['path']
    if (not gate.get('passed') or gate.get('protocol')!=PROTOCOL or not gate.get('checkpoint_restart_verified')
            or gate.get('helper_sha256')!=hashlib.sha256(Path(__file__).with_name('worker_cuda_checkpoint.py').read_bytes()).hexdigest()
            or not nsys):
        state.update(status='blocked_before_dit',blocker='Minimal flushed/graceful worker capture not qualified');save()
        raise RuntimeError(state['blocker'])
    state['capture_gate_sha256']=hashlib.sha256(args.capture_gate.read_bytes()).hexdigest()
    state['counter_permission']=dict(actual_dit_counters=False,blocker='Prior actual T4 ERR_NVGPUCTRPERM; counter probe not repeated')
    env=dict(os.environ,H3_TRACE_CONTROL='1',H3_WORKER_CPU_OUTPUT=str(out/'worker_cpu'))
    row=run([nsys,'profile','--trace=cuda,nvtx,osrt','--sample=none','--cpuctxsw=none',
        '--trace-fork-before-exec=true','--flush-on-cudaprofilerstop=true','--wait=all','-o',str(out/'dit_single'),
        sys.executable,'-u',str(Path(__file__).with_name('profile_dit_single.py')),
        '--config',str(args.config),'--inputs',str(args.inputs),'--output',str(out)],'one_dit_call',1850,env)
    if row['exit_code']!=0:state['execution_error']='Single DiT call failed; export partial coverage only'
    export=run([nsys,'export','--type=sqlite','--output',str(out/'dit_single.sqlite'),
        str(out/'dit_single.nsys-rep')],'nsys_export',90)
    if export['exit_code']!=0:raise RuntimeError('CUDA trace export failed; no attribution')
    sql=out/'dit_single.sqlite'
    state['coverage']=check(sql,prefix='denoise/nfe',minimum=50,require_blocks=True);save()
    # A report or zero-exit profiler is insufficient. Every trained block
    # on both worker GPUs needs correlated kernels before attribution.
    result=json.loads((out/'result.json').read_text()) if (out/'result.json').exists() else dict(status='missing')
    state['worker_exit']=result.get('worker_exit',[])
    flushes=[out/'worker_cpu'/f'rank{rank}_flush.json' for rank in (0,1)]
    clean=(len(state['worker_exit'])==2 and all(r['exit_code']==0 and not r['alive'] for r in state['worker_exit'])
        and all(p.exists() and json.loads(p.read_text()).get('flushed') for p in flushes))
    checkpoints=[json.loads((out/'worker_cpu'/f'rank{rank}_cuda_checkpoints.json').read_text())
        if (out/'worker_cpu'/f'rank{rank}_cuda_checkpoints.json').exists() else {} for rank in (0,1)]
    expected={f'dit_block_{i}' for i in range(4,50,5)}|{'denoise_exit','worker_close'}
    clean=clean and all(r.get('protocol')==PROTOCOL and expected<={v['label'] for v in r.get('records',[])}
        and all(v['status']=='complete' for v in r.get('records',[])) for r in checkpoints)
    state['cuda_checkpoints']=checkpoints
    complete=result['status']=='complete' and state['coverage']['passed'] and clean and row['exit_code']==0
    if complete:
        run([nsys,'stats','--report','cuda_gpu_kern_sum,cuda_api_sum,cuda_gpu_mem_time_sum,nvtx_sum',
            '--format','csv','--output',str(out/'dit_stats'),str(sql)],'nsys_stats',90)
    else:state['attribution_blocker']='Worker CUDA/layer coverage incomplete; summary CSV attribution withheld'
    with sql.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
    with sql.open('rb') as src,gzip.open(out/'dit_single.sqlite.gz','wb',compresslevel=3) as dst:shutil.copyfileobj(src,dst,8*2**20)
    state['sqlite']=dict(uncompressed_bytes=sql.stat().st_size,uncompressed_sha256=digest);sql.unlink()
    state.update(status='complete' if complete else 'trace_incomplete',actual_dit_calls=1,clean_worker_flush=clean)
    save()
    if state['status']!='complete':raise RuntimeError('Do not attribute an incomplete worker trace')


if __name__=='__main__':main()
