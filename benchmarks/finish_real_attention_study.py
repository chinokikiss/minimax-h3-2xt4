"""Continue locally after the dense comparison, without overlapping GPU jobs."""
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
OUT=Path('D:/attention_eval/tokyo_ref2va_20261013')

if __name__=='__main__':
    deadline=time.monotonic()+1800
    while True:
        state=json.loads((OUT/'runs/dense/run_state.json').read_text())
        if state['status']=='complete':break
        if state['status']=='failed':raise RuntimeError('Dense run failed; see run_state.json')
        if time.monotonic()>deadline:raise TimeoutError('Dense run exceeded collection deadline')
        time.sleep(20)
    jobs=[('capture_real_attention.py',['--mode','sparse']),
          ('recapture_video_chunks.py',[]),
          ('evaluate_real_attention.py',['--only','video']),
          ('profile_real_attention.py',[]),
          ('analyze_real_attention.py',[])]
    for entry,options in jobs:
        print('NEXT_STAGE',entry,*options,flush=True)
        subprocess.run([sys.executable,'-u',str(ROOT/'benchmarks'/entry),*options],check=True,timeout=2400)
