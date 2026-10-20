"""Capture one submitted Kaggle run, including errors; never submit or retry it."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import threading
import time

from kaggle.api.kaggle_api_extended import KaggleApi


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--slug',required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--timeout',type=int,default=7600)
    p.add_argument('--prefix',default='pipeline_audit_results/')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    api=KaggleApi();api.authenticate()
    end=time.monotonic()+a.timeout;stop=threading.Event();thread=None;last=None
    def stream():
        with (a.output/'live.log').open('a',encoding='utf-8') as log:
            while not stop.is_set():
                try:
                    for e in api.kernels_logs_stream(a.slug):
                        data=e.get('data','');log.write(data);log.flush()
                        if stop.is_set():return
                except Exception as e:
                    # SDK exception text can contain capability URLs.
                    log.write('LOG_STREAM_RETRY '+type(e).__name__+'\n');log.flush()
                stop.wait(10)
    while time.monotonic()<end:
        try:
            r=api.kernels_status(a.slug);status=str(r.status).split('.')[-1]
        except Exception as e:
            print('STATUS_RETRY',type(e).__name__,flush=True);stop.wait(25);continue
        stamp=datetime.now(timezone.utc).isoformat()
        (a.output/'status.json').write_text(json.dumps(dict(kernel=a.slug,status=status,checked_at=stamp),indent=2))
        if status!=last:print('STATUS',status,stamp,flush=True);last=status
        if status=='RUNNING' and thread is None:
            thread=threading.Thread(target=stream,daemon=True);thread.start()
        if status in ('ERROR','COMPLETE','CANCELED','CANCELLED','CANCEL_ACKNOWLEDGED'):
            stop.set()
            subprocess.run([sys.executable,str(Path(__file__).with_name('pipeline_audit_control.py')),
                '--slug',a.slug,'--output',str(a.output),'--prefix',a.prefix,'--download'],check=True)
            native_path=a.output/'pipeline_audit_results/native_result.json'
            if native_path.exists() and json.loads(native_path.read_text()).get('passed'):
                subprocess.run([sys.executable,str(Path(__file__).with_name('analyze_loading_audit.py')),
                    str(a.output/'pipeline_audit_results')],check=True)
            return
        stop.wait(25)
    stop.set()
    print('WATCH_DEADLINE_REACHED no benchmark cancelled or resubmitted',flush=True)


if __name__=='__main__':main()
