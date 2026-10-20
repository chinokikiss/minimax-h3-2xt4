"""Anonymous pinned remote ranges; only source code/plans are uploaded locally."""
import json
from pathlib import Path
import sys
import time
import requests
from checkpoint_ranges import assemble


def download(plan,output):
    url=f'https://huggingface.co/{plan["repo"]}/resolve/{plan["revision"]}/{plan["remote_file"]}'
    session=requests.Session()
    def read_range(a,b):
        for attempt in range(3):
            try:
                with session.get(url,params={'download':'true','offset':str(a)},headers={'Range':f'bytes={a}-{b-1}'},
                                 stream=True,timeout=(15,70)) as response:
                    if response.status_code!=206 or response.headers.get('Content-Range')!=f'bytes {a}-{b-1}/{plan["source_bytes"]}':
                        raise RuntimeError('Pinned source range mismatch')
                    result=bytearray()
                    for part in response.iter_content(2**20):
                        result.extend(part)
                        if len(result)>b-a:raise RuntimeError('Oversized source range')
                    if len(result)!=b-a:raise RuntimeError('Incomplete source range')
                    return result
            except (requests.RequestException,RuntimeError):
                if attempt==2:raise RuntimeError(f'Fixture range failed at {a}; no credentials required') from None
                time.sleep(1+attempt)
    assemble(plan,output,read_range)
    print('VERIFIED_PINNED_FIXTURE',Path(output).name,plan['bytes'],plan['sha256'],flush=True)


def main():
    plans=json.loads(Path(sys.argv[1]).read_text());out=Path(sys.argv[2]);out.mkdir(parents=True,exist_ok=True)
    for name,plan in plans.items():download(plan,out/(name+'.safetensors'))


if __name__=='__main__':main()
