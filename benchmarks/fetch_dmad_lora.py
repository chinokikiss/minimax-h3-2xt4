"""Pinned, anonymous download of the user's test adapter; never uploads data."""
import hashlib
import json
from pathlib import Path
import sys
import time
import requests
from huggingface_hub import hf_hub_url,get_hf_file_metadata

REVISION='439136a1dd4f435b8137a084f73d558d2cecda2a'
NAME='minimax_h3/dmad_minimax_h3_4step_full_critic_comfyui.safetensors'


def main():
    root=Path(__file__).resolve().parents[1];directory=Path('D:/minimaxH3/loras');directory.mkdir(exist_ok=True)
    target=directory/Path(NAME).name;partial=target.with_suffix('.partial')
    try:meta=get_hf_file_metadata(hf_hub_url('ZhengmingYu/DMAD',NAME,revision=REVISION),token=False)
    except Exception as error:raise RuntimeError('Adapter metadata failed: '+type(error).__name__) from None
    if not target.exists():
        offset=partial.stat().st_size if partial.exists() else 0
        if offset>meta.size:raise ValueError('Partial adapter exceeds immutable size')
        while offset<meta.size:
            stop=min(meta.size-1,offset+64*2**20-1)
            for attempt in range(3):
                try:
                    with requests.get(meta.location,headers={'Range':f'bytes={offset}-{stop}'},stream=True,timeout=(15,90)) as response:
                        if response.status_code!=206 or response.headers.get('Content-Range')!=f'bytes {offset}-{stop}/{meta.size}':
                            raise RuntimeError('Adapter range mismatch')
                        with partial.open('ab') as stream:
                            for part in response.iter_content(2**20):stream.write(part)
                    if partial.stat().st_size!=stop+1:raise IOError('Incomplete adapter range')
                    break
                except requests.RequestException:
                    offset=partial.stat().st_size if partial.exists() else 0
                    if attempt==2:raise RuntimeError('Adapter range download failed; partial retained') from None
                    time.sleep(2)
            offset=partial.stat().st_size
            if offset%(256*2**20)==0 or offset==meta.size:print('DMAD downloaded',offset,'/',meta.size,flush=True)
        partial.replace(target)
    with target.open('rb') as stream:sha=hashlib.file_digest(stream,'sha256').hexdigest()
    if target.stat().st_size!=meta.size or sha!=meta.etag:raise ValueError('Adapter immutable checksum mismatch')
    out=root/'semantic_results/dmad_lora';out.mkdir(exist_ok=True)
    (out/'source_identity.json').write_text(json.dumps(dict(path=str(target),repo='ZhengmingYu/DMAD',revision=REVISION,
        file=NAME,bytes=meta.size,sha256=sha),indent=2))
    print('VERIFIED DMAD adapter',meta.size,sha,flush=True)


if __name__=='__main__':main()
