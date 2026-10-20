"""Restore the previous physical Native reference directly on Kaggle."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import gc
import hashlib
import json
from pathlib import Path
from urllib.parse import urlparse


def main():
    import requests
    parser=argparse.ArgumentParser()
    parser.add_argument('--ticket',type=Path,required=True)
    parser.add_argument('--directory',type=Path,required=True)
    parser.add_argument('--evidence',type=Path,required=True)
    args=parser.parse_args();ticket=json.loads(args.ticket.read_text())
    names={n+'.pt' for n in ('conditioning','video','audio','pixels','waveform')}
    if set(ticket['objects'])!=names:raise ValueError('Complete saved Native reference required')
    args.directory.mkdir(parents=True,exist_ok=True)
    def download(item):
        name,row=item
        if urlparse(row['url']).scheme!='https':raise ValueError('HTTPS reference source required')
        target=args.directory/name;temporary=target.with_suffix('.partial')
        digest=hashlib.sha256();size=0
        with requests.get(row['url'],stream=True,timeout=(15,90)) as response:
            if response.status_code!=200:raise RuntimeError('Saved reference output unavailable')
            with temporary.open('wb') as stream:
                for chunk in response.iter_content(1024*1024):
                    size+=len(chunk)
                    if size>row['bytes']:raise ValueError('Oversized reference output')
                    digest.update(chunk);stream.write(chunk)
        if size!=row['bytes']:raise ValueError('Incomplete reference output')
        temporary.replace(target)
        return name,dict(bytes=size,download_sha256=digest.hexdigest())
    with ThreadPoolExecutor(max_workers=2) as pool:objects=dict(pool.map(download,ticket['objects'].items()))
    import torch
    for name,row in ticket['objects'].items():
        tensor=torch.load(args.directory/name,map_location='cpu',weights_only=True)
        if not isinstance(tensor,torch.Tensor) or list(tensor.shape)!=row['shape']:
            raise ValueError('Saved reference tensor shape mismatch')
        if not torch.isfinite(tensor).all():raise ValueError('Saved reference contains nonfinite values')
        del tensor;gc.collect()
    evidence=dict(passed=True,source='Kaggle causal audit version 1 physical Native outputs',
        fresh_native_execution=False,objects=objects,
        integrity='HTTPS output identity, exact transfer length, tensor shape/finite validation; recorded download hashes are not upstream checksums')
    args.evidence.write_text(json.dumps(evidence,indent=2));args.ticket.unlink()
    print('VERIFIED saved physical Native reference restored on Kaggle',flush=True)


if __name__=='__main__':
    try:main()
    except Exception as error:
        print('ERROR reference restore',type(error).__name__,flush=True)
        raise SystemExit(1) from None
