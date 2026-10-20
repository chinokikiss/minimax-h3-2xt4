"""Bounded target-host download and Native-provenance binding; no conversion."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT),str(ROOT/'benchmarks')]


def atomic_json(path,value):
    path=Path(path);temporary=path.with_suffix(path.suffix+'.partial')
    temporary.write_text(json.dumps(value,indent=2));temporary.replace(path)


def main():
    import requests
    p=argparse.ArgumentParser();p.add_argument('--ticket',required=True)
    p.add_argument('--checkpoints',required=True);p.add_argument('--directory',required=True)
    p.add_argument('--evidence',required=True)
    mode=p.add_mutually_exclusive_group();mode.add_argument('--download-only',action='store_true')
    mode.add_argument('--bind-only',action='store_true');a=p.parse_args()
    ticket=json.loads(Path(a.ticket).read_text());bundle=ticket['bundle'];plan=bundle['plan']
    frozen=json.loads((ROOT/'semantic_results/offline_conversion_plan.json').read_text())
    if (len(ticket['objects'])!=200 or not bundle['complete'] or plan!=frozen or
            ticket['bundle_sha256']!=hashlib.sha256(json.dumps(bundle,indent=2).encode()).hexdigest()):
        raise ValueError('Complete matching immutable Native bundle required')
    directory=Path(a.directory).resolve();directory.mkdir(parents=True,exist_ok=True)
    start=time.monotonic();evidence=dict(passed=False,repo=ticket['repo'],revision=ticket['revision'],
        conversion=False,objects=200,download_bytes=0,worker_count=2,native_execution_this_restore=False)
    def download(item):
        name,row=item;stem=name.removesuffix('.safetensors')
        if name!=stem+'.safetensors' or stem not in plan['blocks']:
            raise ValueError('Unexpected offline file')
        target=(directory/name).resolve()
        if not target.is_relative_to(directory):raise ValueError('Download escapes directory')
        if row['expires']<=time.time()+10:raise RuntimeError('Offline read capability expired')
        expected=plan['blocks'][stem];m=row['manifest']
        if (any(m[k]!=expected[k] for k in ('kind','index','rank','state_sha256','quant_metadata')) or
                m['file_sha256']!=row['sha256'] or m['configuration']!=plan['configuration'] or
                m['format']!=plan['format'] or not m['native_tensor_bit_exact']):
            raise ValueError('Offline manifest differs from Native proof')
        sha=hashlib.sha256();count=0;temporary=target.with_suffix('.download')
        # At most two streaming downloads; no full weight buffers in RAM.
        with requests.get(row['url'],stream=True,timeout=(15,90)) as response:
            if response.status_code!=200:raise RuntimeError('Private offline download rejected')
            with temporary.open('wb') as output:
                for chunk in response.iter_content(1024*1024):
                    count+=len(chunk)
                    if count>row['bytes']:raise ValueError('Oversized offline object')
                    sha.update(chunk);output.write(chunk)
        if count!=row['bytes'] or sha.hexdigest()!=row['sha256']:
            raise ValueError('Offline transfer SHA256/size mismatch')
        temporary.replace(target);atomic_json(directory/(stem+'.json'),m)
        return count
    if not a.bind_only:
        with ThreadPoolExecutor(max_workers=2) as pool:
            for index,count in enumerate(pool.map(download,sorted(ticket['objects'].items())),1):
                evidence['download_bytes']+=count
                if index%20==0:print('VERIFIED offline downloaded',index,'/200',flush=True)
        evidence['download_seconds']=time.monotonic()-start
        if a.download_only:
            atomic_json(a.evidence,dict(evidence,passed=True,phase='download_only'))
            return
    else:
        evidence=json.loads(Path(a.evidence).read_text())
        if not (evidence['passed'] and evidence['phase']=='download_only' and
                evidence['repo']==ticket['repo'] and evidence['revision']==ticket['revision']):
            raise ValueError('Matching complete offline download required before binding')
    import offline_checkpoint_store as store
    from patches.windowed_tp import CheckpointReader
    import torch
    paths=json.loads(Path(a.checkpoints).read_text());readers={}
    torch.set_num_threads(4);torch.set_grad_enabled(False)
    for kind,source in plan['sources'].items():
        reader=CheckpointReader(paths[kind])
        if reader.path.stat().st_size!=source['bytes'] or store.digest(reader.path)!=source['sha256']:
            raise ValueError('Original source SHA256 mismatch')
        readers[kind]=reader
    bind_start=time.monotonic()
    for index,(stem,expected) in enumerate(sorted(plan['blocks'].items()),1):
        store.bind_block(directory,stem,expected,readers[expected['kind']])
        if index%20==0:print('VERIFIED offline Native rebind',index,'/200',flush=True)
    atomic_json(directory/'bundle.json',bundle)
    evidence.update(passed=True,phase='bound',bind_seconds=time.monotonic()-bind_start,
                    binding_phase_seconds=time.monotonic()-start)
    atomic_json(a.evidence,evidence)
    print('VERIFIED offline restore complete; no conversion',flush=True)


if __name__=='__main__':
    try:main()
    except Exception as error:
        print('ERROR offline restore',type(error).__name__,flush=True)
        raise SystemExit(1) from None
