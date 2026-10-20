"""Run small, immutable TE read probes before expensive full-pipeline setup."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--directory',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    import torch
    if torch.cuda.device_count()!=2 or any('T4' not in torch.cuda.get_device_name(r) for r in (0,1)):
        raise RuntimeError('This probe must run on the actual physical dual-T4 pipeline host')
    args.directory.mkdir(parents=True,exist_ok=True);args.output.mkdir(parents=True,exist_ok=True)
    evidence=dict(gpus=[torch.cuda.get_device_name(r) for r in (0,1)],torch=str(torch.__version__),
        scope='CPU I/O isolation before full-model setup; not GPU performance evidence',objects=[])
    from huggingface_hub import hf_hub_url,get_hf_file_metadata
    import requests
    folder=ROOT/'semantic_results/kaggle_hf_publication_v6'
    publication=json.loads((folder/'publication.json').read_text())
    receipts=json.loads((folder/'upload_receipts.json').read_text())
    plan=json.loads((ROOT/'semantic_results/offline_conversion_plan.json').read_text())
    def download(name):
        expected=receipts['objects'][name];stem=name.removesuffix('.safetensors')
        manifest=json.loads((folder/(stem+'.json')).read_text())
        if not manifest['native_tensor_bit_exact'] or manifest['state_sha256']!=plan['blocks'][stem]['state_sha256']:
            raise ValueError('Native-qualified probe manifest mismatch')
        try:
            metadata=get_hf_file_metadata(hf_hub_url(publication['repo'],name,revision=publication['commit']),token=False)
            if metadata.size!=expected['bytes'] or metadata.etag!=expected['sha256']:raise ValueError('Immutable probe identity mismatch')
            path=args.directory/name;sha=hashlib.sha256();size=0
            with requests.get(metadata.location,stream=True,timeout=(15,90)) as response:
                response.raise_for_status()
                with path.open('wb') as stream:
                    for chunk in response.iter_content(1024*1024):
                        size+=len(chunk)
                        if size>expected['bytes']:raise ValueError('Probe object too large')
                        sha.update(chunk);stream.write(chunk)
            if size!=expected['bytes'] or sha.hexdigest()!=expected['sha256']:raise ValueError('Probe download checksum mismatch')
            path.with_suffix('.json').write_text(json.dumps(manifest))
            print('VERIFIED early TE probe shard',name,size,flush=True)
            return dict(name=name,bytes=size,sha256=sha.hexdigest())
        except Exception as error:
            # HF exceptions may include a signed read capability.
            raise RuntimeError('Early probe download failed: '+name+' '+type(error).__name__) from None
    names=[f'te.block{block:02d}.rank{rank}.safetensors' for rank in (0,1) for block in (10,25)]
    with ThreadPoolExecutor(max_workers=2) as pool:evidence['objects']=list(pool.map(download,names))
    (args.output/'early_source_identity.json').write_text(json.dumps(evidence,indent=2))
    subprocess.run([sys.executable,str(ROOT/'benchmarks/test_te_io_telemetry.py')],check=True,timeout=60)
    subprocess.run([sys.executable,'-u',str(ROOT/'benchmarks/te_io_probe.py'),
        '--directory',str(args.directory),'--output',str(args.output),'--blocks','10,25','--repeats','3'],check=True,timeout=900)


if __name__=='__main__':main()
