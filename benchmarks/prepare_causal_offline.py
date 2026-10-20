"""Public/private offline reuse, or Native-verified regeneration, on Kaggle only."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import gc
import hashlib
import json
from pathlib import Path
import sys
import time
from urllib.parse import parse_qs,urlparse

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'benchmarks'),str(ROOT/'ComfyUI')]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--mode',choices=('public','secret','regenerate'),required=True)
    parser.add_argument('--download-only',action='store_true')
    parser.add_argument('--bind-only',action='store_true')
    parser.add_argument('--directory',type=Path,required=True)
    parser.add_argument('--evidence',type=Path,required=True)
    parser.add_argument('--checkpoints',type=Path,default=ROOT/'checkpoint_paths.json')
    parser.add_argument('--qualify-one-block',action='store_true',help='Local correctness only, no performance claim')
    args=parser.parse_args()
    if args.mode in ('public','secret'):
        try:
            token=False
            if args.mode=='secret':
                from kaggle_secrets import UserSecretsClient
                token=UserSecretsClient().get_secret('HF_token')
            from huggingface_hub import get_hf_file_metadata,hf_hub_url
            folder=ROOT/'semantic_results/kaggle_hf_publication_v6'
            publication=json.loads((folder/'publication.json').read_text())
            bundle=json.loads((folder/'bundle.json').read_text())
            receipts=json.loads((folder/'upload_receipts.json').read_text())
            def authorize(item):
                name,expected=item
                metadata=get_hf_file_metadata(hf_hub_url(publication['repo'],name,revision=publication['commit']),token=token)
                expires=int(parse_qs(urlparse(metadata.location).query).get('Expires',['0'])[0])
                if metadata.size!=expected['bytes'] or metadata.etag!=expected['sha256'] or expires<time.time()+1200:
                    raise ValueError('Immutable offline object identity or expiry mismatch')
                manifest=json.loads((folder/(name.removesuffix('.safetensors')+'.json')).read_text())
                return name,dict(bytes=expected['bytes'],sha256=expected['sha256'],url=metadata.location,expires=expires,manifest=manifest)
            # Signed URLs and credentials stay in /tmp, never evidence output.
            ticket=Path('/tmp/causal_offline_ticket.json')
            if not args.bind_only:
                with ThreadPoolExecutor(max_workers=4) as pool:
                    objects=dict(pool.map(authorize,receipts['objects'].items()))
                ticket.write_text(json.dumps(dict(repo=publication['repo'],revision=publication['commit'],
                    bundle_sha256=publication['bundle_sha256'],bundle=bundle,objects=objects)))
            del token
            import subprocess
            command=[sys.executable,str(ROOT/'benchmarks/restore_offline_bundle.py'),
                '--ticket',str(ticket),'--directory',str(args.directory),
                '--checkpoints',str(args.checkpoints),'--evidence',str(args.evidence)]
            if args.download_only:command+=['--download-only']
            if args.bind_only:command+=['--bind-only']
            subprocess.run(command,check=True,timeout=1800)
            if not args.download_only:ticket.unlink(missing_ok=True)
            return
        except Exception as error:
            # Authentication/HTTP exceptions can contain a private URL. Do not
            # expose their message or chained traceback to notebook output.
            args.evidence.write_text(json.dumps(dict(passed=False,source=args.mode,error_type=type(error).__name__)))
            print('AUDIT offline reuse unavailable',type(error).__name__,flush=True)
            raise SystemExit(2) from None
    import torch
    from patches.windowed_tp import CheckpointReader,prepare_tp_block_legacy
    from patches.rank_local_int8 import export_block,load_offline_block
    torch.set_num_threads(4);torch.set_grad_enabled(False)
    paths=json.loads(args.checkpoints.read_text())
    plan=json.loads((ROOT/'semantic_results/offline_conversion_plan.json').read_text())
    args.directory.mkdir(parents=True,exist_ok=True)
    begin=time.monotonic();cases=0
    for kind in ('te','dit'):
        reader=CheckpointReader(paths[kind])
        for index in range(50):
            for rank in (0,1):
                path=export_block(reader,kind,index,rank,args.directory,native_comparator=prepare_tp_block_legacy)
                stem=path.stem
                manifest=json.loads(path.with_suffix('.json').read_text())
                expected=plan['blocks'][stem]
                if any(manifest[k]!=expected[k] for k in ('kind','index','rank','state_sha256','quant_metadata')):
                    raise ValueError('Regenerated shard differs from immutable Native-qualified plan')
                module=load_offline_block(reader,kind,index,rank,args.directory)
                del module;gc.collect();cases+=1
                if args.qualify_one_block:
                    args.evidence.write_text(json.dumps(dict(passed=True,cases=1,native_tensor_bit_exact=True,
                        immutable_plan_match=True,physical=False,scope='Target-host regeneration helper correctness only')))
                    return
            if index%10==9:print('VERIFIED regenerated offline',kind,index+1,flush=True)
    args.evidence.write_text(json.dumps(dict(passed=True,source='Kaggle regeneration after Direct diagnostics',
        cases=cases,native_tensor_bit_exact=True,immutable_plan_match=True,seconds=time.monotonic()-begin),indent=2))


if __name__=='__main__':main()
