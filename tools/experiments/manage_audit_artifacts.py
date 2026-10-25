"""Inspect/compress generated traces, preserving verified reproducible evidence.

No checkpoint, tensor, user input, reference verdict or ordinary source file
is eligible. Destructive steps are opt-in and occur only after round-trip
SHA256 verification and a second unchanged-source check.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import time

ROOT=Path(__file__).resolve().parents[2]
ALLOWED_ROOTS=('kaggle_causal_audit_output','kaggle_causal_resume_output','kaggle_cpu_audit_output',
               'kaggle_te_critical_path_output','semantic_results/cpu_qualification',
               'semantic_results/te_loading_qualification','kaggle_te_io_output',
               'kaggle_residency_p01_output','kaggle_idle_residency_v2_output')


def digest(stream):
    value=hashlib.sha256();size=0
    while block:=stream.read(4*1024*1024):value.update(block);size+=len(block)
    return value.hexdigest(),size


def eligible(path):
    io_trace=(path.resolve().is_relative_to((ROOT/'kaggle_te_io_output').resolve()) and
              (path.name in ('io_probe.json','io_reads_r0.jsonl','io_reads_r1.jsonl','io_reads_rparent.jsonl')))
    matrix_summary=(path.resolve()==(ROOT/'kaggle_residency_p01_output/residency_results/residency_ablation.json').resolve())
    matrix_summary=matrix_summary or (path.resolve()==(ROOT/'kaggle_idle_residency_v2_output/idle_residency_results/idle_ablation.json').resolve())
    return (path.name.endswith(('_kineto.json','_causal.json')) or path.name=='timeline.html' or io_trace or matrix_summary)


def compress(path,prune):
    resolved=path.resolve()
    if not resolved.is_relative_to(ROOT) or path.is_symlink():raise ValueError('Artifact outside workspace')
    before=path.stat();signature=(before.st_size,before.st_mtime_ns)
    archive=path.with_suffix(path.suffix+'.gz')
    with path.open('rb') as source:expected,total=digest(source)
    if not archive.exists():
        temporary=archive.with_suffix(archive.suffix+'.partial')
        with path.open('rb') as source,temporary.open('xb') as output:
            with gzip.GzipFile(filename='',mode='wb',compresslevel=1,fileobj=output,mtime=0) as zipped:
                while block:=source.read(4*1024*1024):zipped.write(block)
        temporary.replace(archive)
    with gzip.open(archive,'rb') as source:actual,count=digest(source)
    if (actual,count)!=(expected,total):raise ValueError('Compressed evidence failed round-trip checksum')
    after=path.stat()
    if (after.st_size,after.st_mtime_ns)!=signature:raise RuntimeError('Active artifact changed; original retained')
    if prune:path.unlink()
    return dict(path=str(path.relative_to(ROOT)),archive=str(archive.relative_to(ROOT)),sha256=expected,
                original_bytes=total,compressed_bytes=archive.stat().st_size,original_removed=prune)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--compress',action='store_true')
    parser.add_argument('--prune-verified',action='store_true')
    parser.add_argument('--restore',type=Path,help='Restore one archived generated trace inside an allowed root')
    parser.add_argument('--root',action='append',choices=ALLOWED_ROOTS,default=[])
    parser.add_argument('--max-files',type=int,default=100)
    args=parser.parse_args()
    if args.restore:
        archive=args.restore.resolve()
        if archive.suffix!='.gz' or not any(archive.is_relative_to((ROOT/name).resolve()) for name in ALLOWED_ROOTS):
            parser.error('Restore only a generated archive inside an allowed artifact root')
        target=archive.with_suffix('')
        if not eligible(target) or target.exists():parser.error('Destination must be an absent generated trace')
        matching=[]
        for receipt in (ROOT/'semantic_results/artifact_retention').glob('compression_*.json'):
            matching += [r for r in json.loads(receipt.read_text())['files'] if (ROOT/r['archive']).resolve()==archive]
        if not matching:parser.error('A checksum receipt is required')
        temporary=target.with_suffix(target.suffix+'.restore.partial')
        with gzip.open(archive,'rb') as source,temporary.open('xb') as destination:
            while block:=source.read(4*1024*1024):destination.write(block)
        with temporary.open('rb') as source:sha,size=digest(source)
        if (sha,size)!=(matching[-1]['sha256'],matching[-1]['original_bytes']):raise ValueError('Restore checksum mismatch')
        temporary.replace(target);print('RESTORED',target.relative_to(ROOT));return
    if args.prune_verified and not args.compress:parser.error('Pruning requires verified compression')
    selected=[]
    for name in args.root or ALLOWED_ROOTS:
        folder=(ROOT/name).resolve()
        if not folder.is_relative_to(ROOT):raise ValueError('Unsafe artifact root')
        if folder.exists():
            selected.extend(p for p in folder.rglob('*') if p.is_file() and eligible(p) and not p.is_symlink())
    selected=sorted(set(selected),key=lambda p:p.stat().st_size,reverse=True)[:args.max_files]
    receipt=dict(time_ns=time.time_ns(),policy='Only generated trace JSON/HTML; preserved by round-trip SHA256',files=[])
    for path in selected:
        if args.compress:
            row=compress(path,args.prune_verified);receipt['files'].append(row)
            print('ARCHIVED',row['path'],row['original_bytes'],row['compressed_bytes'],flush=True)
        else:print('ELIGIBLE',path.relative_to(ROOT),path.stat().st_size)
    if args.compress:
        receipt['net_removed_bytes']=sum(r['original_bytes']-r['compressed_bytes'] for r in receipt['files'] if r['original_removed'])
        directory=ROOT/'semantic_results/artifact_retention';directory.mkdir(exist_ok=True)
        (directory/f'compression_{receipt["time_ns"]}.json').write_text(json.dumps(receipt,indent=2))
        print('NET REMOVED BYTES',receipt['net_removed_bytes'],flush=True)


if __name__=='__main__':main()
