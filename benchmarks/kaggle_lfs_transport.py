"""Upload weight bytes from Kaggle using file-scoped, expiring LFS capabilities.

The coordinator only authorizes hashes and commits pointers/small JSON evidence.
Its HF token is never included in the notebook, transport plan or output.
"""
import argparse
import copy
import hashlib
import json
import logging
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'benchmarks'))


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2))


def info(row):
    from huggingface_hub.lfs import UploadInfo
    return UploadInfo(sha256=bytes.fromhex(row['sha256']), size=row['bytes'], sample=b'')


def authorize(directory, output, repo, limit=None):
    from huggingface_hub import HfApi, get_token
    from huggingface_hub.lfs import post_lfs_batch_info
    plan = read(ROOT / 'semantic_results/offline_conversion_plan.json')
    api = HfApi(); api.whoami()
    rows = {}
    for stem, expected in sorted(plan['blocks'].items()):
        m = read(Path(directory) / (stem + '.json'))
        if m['state_sha256'] != expected['state_sha256'] or not m['native_tensor_bit_exact']:
            raise ValueError('Native proof mismatch')
        name = stem + '.safetensors'
        rows[name] = dict(sha256=m['file_sha256'], bytes=(Path(directory)/name).stat().st_size)
    if limit: rows = dict(list(rows.items())[:limit])
    actions = {}
    names = list(rows)
    for offset in range(0, len(names), 100):
        batch, errors, adapter = post_lfs_batch_info(
            [info(rows[n]) for n in names[offset:offset+100]],
            token=get_token(), repo_type='model', repo_id=repo, transfers=['basic','multipart'])
        if errors or adapter not in (None, 'basic', 'multipart'):
            raise ValueError('Unsupported LFS authorization response')
        for action in batch:
            # Verification/commit remain authenticated on the coordinator.
            action.get('actions', {}).pop('verify', None)
            actions[action['oid']] = action
    for row in rows.values(): row['action'] = actions[row['sha256']]
    write(output, dict(repo=repo, plan=plan, objects=rows, authorized_at=time.time()))
    hosts = set()
    from urllib.parse import urlparse
    for row in rows.values():
        action = row['action'].get('actions', {}).get('upload', {})
        if action: hosts.add(urlparse(action['href']).hostname)
    print('AUTHORIZED file-scoped objects', len(rows), 'upload hosts', sorted(hosts), flush=True)


def convert(ticket_path, checkpoints, directory, evidence):
    os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
    logging.getLogger('huggingface_hub').setLevel(logging.CRITICAL)
    import offline_checkpoint_store as store
    import torch
    from huggingface_hub import CommitOperationAdd
    from huggingface_hub.lfs import lfs_upload
    ticket = read(ticket_path); evidence = Path(evidence)
    if len(ticket['objects']) != 200:
        raise ValueError('Complete upload authorization required')
    receipts = {}; original = store.bind_block
    def bind_and_upload(root, stem, expected, reader):
        original(root, stem, expected, reader)
        name = stem + '.safetensors'; row = ticket['objects'][name]
        path = Path(root)/name
        operation = CommitOperationAdd(path_in_repo=name, path_or_fileobj=path)
        if (operation.upload_info.sha256.hex() != row['sha256'] or
                operation.upload_info.size != row['bytes']):
            raise ValueError('Target safetensors serialization differs from authorized Native artifact')
        started = time.monotonic()
        lfs_upload(operation=operation, lfs_batch_action=copy.deepcopy(row['action']), token=False)
        receipts[name] = dict(sha256=row['sha256'], bytes=row['bytes'],
                             upload_seconds=time.monotonic()-started)
        write(evidence/'upload_receipts.json', dict(repo=ticket['repo'], objects=receipts,
                                                   complete=len(receipts)==200))
        write(evidence/(stem+'.json'), read(Path(root)/(stem+'.json')))
        print('PUBLISHED Kaggle weight object', len(receipts), '/200', stem, flush=True)
    store.bind_block = bind_and_upload
    torch.set_num_threads(4); torch.set_grad_enabled(False)
    plan_path = evidence/'conversion_plan.json'; write(plan_path, ticket['plan'])
    store.prepare(plan_path, checkpoints, directory)
    write(evidence/'bundle.json', read(Path(directory)/'bundle.json'))
    print('READY Kaggle upload objects; repository pointer commit pending', flush=True)


def finalize(ticket_path, evidence):
    from huggingface_hub import HfApi, CommitOperationAdd, get_token
    from huggingface_hub.lfs import post_lfs_batch_info
    from huggingface_hub import set_client_factory
    import httpx
    # Metadata requests must tolerate a slow connection/upload without
    # re-running conversion or transmitting weight payloads.
    set_client_factory(lambda: httpx.Client(follow_redirects=True,
        timeout=httpx.Timeout(120, connect=30)))
    ticket = read(ticket_path); evidence = Path(evidence)
    receipts = read(evidence/'upload_receipts.json'); bundle = read(evidence/'bundle.json')
    if not (receipts['complete'] and bundle['complete'] and bundle['plan'] == ticket['plan']
            and set(receipts['objects']) == set(ticket['objects'])):
        raise ValueError('Complete matching Kaggle evidence required')
    print('VERIFIED complete Kaggle receipts and Native plan', flush=True)
    operations = []; api = HfApi(); api.whoami()
    print('VERIFIED HF coordinator identity', flush=True)
    for name, row in ticket['objects'].items():
        receipt = receipts['objects'][name]
        if any(receipt[k] != row[k] for k in ('sha256','bytes')):
            raise ValueError('Kaggle receipt mismatch')
    rows = list(ticket['objects'].values())
    for offset in range(0, len(rows), 100):
        batch, errors, _ = post_lfs_batch_info([info(r) for r in rows[offset:offset+100]],
            token=get_token(), repo_type='model', repo_id=ticket['repo'], transfers=['basic','multipart'])
        if errors or any(action.get('actions') for action in batch):
            raise ValueError('HF has not confirmed all uploaded objects')
        print('VERIFIED HF object availability', min(offset+100,len(rows)), '/200', flush=True)
    for name, row in ticket['objects'].items():
        op = CommitOperationAdd(path_in_repo=name, path_or_fileobj=b'')
        op.upload_info = info(row); op._upload_mode = 'lfs'; op._is_uploaded = True
        operations.append(op)
        stem = name.removesuffix('.safetensors'); metadata = evidence/(stem+'.json')
        m = read(metadata); expected = ticket['plan']['blocks'][stem]
        if (m['state_sha256'] != expected['state_sha256'] or
                m['quant_metadata'] != expected['quant_metadata'] or
                m['file_sha256'] != row['sha256'] or not m['native_tensor_bit_exact']):
            raise ValueError('Kaggle Native metadata mismatch')
        operations.append(CommitOperationAdd(path_in_repo=metadata.name, path_or_fileobj=metadata))
    operations.append(CommitOperationAdd(path_in_repo='bundle.json', path_or_fileobj=evidence/'bundle.json'))
    print('VERIFIED Native manifests; committing JSON and LFS pointers only', flush=True)
    commit = api.create_commit(repo_id=ticket['repo'], operations=operations,
                               commit_message='200 Native-verified rank-local shards uploaded by Kaggle')
    files = {f.rfilename:f for f in api.repo_info(ticket['repo'], revision=commit.oid, files_metadata=True).siblings}
    for name, row in ticket['objects'].items():
        f = files[name]
        if not f.lfs or f.lfs.sha256 != row['sha256'] or f.size != row['bytes']:
            raise ValueError('Final HF commit differs from Kaggle receipt')
    write(evidence/'publication.json', dict(repo=ticket['repo'], commit=commit.oid,
          bundle_sha256=hashlib.sha256((evidence/'bundle.json').read_bytes()).hexdigest(), complete=True))
    print('READY HF checkpoint', ticket['repo'], commit.oid, flush=True)


def main():
    p = argparse.ArgumentParser(); p.add_argument('mode', choices=['authorize','convert','finalize'])
    p.add_argument('--directory'); p.add_argument('--ticket', required=True)
    p.add_argument('--repo'); p.add_argument('--checkpoints'); p.add_argument('--evidence'); p.add_argument('--limit', type=int)
    a = p.parse_args()
    try:
        if a.mode == 'authorize': authorize(a.directory, a.ticket, a.repo, a.limit)
        elif a.mode == 'convert': convert(a.ticket, a.checkpoints, a.directory, a.evidence)
        else: finalize(a.ticket, a.evidence)
    except Exception as error:
        print('ERROR LFS transport', type(error).__name__, flush=True)
        raise SystemExit(1) from None


if __name__ == '__main__': main()
