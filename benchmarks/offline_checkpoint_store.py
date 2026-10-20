"""Prepare once, publish verified chunks, and bind portable shards on reuse.

Native equality is inherited from the complete local trained tensor proof,
then checked by hash after target-host conversion. No model operators change.
"""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'benchmarks')]
import retest
import torch
from patches.windowed_tp import CheckpointReader
from patches.rank_local_int8 import CONFIG, FORMAT, load_rank_block, load_offline_block, state_hashes


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def atomic_json(path, value):
    path = Path(path); temporary = path.with_suffix(path.suffix+'.partial')
    temporary.write_text(json.dumps(value, indent=2)); temporary.replace(path)


def make_plan(directory, output):
    proof = ROOT/'semantic_results/rank_local_loader_final.json'
    result = json.loads(proof.read_text())
    ancestor = ROOT/'semantic_results/rank_local_loader_full.json'
    if not (result['passed'] and len(result['cases']) == 200 and
            result['native_evidence_parent_sha256'] == digest(ancestor)):
        raise ValueError('Complete current Native tensor proof required')
    for name, sha in result['source_sha256'].items():
        if digest(ROOT/name) != sha: raise ValueError('Loader proof is stale')
    checkpoints = json.loads((ROOT/'semantic_results/pipeline_checkpoint_identity.json').read_text())
    blocks = {}
    for row in result['cases']:
        stem = f"{row['kind']}.block{row['index']:02d}.rank{row['rank']}"
        m = json.loads((Path(directory)/(stem+'.json')).read_text())
        if not (row['native_tensor_bit_exact'] and row['offline_tensor_bit_exact'] and m['native_tensor_bit_exact']):
            raise ValueError('Block lacks Native equality proof')
        blocks[stem] = {k:m[k] for k in ('kind','index','rank','state_sha256','quant_metadata')}
    atomic_json(output, dict(format=FORMAT, configuration=CONFIG, blocks=blocks,
        sources={k:{n:checkpoints[k][n] for n in ('bytes','sha256')} for k in ('te','dit')},
        native_proof_sha256=digest(ancestor), loader_proof_sha256=digest(proof)))


def bind_block(directory, stem, expected, reader):
    """Portable provenance checked before rebinding the existing strict reader."""
    root = Path(directory); path = root/(stem+'.safetensors'); metadata = root/(stem+'.json')
    m = json.loads(metadata.read_text())
    if (m['format'] != FORMAT or m['configuration'] != CONFIG or
        m['state_sha256'] != expected['state_sha256'] or not m['native_tensor_bit_exact'] or
        m['quant_metadata'] != expected['quant_metadata'] or
        any(m[k] != expected[k] for k in ('kind','index','rank')) or digest(path) != m['file_sha256']):
        raise ValueError('Portable shard integrity/provenance mismatch')
    m['source_identity'] = reader.identity
    atomic_json(metadata, m)
    module = load_offline_block(reader, expected['kind'], expected['index'], expected['rank'], root)
    del module; gc.collect()


def prepare(plan_path, paths_path, directory, *, repo=None, secret=None, reuse_revision=None):
    from safetensors.torch import save_file
    plan = json.loads(Path(plan_path).read_text()); root = Path(directory).resolve()
    root.mkdir(parents=True, exist_ok=True)
    if plan['format'] != FORMAT or plan['configuration'] != CONFIG or len(plan['blocks']) != 200:
        raise ValueError('Unsupported or incomplete portable plan')
    paths = json.loads(Path(paths_path).read_text()); readers = {}
    for kind, source in plan['sources'].items():
        reader = CheckpointReader(paths[kind])
        if reader.path.stat().st_size != source['bytes'] or digest(reader.path) != source['sha256']:
            raise ValueError('Source checkpoint checksum mismatch')
        readers[kind] = reader
    api = None
    if repo:
        from huggingface_hub import HfApi, snapshot_download
        token = None
        if secret:
            from kaggle_secrets import UserSecretsClient
            token = UserSecretsClient().get_secret(secret)
        api = HfApi(token=token)
        if reuse_revision:
            if len(reuse_revision) != 40 or any(c not in '0123456789abcdef' for c in reuse_revision):
                raise ValueError('Reuse requires an immutable HF commit SHA')
            snapshot_download(repo, revision=reuse_revision, token=token, local_dir=root,
                              allow_patterns=['*.safetensors','*.json'], max_workers=2)
        else:
            # Authentication fails before expensive conversion, never after it.
            api.whoami()
            api.create_repo(repo_id=repo, private=True, exist_ok=True)
    if reuse_revision:
        index = json.loads((root/'bundle.json').read_text())
        if not index['complete'] or index['plan'] != plan:
            raise ValueError('HF bundle is incomplete or belongs to another Native proof')
    start = time.monotonic(); pending = []; completed = []
    for stem, expected in sorted(plan['blocks'].items()):
        reader = readers[expected['kind']]
        path = root/(stem+'.safetensors'); metadata = root/(stem+'.json')
        if reuse_revision and not (path.exists() and metadata.exists()):
            raise ValueError('HF bundle is missing a required shard')
        if not (path.exists() and metadata.exists()):
            module = load_rank_block(reader, expected['kind'], expected['index'], expected['rank'])
            if state_hashes(module) != expected['state_sha256']:
                raise ValueError('Target conversion differs from Native-validated tensor bytes')
            temporary = root/(stem+'.partial')
            save_file(module.state_dict(), str(temporary), metadata={'format':FORMAT})
            temporary.replace(path); del module; gc.collect()
            atomic_json(metadata, dict(format=FORMAT, configuration=CONFIG, source_identity=reader.identity,
                **expected, native_tensor_bit_exact=True, file_sha256=digest(path),
                native_proof_sha256=plan['native_proof_sha256'], native_execution_this_conversion=False))
        bind_block(root, stem, expected, reader)
        completed.append(stem); pending.extend([path.name, metadata.name])
        atomic_json(root/'progress.json', dict(complete=False, blocks=completed, sources=plan['sources']))
        if api and not reuse_revision and (len(completed)%10 == 0 or len(completed) == 200):
            commit = api.upload_folder(repo_id=repo, folder_path=root,
                allow_patterns=[*pending, 'progress.json'], commit_message=f'Verified shards {len(completed)}/200')
            pending.clear(); print('PUBLISHED verified shards', len(completed), commit.oid, flush=True)
        if len(completed)%10 == 0: print('VERIFIED portable shards', len(completed), flush=True)
    bundle = dict(complete=True, plan=plan, blocks=completed, seconds=time.monotonic()-start,
                  native_execution_this_conversion=False)
    atomic_json(root/'bundle.json', bundle)
    if api and not reuse_revision:
        commit = api.upload_file(repo_id=repo, path_or_fileobj=root/'bundle.json', path_in_repo='bundle.json',
            commit_message='Mark all 200 Native-verified shards ready')
        atomic_json(root/'publication.json', dict(repo=repo, commit=commit.oid,
            bundle_sha256=digest(root/'bundle.json'), complete=True))
        print('READY HF checkpoint', repo, commit.oid, flush=True)
    return bundle


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--make-plan', action='store_true')
    p.add_argument('--directory', required=True)
    p.add_argument('--plan', required=True)
    p.add_argument('--checkpoints')
    p.add_argument('--repo')
    p.add_argument('--secret')
    p.add_argument('--reuse-revision')
    a = p.parse_args(); torch.set_num_threads(4); torch.set_grad_enabled(False)
    try:
        if a.make_plan: make_plan(a.directory, a.plan)
        else:
            if not a.checkpoints: p.error('--checkpoints is required')
            prepare(a.plan, a.checkpoints, a.directory, repo=a.repo, secret=a.secret, reuse_revision=a.reuse_revision)
    except Exception as error:
        # HTTP exception strings may contain signed URLs or credentials.
        print('ERROR offline store:', type(error).__name__, flush=True); raise SystemExit(1) from None


if __name__ == '__main__': main()
