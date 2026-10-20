"""Verify relocated rank manifests against trained Native tensor hashes."""
import copy
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'benchmarks'), str(ROOT)]
from offline_checkpoint_store import bind_block
from patches.windowed_tp import CheckpointReader


def main():
    plan = json.loads((ROOT/'semantic_results/offline_conversion_plan.json').read_text())
    folder = Path('D:/minimaxH3/rank_local_tp2_portable_test'); folder.mkdir(exist_ok=True)
    paths = {'te':'D:/minimaxH3/qwen3vl_32b_minimax_h3_int8_convrot.safetensors',
             'dit':'D:/minimaxH3/MiniMax_H3_Ref2VA_pruned_int8_convrot.safetensors'}
    cases = []
    for kind in ('te','dit'):
        reader = CheckpointReader(paths[kind])
        for rank in (0,1):
            stem = f'{kind}.block00.rank{rank}'; source = Path('D:/minimaxH3/rank_local_tp2')
            target = folder/(stem+'.safetensors')
            if not target.exists(): target.hardlink_to(source/target.name)
            metadata = json.loads((source/(stem+'.json')).read_text())
            metadata['source_identity'] = ['/former-kaggle-path',0,0]
            (folder/(stem+'.json')).write_text(json.dumps(metadata))
            bind_block(folder, stem, plan['blocks'][stem], reader)
            wrong = copy.deepcopy(plan['blocks'][stem]); key = next(iter(wrong['state_sha256']))
            wrong['state_sha256'][key]['sha256'] = '0'*64
            try: bind_block(folder, stem, wrong, reader)
            except ValueError: pass
            else: raise AssertionError('Incorrect Native proof was accepted')
            cases.append(dict(kind=kind,rank=rank,relocated_native_bits_verified=True,wrong_proof_rejected=True))
        print('QUALIFIED portable',kind,flush=True)
    paths = [ROOT/'benchmarks/offline_checkpoint_store.py',Path(__file__)]
    result = dict(passed=True,physical=False,cases=cases,source_sha256={p.relative_to(ROOT).as_posix():
        hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
    (ROOT/'semantic_results/offline_portability.json').write_text(json.dumps(result,indent=2))


if __name__ == '__main__': main()
