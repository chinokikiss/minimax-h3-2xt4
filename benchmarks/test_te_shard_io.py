"""Check candidate I/O materialization against Native-qualified tensor hashes."""
from pathlib import Path
import hashlib
import json
import os
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from te_shard_io import read,layout,MODES


def main():
    root=Path(__file__).resolve().parents[1];cases=[];torch.set_num_threads(1)
    for rank in (0,1):
        path=Path(f'D:/minimaxH3/rank_local_tp2/te.block10.rank{rank}.safetensors')
        expected=json.loads(path.with_suffix('.json').read_text())['state_sha256']
        for mode in MODES:
            if mode in ('sequential','willneed','pread') and os.name!='posix':continue
            state=read(path,list(expected),mode)
            actual={k:dict(dtype=str(t.dtype),shape=list(t.shape),sha256=hashlib.sha256(
                memoryview(t.reshape(-1).view(torch.uint8).numpy()).cast('B')).hexdigest()) for k,t in state.items()}
            assert actual==expected,(rank,mode)
            cases.append(dict(rank=rank,mode=mode,all_tensors_native_bit_exact=True));del state
    files=['benchmarks/te_shard_io.py','benchmarks/test_te_shard_io.py']
    result=dict(passed=True,physical=False,scope='Actual trained TE shards, Windows local byte-equivalence; no T4 performance',
        cases=cases,layout=layout(path,list(expected)),source_sha256={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in files})
    (root/'semantic_results/te_shard_io_qualification.json').write_text(json.dumps(result,indent=2))
    print('VERIFIED shard I/O Native tensor bytes',len(cases),'local cases',flush=True)


if __name__=='__main__':main()
