"""Trained Native shard equality and local CPU loader comparison.

Local I/O/CPU timings are not evidence of physical dual-T4 pipeline latency.
Offline conversion is rejected unless every saved tensor equals Native.
"""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT),os.environ.get('COMFY_ROOT',str(ROOT/'ComfyUI')),str(ROOT/'benchmarks')]
import psutil
import torch
from patches.windowed_tp import CheckpointReader,prepare_tp_block,prepare_tp_block_legacy
from patches.rank_local_int8 import export_block,load_offline_block,state_hashes

PATHS = dict(te='D:/minimaxH3/qwen3vl_32b_minimax_h3_int8_convrot.safetensors',
             dit='D:/minimaxH3/MiniMax_H3_Ref2VA_pruned_int8_convrot.safetensors')


def measured(fn):
    process = psutil.Process(); initial = process.memory_info().rss
    peak = [initial]; stop = threading.Event()
    def monitor():
        while not stop.wait(.005):
            peak[0] = max(peak[0],process.memory_info().rss)
    thread = threading.Thread(target=monitor); thread.start()
    start = time.perf_counter()
    try:
        value = fn()
        seconds = time.perf_counter()-start
        peak[0] = max(peak[0],process.memory_info().rss)
    finally:
        stop.set(); thread.join()
    return value,dict(seconds=seconds,process_rss_before=initial,
                      peak_process_rss=peak[0],rss_increase=max(0,peak[0]-initial))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--indices',type=int,nargs='+',default=[0,24,49])
    parser.add_argument('--convert-all',action='store_true')
    parser.add_argument('--verify-existing',action='store_true',
        help='Verify current reads against existing Native-validated artifact hashes; no reconversion')
    parser.add_argument('--directory',type=Path,default=Path('D:/minimaxH3/rank_local_tp2'))
    parser.add_argument('--output',type=Path,default=ROOT/'semantic_results/rank_local_loader.json')
    args = parser.parse_args()
    torch.set_num_threads(4)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    result = dict(passed=False,scope='Local CPU loader; physical GPU performance unmeasured',
        cache_coldness='OS page cache uncontrolled; comparison is warm and order-sensitive',
        rss_scope='5ms sampled process RSS; allocator history can affect deltas',cases=[],
        source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (ROOT/'patches/rank_local_int8.py',ROOT/'patches/windowed_tp.py',Path(__file__))})
    if args.verify_existing:
        parent = ROOT/'semantic_results/rank_local_loader_full.json'
        evidence = json.loads(parent.read_text())
        if not evidence['passed'] or len(evidence['cases']) != 200:
            raise ValueError('Complete Native-validated conversion must precede reuse')
        result['native_evidence_parent_sha256'] = hashlib.sha256(parent.read_bytes()).hexdigest()
        result['native_execution_this_run'] = False
    indices = range(50) if args.convert_all else args.indices
    for kind,path in PATHS.items():
        reader = CheckpointReader(path)
        for index in indices:
            for rank in (0,1):
                if args.verify_existing:
                    manifest = json.loads((args.directory/f'{kind}.block{index:02d}.rank{rank}.json').read_text())
                    if not manifest['native_tensor_bit_exact']:
                        raise AssertionError('Artifact lacks Native equality proof')
                    native_hash = manifest['state_sha256']
                    baseline = next(row['legacy'] for row in evidence['cases']
                        if (row['kind'],row['index'],row['rank']) == (kind,index,rank))
                else:
                    native,baseline = measured(lambda:prepare_tp_block_legacy(reader,kind,index,rank))
                    native_hash = state_hashes(native); del native; gc.collect()
                before = reader.logical_read_bytes; before_calls = reader.range_read_calls
                direct,direct_time = measured(lambda:prepare_tp_block(reader,kind,index,rank))
                if state_hashes(direct) != native_hash:
                    raise AssertionError(f'Native tensor mismatch {kind}/{index}/{rank}')
                direct_time.update(logical_read_bytes=reader.logical_read_bytes-before,
                    range_read_calls=reader.range_read_calls-before_calls)
                del direct; gc.collect()
                # Reuse the measured Native state hash, while requiring export
                # to compare with a genuinely Native-loaded module once more.
                if not args.verify_existing:
                    export_block(reader,kind,index,rank,args.directory,native_comparator=prepare_tp_block_legacy)
                offline,offline_time = measured(lambda:load_offline_block(reader,kind,index,rank,args.directory))
                if state_hashes(offline) != native_hash:
                    raise AssertionError('Offline Native tensor mismatch')
                offline_time['logical_read_bytes'] = offline._rank_offline_read_bytes
                del offline; gc.collect()
                offline,warm_time = measured(lambda:load_offline_block(reader,kind,index,rank,args.directory))
                if state_hashes(offline) != native_hash:
                    raise AssertionError('Warm offline Native tensor mismatch')
                warm_time['logical_read_bytes'] = offline._rank_offline_read_bytes
                prefix=f'model.layers.{index}.' if kind=='te' else f'blocks.{index}.'
                baseline['logical_read_bytes'] = reader.bytes(prefix)
                result['cases'].append(dict(kind=kind,index=index,rank=rank,native_tensor_bit_exact=True,
                    offline_tensor_bit_exact=True,legacy=baseline,direct=direct_time,
                    offline_first_verified_load=offline_time,offline_warm=warm_time))
                del offline; gc.collect()
                args.output.write_text(json.dumps(result,indent=2))
                print('QUALIFIED',kind,index,rank,'seconds',round(baseline['seconds'],3),
                      round(direct_time['seconds'],3),round(offline_time['seconds'],3),
                      round(warm_time['seconds'],3),flush=True)
    result['passed'] = True
    args.output.write_text(json.dumps(result,indent=2))


if __name__ == '__main__':main()
