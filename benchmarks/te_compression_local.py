"""Local, lossless TE codec experiment; never imported by the H3 runtime.

Run with the validated conda Python and --deps pointing at isolated packages.
Codec arithmetic is entirely upstream Zstandard, Blosc2, FSE and OpenZL.
The small FSE frame below only stores mode/length (including raw/RLE fallback).
"""
from __future__ import annotations

import argparse
import concurrent.futures as futures
import ctypes as C
import gc
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import random
import statistics
import struct
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
MIB = 2**20


def save(path, value):
    path.write_text(json.dumps(value, indent=2), encoding='utf8')


def summary(values):
    ordered = sorted(values)
    return dict(samples_ms=values, median_ms=statistics.median(values),
                min_ms=min(values), max_ms=max(values),
                p95_ms=ordered[min(len(values)-1, int(.95*len(values)))])


class Codecs:
    def __init__(self, dll):
        import numpy as np
        import zstandard as zstd
        import blosc2
        self.np, self.zstd, self.blosc = np, zstd, blosc2
        blosc2.set_releasegil(True)
        self.local = threading.local()
        self.dll = C.CDLL(str(dll))
        for name in ('FSE_compress', 'FSE_decompress'):
            fn = getattr(self.dll, name)
            fn.argtypes = [C.c_void_p, C.c_size_t, C.c_void_p, C.c_size_t]
            fn.restype = C.c_size_t
        self.dll.FSE_compressBound.argtypes = [C.c_size_t]
        self.dll.FSE_compressBound.restype = C.c_size_t
        self.dll.FSE_isError.argtypes = [C.c_size_t]
        self.dll.FSE_isError.restype = C.c_uint
        for name in ('ZL_zigzagEncode8', 'ZL_zigzagDecode8'):
            fn = getattr(self.dll, name)
            fn.argtypes = [C.c_void_p, C.c_void_p, C.c_size_t]
            fn.restype = None

    def encode(self, codec, raw, typesize=1, threads=1):
        if codec == 'raw':
            return bytes(raw)
        if codec.startswith('zstd'):
            return self.zstd.ZstdCompressor(level=int(codec[4:])).compress(raw)
        if codec.startswith('blosc'):
            return self.blosc.compress2(raw, codec=self.blosc.Codec.ZSTD,
                clevel=int(codec[5:]), typesize=typesize, nthreads=threads,
                filters=[self.blosc.Filter.NOFILTER]*5+[self.blosc.Filter.BITSHUFFLE])
        # FSE's API takes bytes. OpenZL performs the signed INT8 mapping.
        n = len(raw)
        data = self.np.frombuffer(raw, dtype=self.np.uint8)
        if codec == 'zigfse':
            mapped = self.np.empty(n, dtype=self.np.uint8)
            self.dll.ZL_zigzagEncode8(mapped.ctypes.data, data.ctypes.data, n)
            data = mapped
        dest = self.np.empty(self.dll.FSE_compressBound(n), dtype=self.np.uint8)
        count = self.dll.FSE_compress(dest.ctypes.data, dest.nbytes, data.ctypes.data, n)
        if self.dll.FSE_isError(count):
            raise RuntimeError('Upstream FSE compression failed')
        if count == 0:
            return struct.pack('<BQ', 0, n) + data.tobytes()
        if count == 1:
            return struct.pack('<BQ', 1, n) + data[:1].tobytes()
        return struct.pack('<BQ', 2, n) + dest[:count].tobytes()

    def decode(self, codec, blob, threads=1):
        if codec == 'raw':
            # Materialize a new destination, like all decoder APIs.
            return bytearray(blob)
        if codec.startswith('zstd'):
            if not hasattr(self.local, 'zdc'):
                self.local.zdc = self.zstd.ZstdDecompressor()
            return self.local.zdc.decompress(blob)
        if codec.startswith('blosc'):
            return self.blosc.decompress2(blob, nthreads=threads)
        mode, n = struct.unpack_from('<BQ', blob)
        data = self.np.frombuffer(blob, dtype=self.np.uint8, offset=9)
        dest = self.np.empty(n, dtype=self.np.uint8)
        if mode == 0:
            dest[:] = data
        elif mode == 1:
            dest.fill(data[0])
        elif mode == 2:
            count = self.dll.FSE_decompress(dest.ctypes.data, n, data.ctypes.data, data.nbytes)
            if self.dll.FSE_isError(count) or count != n:
                raise RuntimeError('Upstream FSE decompression failed')
        else:
            raise ValueError('Invalid frame mode')
        if codec == 'zigfse':
            self.dll.ZL_zigzagDecode8(dest.ctypes.data, dest.ctypes.data, n)
        return dest


def prepare(args):
    """Use the actual production rank loader, including Native dtype casts."""
    sys.path[:0] = [str(ROOT), str(ROOT/'ComfyUI')]
    import torch
    from patches.windowed_tp import CheckpointReader
    from patches.rank_local_int8 import load_rank_block
    torch.set_num_threads(1)
    reader = CheckpointReader(args.checkpoint)
    result = dict(checkpoint=reader.identity, layers=args.layers, blocks=[],
                  source_signature=reader.signature, source_path=str(reader.path))
    for index in args.layers:
        for rank in (0, 1):
            begin = time.perf_counter()
            module = load_rank_block(reader, 'te', index, rank)
            block = dict(layer=index, rank=rank, tensors=[], bytes=0,
                         file=f'te.layer{index:02d}.rank{rank}.raw')
            with (args.output/block['file']).open('wb') as out:
                for name, tensor in module.state_dict().items():
                    raw = memoryview(tensor.contiguous().reshape(-1).view(torch.uint8).numpy()).cast('B')
                    out.write(raw)
                    block['tensors'].append(dict(name=name, dtype=str(tensor.dtype),
                        shape=list(tensor.shape), typesize=tensor.element_size(),
                        offset=block['bytes'], bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest()))
                    block['bytes'] += len(raw)
            result['blocks'].append(block)
            print('prepared', index, rank, block['bytes'], round(time.perf_counter()-begin, 3), flush=True)
            del module
    result['raw_bytes'] = sum(b['bytes'] for b in result['blocks'])
    save(args.output/'corpus.json', result)


def verify_corpus(args, corpus):
    """Integrity check outside the codec timers, including production casts."""
    for block in corpus['blocks']:
        with (args.output/block['file']).open('rb') as stream:
            for tensor in block['tensors']:
                stream.seek(tensor['offset'])
                raw = stream.read(tensor['bytes'])
                if len(raw) != tensor['bytes'] or hashlib.sha256(raw).hexdigest() != tensor['sha256']:
                    raise ValueError('Corpus changed after TP2 preparation')


def chunks(args, corpus, chunk_mib):
    """Never mix tensor dtypes, layer ownership, or TP ranks in a frame."""
    for bi, block in enumerate(corpus['blocks']):
        with (args.output/block['file']).open('rb', buffering=0) as stream:
            for tensor in block['tensors']:
                for start in range(0, tensor['bytes'], chunk_mib*MIB):
                    size = min(chunk_mib*MIB, tensor['bytes']-start)
                    raw = stream.read(size)
                    if len(raw) != size:
                        raise IOError('Truncated corpus')
                    yield dict(block=bi, tensor=tensor['name'], typesize=tensor['typesize'],
                               dtype=tensor['dtype'], raw_bytes=size), raw


def screen(args, codecs, corpus):
    cases = [(c, s) for s in args.chunks for c in args.codecs]
    random.Random(20261020).shuffle(cases)
    results = json.loads((args.output/'screen.json').read_text()) if (args.output/'screen.json').exists() else []
    done = {(r['codec'], r['chunk_mib']) for r in results}
    for codec, size in cases:
        if (codec, size) in done:
            continue
        directory = args.output/f'{codec}_{size}MiB'
        directory.mkdir(exist_ok=True)
        index = []
        encode_s = 0
        # Packed files use independently decodable frames and a JSON side index.
        with (directory/'frames.bin').open('wb') as out:
            for entry, raw in chunks(args, corpus, size):
                begin = time.perf_counter()
                blob = codecs.encode(codec, raw, entry['typesize'])
                encode_s += time.perf_counter()-begin
                decoded = codecs.decode(codec, blob)
                if hashlib.sha256(decoded).digest() != hashlib.sha256(raw).digest():
                    raise AssertionError((codec, size, entry))
                entry.update(offset=out.tell(), packed_bytes=len(blob))
                out.write(blob)
                index.append(entry)
        save(directory/'index.json', index)
        raw_n = sum(e['raw_bytes'] for e in index)
        stored = (directory/'frames.bin').stat().st_size+(directory/'index.json').stat().st_size
        # Decoder output is allocated/materialized each call. Validation is untimed.
        memory = (directory/'frames.bin').read_bytes()
        blobs = [memory[e['offset']:e['offset']+e['packed_bytes']] for e in index]
        del memory
        records = []
        for workers in args.workers:
            with futures.ThreadPoolExecutor(max_workers=workers) as pool:
                def one(blob):
                    decoded = codecs.decode(codec, blob)
                    return len(decoded)
                def run():
                    if workers == 1:
                        for blob in blobs:
                            one(blob)
                    else:
                        # Output buffers die in each worker; bounded by worker count.
                        for _ in pool.map(one, blobs):
                            pass
                run()
                samples, cpu = [], []
                for _ in range(args.repeats):
                    t, ct = time.perf_counter(), time.process_time()
                    run()
                    samples.append((time.perf_counter()-t)*1000)
                    cpu.append((time.process_time()-ct)*1000)
                rec = dict(workers=workers, **summary(samples), cpu_ms=cpu)
                rec['decode_GB_s'] = raw_n/(rec['median_ms']/1000)/1e9
                records.append(rec)
        row = dict(codec=codec, chunk_mib=size, raw_bytes=raw_n, stored_bytes=stored,
                   ratio=raw_n/stored, saved_fraction=1-stored/raw_n,
                   encode_GB_s=raw_n/encode_s/1e9, decode=records,
                   byte_exact=True, frames=len(index), path=str(directory))
        results.append(row)
        save(args.output/'screen.json', results)
        print('screen', codec, size, 'ratio', round(row['ratio'], 4),
              'GB/s', [(r['workers'], round(r['decode_GB_s'], 3)) for r in records], flush=True)
        del blobs
        gc.collect()


def run(args):
    if args.stage == 'prepare':
        if (args.output/'corpus.json').exists():
            raise FileExistsError('Preserve the existing corpus; choose a fresh output directory')
        import psutil
        save(args.output/'machine.json', dict(python=sys.executable, version=sys.version,
             platform=platform.platform(), cores=psutil.cpu_count(logical=False),
             logical=psutil.cpu_count(), ram=psutil.virtual_memory()._asdict(),
             packages={n:importlib.metadata.version(n) for n in ['numpy','torch','zstandard','blosc2']},
             argv=sys.argv))
        prepare(args)
    else:
        codecs = Codecs(args.output/'native'/'zigzag_fse.dll')
        corpus = json.loads((args.output/'corpus.json').read_text())
        verify_corpus(args, corpus)
        screen(args, codecs, corpus)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('stage', choices=['prepare', 'screen'])
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--deps', type=Path, required=True)
    ap.add_argument('--checkpoint', default=r'D:\minimaxH3\qwen3vl_32b_minimax_h3_int8_convrot.safetensors')
    ap.add_argument('--layers', type=int, nargs='+', default=[0, 24, 49])
    ap.add_argument('--chunks', type=int, nargs='+', default=[1, 4, 16, 64])
    ap.add_argument('--codecs', nargs='+', default=['raw', 'zstd1', 'zstd3', 'blosc1', 'blosc3', 'zigfse', 'fse'])
    ap.add_argument('--workers', type=int, nargs='+', default=[1, 2, 4, 6])
    ap.add_argument('--repeats', type=int, default=3)
    args = ap.parse_args()
    if min(args.chunks+args.workers+[args.repeats]) <= 0:
        ap.error('Chunk sizes, worker counts and repeat count must be positive')
    allowed = {'raw','zstd1','zstd3','blosc1','blosc3','zigfse','fse'}
    if not set(args.codecs) <= allowed:
        ap.error('Unsupported codec: '+str(set(args.codecs)-allowed))
    sys.path.insert(0, str(args.deps))
    args.output.mkdir(parents=True, exist_ok=True)
    # Prevent overlapping CPU experiments and concurrent result writers.
    from filelock import FileLock
    with FileLock(args.output/'benchmark.lock', timeout=0):
        run(args)


if __name__ == '__main__':
    main()
