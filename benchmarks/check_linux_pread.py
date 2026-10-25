"""Real Linux positional reader, short reads, fallback and concurrent ranges."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import tempfile
from unittest.mock import patch
import torch
from safetensors.torch import save_file
from patches.windowed_tp import CheckpointReader


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    if not hasattr(os,'pread'):raise RuntimeError('A Linux positional descriptor is required')
    records=[]
    with tempfile.TemporaryDirectory() as folder:
        path=Path(folder)/'fixture.safetensors'
        tensors={'matrix':torch.arange(77,dtype=torch.float32).reshape(11,7),
                 'bf16':torch.arange(24,dtype=torch.bfloat16).reshape(6,4),
                 'cube':torch.arange(24,dtype=torch.int8).reshape(2,3,4),'scalar':torch.tensor(2.)}
        save_file(tensors,str(path));reader=CheckpointReader(path);native=os.pread
        calls=[]
        def positional(fd,size,offset):
            calls.append((size,offset));return native(fd,size,offset)
        def check():
            for name,value in tensors.items():assert torch.equal(reader.read_tensor(name),value)
            actual=reader.read_tensor('matrix',rows=[(7,10),(1,3)],columns=(2,6))
            expected=torch.cat([tensors['matrix'][7:10,2:6],tensors['matrix'][1:3,2:6]])
            assert torch.equal(actual,expected)
        with patch('patches.windowed_tp.os.pread',side_effect=positional):check()
        records.append(dict(name='native_linux_pread',status='byte_exact',syscalls=len(calls)))
        with patch('patches.windowed_tp.os.pread',side_effect=lambda fd,size,offset:native(fd,min(size,7),offset)):check()
        records.append(dict(name='short_native_pread',status='byte_exact'))
        with patch('patches.windowed_tp.os.pread',side_effect=OSError('unsupported fd')):check()
        records.append(dict(name='seek_readinto_fallback',status='byte_exact'))
        with ThreadPoolExecutor(4) as pool:list(pool.map(lambda _:check(),range(12)))
        records.append(dict(name='concurrent_real_descriptors',status='byte_exact',workers=4))
        partial_calls=0
        def partial(fd,size,offset):
            nonlocal partial_calls
            partial_calls+=1
            if partial_calls==1:return native(fd,min(size,7),offset)
            raise OSError('fallback after partial read')
        with patch('patches.windowed_tp.os.pread',side_effect=partial):check()
        records.append(dict(name='fallback_after_partial_pread',status='byte_exact'))
        with patch('patches.windowed_tp.os.pread',return_value=b''):
            try:reader.read_tensor('matrix')
            except IOError:pass
            else:raise AssertionError('Truncation must fail')
        records.append(dict(name='truncated_range',status='rejected'))
    args.output.write_text(json.dumps(dict(status='passed',scope='Real Linux CPU descriptor gate, no GPU/I-O speed claim',
                                          torch=torch.__version__,records=records),indent=2))


if __name__=='__main__':main()
