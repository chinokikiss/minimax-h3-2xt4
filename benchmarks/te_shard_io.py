"""I/O-only candidate readers; no inference patch, layout or numerical change."""
import math
import mmap
import os
from pathlib import Path
import struct
import json
import torch
from patches.windowed_tp import CheckpointReader

MODES=('baseline','sorted','chunked','sequential','willneed','pread','mmap')


def layout(path,names):
    reader=CheckpointReader(path)
    items=[dict(name=n,offset=reader.offset+reader.header[n]['data_offsets'][0],
        bytes=reader.header[n]['data_offsets'][1]-reader.header[n]['data_offsets'][0]) for n in names]
    cursor=reader.offset;backwards=travel=0
    for item in items:
        backwards+=item['offset']<cursor;travel+=abs(item['offset']-cursor)
        cursor=item['offset']+item['bytes']
    return dict(file_bytes=Path(path).stat().st_size,payload_bytes=sum(i['bytes'] for i in items),
        baseline_backward_seeks=backwards,baseline_seek_distance=travel,items=items)


def read(path,names,mode,*,chunk_bytes=8*2**20):
    if mode not in MODES:raise ValueError('Unknown I/O mode')
    reader=CheckpointReader(path)
    if mode=='baseline':return {name:reader.read_tensor(name) for name in names}
    ordered=sorted(names,key=lambda n:reader.header[n]['data_offsets'][0]);state={}
    with Path(path).open('rb',buffering=0) as stream:
        if mode in ('sequential','willneed') and not hasattr(os,'posix_fadvise'):
            raise NotImplementedError('Linux descriptor advice required')
        if mode=='sequential':os.posix_fadvise(stream.fileno(),0,0,os.POSIX_FADV_SEQUENTIAL)
        mapping=mmap.mmap(stream.fileno(),0,access=mmap.ACCESS_COPY) if mode=='mmap' else None
        try:
            for name in ordered:
                info=reader.header[name];a,b=info['data_offsets'];size=b-a;offset=reader.offset+a
                value=torch.empty(info['shape'],dtype=reader.DTYPES[info['dtype']])
                destination=memoryview(value.reshape(-1).view(torch.uint8).numpy()).cast('B')
                if len(destination)!=size:raise ValueError('Invalid tensor extent')
                if mode=='mmap':
                    source=torch.frombuffer(mapping,dtype=value.dtype,count=math.prod(info['shape']),offset=offset)
                    value.copy_(source.reshape(info['shape']));del source
                else:
                    cursor=0
                    if mode!='pread':stream.seek(offset)
                    while cursor<size:
                        length=min(chunk_bytes,size-cursor) if mode in ('chunked','willneed') else size-cursor
                        if mode=='willneed' and cursor+length<size:
                            os.posix_fadvise(stream.fileno(),offset+cursor+length,
                                min(chunk_bytes,size-cursor-length),os.POSIX_FADV_WILLNEED)
                        remaining=destination[cursor:cursor+length]
                        while remaining:
                            if mode=='pread':
                                if not hasattr(os,'preadv'):raise NotImplementedError('preadv required')
                                count=os.preadv(stream.fileno(),[remaining],offset+cursor)
                            else:count=stream.readinto(remaining)
                            if not count:raise IOError('Incomplete immutable shard')
                            cursor+=count;remaining=remaining[count:]
                state[name]=value
                del destination,value
        finally:
            if mapping is not None:mapping.close()
    return {name:state[name] for name in names}
