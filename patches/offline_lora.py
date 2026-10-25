"""Offline Native FP32 LoRA merge followed by full-width INT8 ConvRot.

Creates a new checkpoint; never edits or uploads the base. --prefix supports
small reproducible block fixtures. Full conversion copies the base once and
patches selected tensors with whole layers assigned to up to two GPU workers.
Each worker processes one full-width matrix at a time.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import re
import math
import sys
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED,CancelledError
import multiprocessing as mp
import torch
from patches.windowed_tp import CheckpointReader
from patches.h3_lora import H3LoRAStore

_BAKE_CANCEL=None


def _worker_init(cancel):
    global _BAKE_CANCEL
    _BAKE_CANCEL=cancel


def _check_cancel():
    if _BAKE_CANCEL is not None and _BAKE_CANCEL.is_set():raise CancelledError('Offline LoRA baking cancelled')
    management=sys.modules.get('comfy.model_management')
    if management is not None:management.throw_exception_if_processing_interrupted()


def merged_weight(reader,store,target,device='cuda:0'):
    from comfy_kitchen.tensor.int8 import TensorWiseINT8Layout as Layout
    from comfy.weight_adapter.lora import LoRAAdapter
    source=reader.read_tensor(target+'.weight').to(device)
    original_dtype=source.dtype
    if original_dtype==torch.int8:
        config=json.loads(bytes(reader.read_tensor(target+'.comfy_quant').tolist()).decode())
        if config!={'format':'int8_tensorwise','convrot':True,'convrot_groupsize':256}:
            raise ValueError('Unsupported offline quantization metadata')
        scale=reader.read_tensor(target+'.weight_scale').to(device)
        params=Layout.Params(scale=scale,orig_dtype=torch.float32,orig_shape=tuple(source.shape),
                             is_weight=True,convrot=True,convrot_groupsize=256)
        value=Layout.dequantize(source,params);del scale,params
    elif original_dtype in (torch.float16,torch.bfloat16,torch.float32):value=source.float()
    else:raise ValueError('Unsupported offline base weight dtype')
    del source
    for down,up,strength in store.terms(target):
        down=down.to(device,dtype=torch.float32);up=up.to(device,dtype=torch.float32)
        if not torch.isfinite(down).all() or not torch.isfinite(up).all():raise ValueError('Nonfinite LoRA factors')
        adapter=LoRAAdapter(set(),(up,down,None,None,None,None))
        applied=[]
        def apply(delta):applied.append(True);return delta
        value=adapter.calculate_weight(value,target,strength,1.,None,apply)
        if applied!=[True]:raise RuntimeError('Native LoRA merge did not apply the adapter')
        del down,up,adapter
    if not torch.isfinite(value).all():raise ValueError('Nonfinite offline merged weights')
    if original_dtype!=torch.int8:return value.to(original_dtype).cpu().contiguous(),None
    # Quantize the full input width BEFORE splitting into TP ranks. Each row
    # shard must use the same Native weight scale, not an independent half max.
    q,params=Layout.quantize(value,per_channel=True,convrot=True,convrot_groupsize=256,
                             stochastic_rounding=0)
    return q.cpu().contiguous(),params.scale.cpu().contiguous()


def layer_partitions(reader,targets,count):
    """Balance whole layers; a matrix and its full-width scale never split GPUs."""
    groups={}
    for target in targets:
        match=re.match(r'^(blocks\.\d+|token_refiner\.blocks\.\d+)\.',target)
        if match is None:raise ValueError('Unqualified bake layer: '+target)
        groups.setdefault(match[1],[]).append(target)
    size=lambda group:sum(math.prod(reader.header[t+'.weight']['shape']) for t in group)
    parts=[[] for _ in range(count)];cost=[0]*count
    for _,group in sorted(groups.items(),key=lambda item:(-size(item[1]),item[0])):
        rank=min(range(count),key=lambda i:(cost[i],i))
        parts[rank].extend(sorted(group));cost[rank]+=size(group)
    return parts


def _bake_partition(base,lora_stack,partial,targets,device,verify_native,comfy_root):
    if comfy_root:sys.path.insert(0,comfy_root)
    torch.set_grad_enabled(False);torch.backends.cuda.matmul.allow_tf32=False
    if torch.device(device).type=='cuda':torch.cuda.set_device(device)
    reader=CheckpointReader(base);store=H3LoRAStore(tuple(lora_stack),reader)
    destination=CheckpointReader(partial);checks=[]
    # Independent descriptors and disjoint ranges. Only the parent publishes
    # the checkpoint after every worker has completed and fsynced its writes.
    with Path(partial).open('r+b',buffering=0) as stream:
        for target in targets:
            _check_cancel()
            q,scale=merged_weight(reader,store,target,device)
            if verify_native:
                from patches.offline_lora_reference import verify_weight
                verify_weight(reader,store,target,q,scale,device)
            for suffix,value in ([('weight',q),('weight_scale',scale)] if scale is not None else [('weight',q)]):
                name=target+'.'+suffix;info=destination.header[name]
                raw=memoryview(value.reshape(-1).view(torch.uint8).numpy()).cast('B')
                if list(value.shape)!=info['shape'] or len(raw)!=info['data_offsets'][1]-info['data_offsets'][0]:
                    raise ValueError('Offline serialization would change Native schema: '+name)
                stream.seek(destination.offset+info['data_offsets'][0]);remaining=raw
                while remaining:
                    n=stream.write(remaining)
                    if not n:raise IOError('Incomplete offline bake write')
                    remaining=remaining[n:]
                checks.append(dict(tensor=name,sha256=hashlib.sha256(raw).hexdigest(),device=str(device)))
                del raw,remaining,value
            del q,scale
            print('MERGED',device,target,flush=True)
        os.fsync(stream.fileno())
    store.check()
    return checks


def convert(base,lora_stack,output,prefix='',verify_native=False,devices=None,overwrite=False):
    from safetensors.torch import save_file
    reader=CheckpointReader(base);store=H3LoRAStore(tuple(lora_stack),reader)
    if not any(scale for _,_,mapping in store.sources for _,_,scale in mapping.values()):return str(reader.path)
    output=Path(output).resolve();output.parent.mkdir(parents=True,exist_ok=True)
    partial=output.with_name(output.name+'.partial')
    receipt=output.with_suffix(output.suffix+'.json')
    if not overwrite and (output.exists() or partial.exists() or receipt.exists()):raise FileExistsError('Output/partial/receipt already exists')
    active={t for _,_,mapping in store.sources for t,(_,_,scale) in mapping.items() if scale}
    targets=sorted(t for t in active if (t+'.').startswith(prefix))
    if not targets:raise ValueError('No matching LoRA targets')
    devices=tuple(devices) if devices is not None else tuple(f'cuda:{i}' for i in range(min(2,torch.cuda.device_count())))
    if not devices:raise RuntimeError('Offline LoRA baking requires a CUDA GPU')
    if len(devices)>2:raise ValueError('At most two baking devices are supported')
    selected_bytes=reader.bytes(prefix) if prefix else reader.path.stat().st_size
    if shutil.disk_usage(output.parent).free<selected_bytes+2**30:raise OSError('Insufficient free disk for new checkpoint')
    # Claim scratch exclusively, including fixture exports. Cancellation and
    # disk failures remove only scratch claimed by this invocation.
    if overwrite and partial.exists():partial.unlink(missing_ok=True)
    with partial.open('xb'):pass
    try:
        if prefix:
            _check_cancel();tensors=reader.read(prefix)
            save_file({prefix+k:v.contiguous() for k,v in tensors.items()},str(partial))
            del tensors
        else:
            with reader.path.open('rb') as source,partial.open('wb') as dest:
                while True:
                    _check_cancel();chunk=source.read(16*2**20)
                    if not chunk:break
                    dest.write(chunk)
    except BaseException:
        partial.unlink(missing_ok=True);raise
    partitions=layer_partitions(reader,targets,len(devices))
    comfy_root=os.environ.get('COMFY_ROOT')
    if not comfy_root:
        import comfy
        comfy_root=str(Path(next(iter(comfy.__path__))).parent)
    jobs=[(str(reader.path),tuple(lora_stack),str(partial),part,device,verify_native,comfy_root)
          for part,device in zip(partitions,devices) if part]
    checks=[]
    try:
        if len(jobs)==1:checks=_bake_partition(*jobs[0])
        else:
            context=mp.get_context('spawn');cancel=context.Event()
            with ProcessPoolExecutor(max_workers=len(jobs),mp_context=context,initializer=_worker_init,initargs=(cancel,)) as pool:
                try:
                    pending={pool.submit(_bake_partition,*job) for job in jobs}
                    while pending:
                        _check_cancel()
                        done,pending=wait(pending,timeout=.1,return_when=FIRST_COMPLETED)
                        for future in done:checks.extend(future.result())
                except BaseException:
                    cancel.set();raise
    except BaseException:
        # All workers have exited the context before this owned scratch is
        # removed. The base and any previously published artifact are intact.
        partial.unlink(missing_ok=True)
        raise
    store.check()
    if reader.path.stat().st_size!=reader.signature[0] or reader.path.stat().st_mtime_ns!=reader.signature[1]:
        raise RuntimeError('Base changed during offline merge')
    partial.replace(output)
    signature=output.stat()
    receipt.write_text(json.dumps(dict(format='h3-native-int8-offline-lora-v2',base=reader.identity,
        lora_identity=store.identity,stack=list(lora_stack),prefix=prefix,tensors=checks,
        verified_native_targets=len(targets) if verify_native else 0,
        native_independent_verification=bool(verify_native),
        devices=list(devices),layer_partitions=partitions,
        output_signature=[signature.st_size,signature.st_mtime_ns],
        semantics='Native FP32 sequential merge; deterministic full-width ConvRot requantization for INT8, original storage dtype for dense weights; not online bypass bit-exact'),indent=2))
    return str(output)


def cached_bake(base,lora_stack,*,before_bake=None,target_name='latest_baked_dit.safetensors'):
    """Return a complete immutable bake, or publish a new one. Never double-apply."""
    reader=CheckpointReader(base);store=H3LoRAStore(tuple(lora_stack),reader)
    if not any(scale for _,_,mapping in store.sources for _,_,scale in mapping.values()):return str(reader.path)
    output=reader.path.parent/'h3_lora_baked'/target_name
    receipt=output.with_suffix(output.suffix+'.json')
    if output.exists() and receipt.is_file():
        try:
            record=json.loads(receipt.read_text());stat=output.stat()
            if (record.get('format')=='h3-native-int8-offline-lora-v2' and record.get('lora_identity')==store.identity
                    and record.get('base')==list(reader.identity) and record.get('output_signature')==[stat.st_size,stat.st_mtime_ns]):
                store.check();return str(output)
            if (record.get('format')=='h3-native-int8-offline-lora-v2' and record.get('lora_identity')==store.identity
                    and record.get('base')==list(reader.identity) and record.get('output_signature')!=[stat.st_size,stat.st_mtime_ns]):
                raise RuntimeError('Baked checkpoint identity changed; choose a new cache destination')
        except (json.JSONDecodeError,KeyError):
            pass
    if before_bake:before_bake()
    return convert(base,lora_stack,output,overwrite=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--base',default='D:/minimaxH3/MiniMax_H3_Ref2VA_pruned_int8_convrot.safetensors')
    parser.add_argument('--lora',action='append',nargs=2,metavar=('PATH','STRENGTH'),required=True)
    parser.add_argument('--output',required=True);parser.add_argument('--prefix',default='')
    parser.add_argument('--verify-native',action='store_true',help='Verify every merged target against an independent ordinary A/B formula')
    parser.add_argument('--devices',nargs='+',help='Whole-layer workers, e.g. cuda:0 cuda:1; default uses up to two available GPUs')
    args=parser.parse_args();torch.set_grad_enabled(False);torch.backends.cuda.matmul.allow_tf32=False
    convert(args.base,[(p,float(s)) for p,s in args.lora],args.output,args.prefix,args.verify_native,args.devices)


if __name__=='__main__':main()
