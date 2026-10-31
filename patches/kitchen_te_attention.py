"""Kitchen TE prefill, preserving boolean/additive masks and compact GQA.

The default uses the shared Kitchen adapter, with direct FP32 output for FP32
inference. The historical implicit native kernel remains an explicit experiment.
"""
from collections import OrderedDict
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
import weakref

import torch
from patches.file_metadata import admit_library, file_metadata


_mask_cache=OrderedDict()
LIBRARY_ENV='H3_KITCHEN_ATTENTION_LIBRARY'


def _source_paths():
    from experiments.kitchen_attention.derive import ROOT
    paths=[ROOT/name for name in ('attention.py','build.py','derive.py')]
    paths += [p for p in (ROOT/'native').rglob('*') if p.suffix in ('.cu','.cuh','.json')]
    return sorted(paths)


def _source_hashes():
    from experiments.kitchen_attention.derive import ROOT
    return {p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in _source_paths()}


def _source_metadata():
    return {str(p):file_metadata(p) for p in _source_paths()}


def validate_library(path, *, verify_hashes=True):
    path=Path(path).resolve()
    receipt=(json.loads(path.with_name('build_receipt.json').read_text()) if verify_hashes
             else admit_library(path,_source_paths()))
    if receipt['exit_code'] or receipt['target']!='sm_75':raise RuntimeError('Unqualified Kitchen SM75 build')
    if 'V12.8.93' not in receipt['compiler_version']:raise RuntimeError('Kitchen SM75 requires the qualified CUDA 12.8.93 build')
    if verify_hashes and hashlib.sha256(path.read_bytes()).hexdigest()!=receipt['artifact_sha256']:
        raise RuntimeError('Kitchen native library hash mismatch')
    sources={n.replace('\\','/'):v for n,v in receipt['sources'].items()}
    if verify_hashes and any(sources.get(n)!=v for n,v in _source_hashes().items()):
        raise RuntimeError('Kitchen native library does not match the current retained source')
    return path


def prepare_native(nvcc):
    """CPU compilation at explicit SM75 worker startup; no package installation."""
    configured=os.environ.get(LIBRARY_ENV)
    if configured:return validate_library(configured,verify_hashes=False)
    fingerprint=hashlib.sha256(json.dumps(_source_metadata(),sort_keys=True).encode()).hexdigest()[:24]
    root=Path(os.environ.get('H3_KITCHEN_ATTENTION_CACHE',str(Path(tempfile.gettempdir())/'h3-kitchen-sm75')))/fingerprint
    root.mkdir(parents=True,exist_ok=True)
    library=root/('kitchen_sm75.dll' if os.name=='nt' else 'kitchen_sm75.so')
    lock=root/'build.lock';started=time.monotonic()
    while True:
        if library.exists() and library.with_name('build_receipt.json').exists() and not lock.exists():
            result=validate_library(library,verify_hashes=False);break
        try:fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
        except FileExistsError:
            if time.monotonic()-started>330:raise TimeoutError('Kitchen native CPU build lock exceeded 330 seconds')
            time.sleep(.2);continue
        os.close(fd)
        try:
            from experiments.kitchen_attention.build import build
            result=validate_library(build(root,nvcc=nvcc,verify_hashes=False),verify_hashes=False)
        finally:lock.unlink(missing_ok=True)
        break
    os.environ[LIBRARY_ENV]=str(result)
    return result


@lru_cache(maxsize=4)
def _backend(path):
    from experiments.kitchen_attention.attention import Backend
    return Backend(validate_library(path,verify_hashes=False))


def _backend_for_device(device):
    path=os.environ.get(LIBRARY_ENV)
    if not path or torch.device(device).type!='cuda':return None
    if torch.cuda.get_device_capability(device)!=(7,5):return None
    return _backend(path)


def plain_causal_mask(mask,n,nk):
    """Recognize only the exact plain prefill triangle; never discard padding/bias.

    Normal tensors are cached by identity and mutation version, using weak
    references. Inference tensors have no version and are rechecked each time.
    """
    if mask is None or n!=nk or mask.ndim not in (2,3,4) or mask.shape[-2:]!=(n,nk):return False
    if mask.dtype!=torch.bool and not mask.is_floating_point():return False
    if mask.is_cuda and torch.cuda.is_current_stream_capturing():return False
    try:version=mask._version
    except RuntimeError:version=None
    key=id(mask)
    prior=_mask_cache.get(key)
    if version is not None and prior and prior[0]() is mask and prior[1:3]==(version,(n,nk)):
        _mask_cache.move_to_end(key);return prior[3]
    allowed=torch.arange(nk,device=mask.device)[None,:]<=torch.arange(n,device=mask.device)[:,None]
    if mask.dtype==torch.bool:correct=(mask==allowed)
    else:
        blocked=torch.isneginf(mask)|(mask==torch.finfo(mask.dtype).min/4)
        correct=torch.where(allowed,mask==0,blocked)
    result=bool(correct.all())
    if version is not None:
        _mask_cache[key]=(weakref.ref(mask),version,(n,nk),result)
        _mask_cache.move_to_end(key)
        while len(_mask_cache)>32:_mask_cache.popitem(last=False)
    return result


def attention(q,k,v,mask=None,*,use_native=False):
    from patches.kitchen_attention import int8_attention
    if not use_native:
        out=int8_attention(q,k,v,attn_mask=mask)
        return out.transpose(1,2).reshape(q.shape[0],q.shape[2],-1)
    backend=_backend_for_device(q.device)
    shape=tuple(mask.shape) if mask is not None else ()
    shape=(1,)*(4-len(shape))+shape if len(shape)<=4 else shape
    native_mask=(mask is not None and len(shape)==4 and mask.device==q.device
        and mask.dtype in (torch.bool,torch.float16,torch.bfloat16,torch.float32) and
        all(a in (1,z) for a,z in zip(shape,(q.shape[0],q.shape[1],q.shape[2],k.shape[2]))))
    if (backend is not None and native_mask and q.shape[-1] in (64,128)
            and not any(t.requires_grad for t in (q,k,v))
            and plain_causal_mask(mask,q.shape[2],k.shape[2])):
        out=backend.attention(q,k,v,is_causal=True)
    else:out=int8_attention(q,k,v,attn_mask=mask)
    return out.transpose(1,2).reshape(q.shape[0],q.shape[2],-1)
