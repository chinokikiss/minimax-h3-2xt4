"""Deployment source/ABI lock for the qualified local ComfyUI implementation."""
import hashlib
import json
from importlib.metadata import version,PackageNotFoundError
from pathlib import Path

COMFY_REVISION='d49e888586dd8ae012c0667b33466b815fee07f7'
# UTF-8 text with universal newlines: identical for Windows and Linux checkouts.
SOURCE_HASHES={
    'comfy/ldm/minimax/vae.py':'8a889ea28fc24b13edfbdea9e85b8a0762b2e7e079bbe34db026c6e6d4f5a556',
    'comfy/ldm/minimax/model.py':'15f0395cf0dbe6e5c13fd859ff359eacc7761b4e15f60aa0038853985d0e599b',
    'comfy/text_encoders/minimax.py':'0e16e44100d6db2bdfd1292eafd1b2f8d71147c546f49afe292fe4d2cdcefe8a',
    'comfy/ops.py':'bf063b4a1b6517e017e3e73e8948caa494177025d71366ae712aaca73f6b9dff',
    'comfy/sd.py':'d88d792a1a901a89e5446f13b4efd866463bc8c66a27480f19401a722558213f',
}
RUNTIME_LOCK=Path(__file__).resolve().parents[1]/'runtime/local_conda.json'


def runtime_manifest():
    import os
    path=Path(os.environ.get('H3_RUNTIME_LOCK',str(RUNTIME_LOCK)))
    if not path.is_file():raise RuntimeError('Missing frozen conda runtime manifest: '+str(path)+'. Do not install latest dependencies.')
    return json.loads(path.read_text(encoding='utf-8'))


def validate_source(root):
    root=Path(root).resolve();mismatch=[]
    for name,expected in SOURCE_HASHES.items():
        path=root/name
        actual=hashlib.sha256(path.read_text(encoding='utf-8').encode()).hexdigest() if path.is_file() else 'missing'
        if actual!=expected:mismatch.append(name)
    if mismatch:
        raise RuntimeError('H3 TP2 uses the local ComfyUI '+COMFY_REVISION+'. Source mismatch: '+', '.join(mismatch)+
            '. Launch with launch_h3_legacy.py; do not mix ComfyUI-master modules into this runtime.')
    return str(root)


def validate_imported_source():
    import comfy
    import sys
    root=Path(validate_source(Path(next(iter(comfy.__path__))).parent))
    for name in SOURCE_HASHES:
        module=sys.modules.get(name.removesuffix('.py').replace('/','.'))
        loaded=getattr(module,'__file__',None)
        if loaded and Path(loaded).resolve()!=root/name:
            raise RuntimeError('Mixed ComfyUI modules: '+str(loaded)+'. Restart with launch_h3_legacy.py.')
    return str(root)


def validate_packages(*,preview=False):
    # Platform differences block setup rather than silently changing versions.
    expected=runtime_manifest()['packages']
    found={}
    for name,required in expected.items():
        try:found[name]=version(name)
        except PackageNotFoundError:found[name]='missing'
    errors=[f'{n}=={v} (found {found[n]})' for n,v in expected.items() if found[n]!=v]
    if errors:raise RuntimeError('Frozen conda runtime dependency mismatch: '+', '.join(errors)+'. Use the existing conda comfyui Python; do not upgrade or downgrade packages automatically.')
    return found


def validate_legacy_attention_api():
    import inspect
    from comfy.ldm.minimax import vae
    signature=inspect.signature(vae.Attention.forward)
    if not {'pre_norm','residual','residual_scale'}.issubset(signature.parameters):
        raise RuntimeError('Expected the legacy fused Video VAE attention API, found '+str(signature))
    if not getattr(vae,'COMFY_KITCHEN_INT8_ATTENTION_IS_AVAILABLE',False):
        raise RuntimeError('The legacy Video VAE requires available Kitchen INT8 attention')
    return dict(signature=str(signature),native_source=vae.__file__,kitchen_int8_available=True)
