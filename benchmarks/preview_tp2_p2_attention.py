"""Explicit SM120 unit preview; never target-hardware performance evidence.

Clone only the wrappers used by unit tests with an explicit actual-SM120 guard
and SM120 compilation. Do not spoof torch hardware identity or modify sources.
"""
import inspect
import json
from pathlib import Path
import sys
sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parent)]
import torch
import tilelang
import tp2_p2_attention as suite

assert torch.cuda.get_device_capability()==(12,0)
original=tilelang.compile
def preview_compile(fn,**kw):
    kw['target']={'kind':'cuda','arch':'sm_120'}
    return original(fn,**kw)
tilelang.compile=preview_compile

def wrapper(module,name):
    fn=getattr(module,name);code=inspect.getsource(fn).replace('(7,5)','(12,0)').replace('Actual SM75','Explicit SM120 unit preview').replace('Physical SM75','Explicit SM120 unit preview')
    namespace=dict(fn.__globals__);exec(compile(code,__file__,'exec'),namespace)
    setattr(module,name,namespace[name])

for module,name in [(suite.candidate,'execute'),(suite.base.attn,'execute'),(suite.base.dense,'prepare'),(suite.base.prep_sparse,'prepare')]:wrapper(module,name)
torch.set_grad_enabled(False);torch.backends.cuda.matmul.allow_tf32=False
records=suite.units()
Path('D:/attention_eval/tp2_p2_20261018/local_attention_units.json').write_text(json.dumps(dict(scope='SM120 arithmetic preview only, not SM75 performance',hardware=torch.cuda.get_device_name(),records=records),indent=2))
print('P2_LOCAL_ATTENTION_UNITS',len(records),'pass')
