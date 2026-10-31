"""Build a current-source overlay that is closed over changed internal imports.

The Kaggle baseline is an immutable publication commit. A seed module may safely
reuse a baseline dependency only when the current dependency is byte-identical
to that commit. Changed or newly-added internal dependencies are recursively
added to the overlay. This prevents mixing a current caller with an older API
from the frozen capsule.
"""
from __future__ import annotations

import ast
from functools import lru_cache
import hashlib
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[1]
_INTERNAL=('patches','benchmarks','experiments','custom_nodes')

# h3_operator_aot imports these by string via import_module().
DYNAMIC_MODULES=(
    'patches/tilelang_column_epilogue.py',
    'patches/tilelang_row_quant.py',
    'patches/tilelang_swiglu_quant.py',
    'patches/tilelang_transport_tradeoffs.py',
    'patches/tilelang_transport_p2.py',
    'patches/tilelang_fp32_rms_rope.py',
    'patches/tilelang_dit_fusion.py',
)


@lru_cache(maxsize=None)
def baseline_blob(commit,name):
    process=subprocess.run(['git','show',f'{commit}:{name}'],cwd=ROOT,
        stdout=subprocess.PIPE,stderr=subprocess.DEVNULL)
    return None if process.returncode else process.stdout


def changed_from_baseline(commit,name):
    path=ROOT/name
    if not path.is_file():
        raise FileNotFoundError(path)
    old=baseline_blob(commit,name)
    return old is None or hashlib.sha256(path.read_bytes()).digest()!=hashlib.sha256(old).digest()


def _module_candidate(parts):
    if not parts or parts[0] not in _INTERNAL:
        return ()
    base='/'.join(parts)
    result=[]
    for candidate in (base+'.py',base+'/__init__.py'):
        if (ROOT/candidate).is_file():result.append(candidate)
    return result


def _resolved_module_parts(owner,node):
    package=Path(owner).with_suffix('').parts[:-1]
    if node.level:
        keep=len(package)-(node.level-1)
        if keep<0:return ()
        parts=list(package[:keep])
    else:
        parts=[]
    if node.module:parts.extend(node.module.split('.'))
    return tuple(parts)


def internal_imports(name):
    path=ROOT/name
    if path.suffix.lower()!='.py':return set()
    tree=ast.parse(path.read_text(encoding='utf-8'),filename=str(path))
    result=set()
    for node in ast.walk(tree):
        if isinstance(node,ast.Import):
            for alias in node.names:
                result.update(_module_candidate(tuple(alias.name.split('.'))))
        elif isinstance(node,ast.ImportFrom):
            parts=_resolved_module_parts(name,node)
            result.update(_module_candidate(parts))
            for alias in node.names:
                if alias.name!='*':
                    result.update(_module_candidate((*parts,*alias.name.split('.'))))
    return result


def changed_dependency_closure(commit,seeds):
    closure=set(seeds)
    queue=list(seeds)
    # Dynamic AOT modules are dependencies of h3_operator_aot even though they
    # are intentionally represented as strings.
    if 'patches/h3_operator_aot.py' in closure:
        for name in DYNAMIC_MODULES:
            if changed_from_baseline(commit,name):
                closure.add(name);queue.append(name)
    while queue:
        owner=queue.pop()
        for dependency in internal_imports(owner):
            if dependency in closure:continue
            if changed_from_baseline(commit,dependency):
                closure.add(dependency);queue.append(dependency)
    return closure


def overlay_files(commit,seeds):
    closure=changed_dependency_closure(commit,seeds)
    names=set(closure)
    # Source/data trees used by source-compiled adapters are not discoverable
    # through Python AST imports.
    trees=['patches/sol_attention_native']
    if 'patches/kitchen_attention.py' in closure:
        trees.append('patches/kitchen_attention_native')
    if any(name.startswith('experiments/kitchen_attention/') or
           name=='patches/kitchen_te_attention.py' for name in closure):
        # Python modules under experiments.kitchen_attention are already covered
        # by the recursive import closure. Only the native/vendor subtree is a
        # runtime file dependency that AST import scanning cannot discover.
        trees.append('experiments/kitchen_attention/native')
    for root in trees:
        # These are curated source/vendor trees. Carry every regular file so
        # headers, manifests, licenses and notices stay source-locked together;
        # exclude only interpreter cache artifacts.
        names.update(p.relative_to(ROOT).as_posix()
            for p in (ROOT/root).rglob('*')
            if p.is_file() and '__pycache__' not in p.parts)
    return tuple(sorted(names)),tuple(sorted(closure))


def assert_closed(commit,overlay,seeds):
    overlay=set(overlay)
    required=changed_dependency_closure(commit,seeds)
    missing=sorted(required-overlay)
    if missing:
        raise RuntimeError('Current-source overlay dependency closure is incomplete: '+', '.join(missing))
    return tuple(sorted(required))
