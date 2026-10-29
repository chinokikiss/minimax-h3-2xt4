"""Overlay the optional PTD adaptation on an immutable validated source capsule.

Only explicitly owned hooks are replayed over mixed-worktree files. Native
ComfyUI and dependency lock bytes must remain identical to the parent.
"""
import argparse
import difflib
import hashlib
import io
import json
from pathlib import Path
import re
import subprocess
import tarfile
from build_kaggle_light_delivery import payload_from_distribution, write_distribution

ROOT=Path(__file__).resolve().parents[1]
HOOKS=('patches/comfy_sampler.py','patches/comfy_tp_runtime.py','custom_nodes/minimax_h3_tp2/__init__.py')
FILES=('patches/h3_operator_aot.py','benchmarks/case_h3_optimization.py',
       'patches/weight_residency.py','patches/host_weight_cache.py',
       'benchmarks/test_h3_optimization_batch.py','benchmarks/test_ptd_tp2.py',
       'benchmarks/test_ptd_queue.py','benchmarks/ptd_worker_probe.py',
       'benchmarks/test_ptd_cpu_schedule.py',
       'benchmarks/ptd_t4_assets.py','benchmarks/run_ptd_tp2_t4.py',
       'custom_nodes/minimax_h3_tp2/ptd_nodes.py')


def owned_hooks(name,raw):
    text=raw.decode('utf-8').replace('\r\n','\n')
    def replace(old,new):
        nonlocal text
        if text.count(old)!=1: raise ValueError('PTD hook anchor mismatch: '+name+' '+old[:55])
        text=text.replace(old,new)
    if name=='patches/comfy_sampler.py':
        replace("def create_model(service, device='cuda:0'):","def create_model(service, device='cuda:0', *, adapter=None):")
        replace('    if flow is not None and flow.enabled:\n        class ScaledConditioningH3',
                '    if (flow is not None and flow.enabled) or adapter is not None:\n        class ScaledConditioningH3')
        replace('                out=super().extra_conds(**kwargs)\n',
                '                out=super().extra_conds(**kwargs)\n                if adapter is not None:out=adapter.conditions(out,kwargs)\n')
        replace('    native.diffusion_model = RemoteDenoiser(service)\n',
                '    native.diffusion_model = RemoteDenoiser(service) if adapter is None else adapter.denoiser(service)\n')
        replace("            service.call('begin_sample')\n            try:\n                return executor(*args, **kwargs)\n            finally:\n                if not service.closed:\n                    service.call('end_sample')\n",
                "            if adapter is not None:adapter.begin()\n            try:\n                service.call('begin_sample')\n                try:\n                    return executor(*args, **kwargs)\n                finally:\n                    if not service.closed:service.call('end_sample')\n            finally:\n                if adapter is not None and not service.closed:adapter.end()\n")
    elif name=='patches/comfy_tp_runtime.py':
        replace('        with torch.cuda.device(self.device), torch.inference_mode():\n',
                "        with torch.cuda.device(self.device), torch.inference_mode():\n            if command.startswith('ptd_'):\n                from patches.ptd.runtime import control\n                return control(self,command,args)\n            ptd_adapter=getattr(self,'_ptd_adapter',None)\n            if command=='denoise' and ptd_adapter is not None:\n                args,kwargs=ptd_adapter.prepare(args,kwargs)\n")
        replace("            if not command.startswith('idle_'):torch.cuda.synchronize(rank)\n",
                "            ptd = getattr(runtime, '_ptd_adapter', None)\n            queued_reply = (ptd is not None and ptd.options.hop_handoff == 'queue'\n                and command in ('ptd_configure','ptd_hop','ptd_state','denoise'))\n            if not command.startswith('idle_') and not queued_reply:torch.cuda.synchronize(rank)\n")
        replace("            elif command == 'end_sample':\n                self._drop_model()\n",
                "            elif command == 'end_sample':\n                if ptd_adapter is not None and ptd_adapter.queue is not None:\n                    ptd_adapter.queue.finish()\n                self._drop_model()\n")
    else:
        text+='\nfrom patches.ptd.nodes import NODE_CLASS_MAPPINGS as PTD_NODE_CLASS_MAPPINGS\nNODE_CLASS_MAPPINGS.update(PTD_NODE_CLASS_MAPPINGS)\n'
    return text.encode('utf-8')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--base',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--owned-index-patch',type=Path)
    args=parser.parse_args();raw=args.base.read_bytes();digest=hashlib.sha256(raw).hexdigest()
    assert args.base.name==digest+'.tar.gz'
    with tarfile.open(fileobj=io.BytesIO(raw),mode='r:gz') as archive:
        distribution={m.name:archive.extractfile(m).read() for m in archive}
    frozen={n:hashlib.sha256(v).hexdigest() for n,v in distribution.items()
            if n.startswith('frozen/') or n=='direct_core_requirements.json'}
    assert 'frozen/comfy_source.tar.gz.bin' in frozen
    original=payload_from_distribution(distribution);files=dict(original)
    additions=(*FILES,*(str(path.relative_to(ROOT)).replace('\\','/')
                       for path in (ROOT/'patches/ptd').iterdir() if path.is_file()))
    for name in additions:files[name]=(ROOT/name).read_bytes()
    for name in HOOKS:files[name]=owned_hooks(name,original[name])
    assert all(files[n]==v for n,v in original.items() if n not in (*additions,*HOOKS))
    assert not any(re.search(rb'(?:hf_[A-Za-z0-9]{25,}|gh[pousr]_[A-Za-z0-9]{30,})',v) for v in files.values())
    write_distribution(args.output,distribution,files,parent=digest)
    assert all(hashlib.sha256(distribution[n]).hexdigest()==value for n,value in frozen.items())
    (args.output/'preservation.json').write_text(json.dumps(dict(parent_sha256=digest,
        native_comfyui_preserved=True,core_pins_preserved=True,unrelated_dirty_changes_excluded=True,
        frozen_archive_and_lock_sha256=frozen,
        ptd_enabled_by_default=False,weights_downloaded_locally=False,
        overlays={n:hashlib.sha256(files[n]).hexdigest() for n in (*additions,*HOOKS)}),indent=2),encoding='utf-8')
    if args.owned_index_patch:
        changes=[]
        for name in HOOKS:
            head=subprocess.check_output(['git','show','HEAD:'+name],cwd=ROOT)
            old=head.replace(b'\r\n',b'\n');new=owned_hooks(name,head)
            changes.extend(difflib.diff_bytes(difflib.unified_diff,old.splitlines(True),new.splitlines(True),
                                            fromfile=('a/'+name).encode(),tofile=('b/'+name).encode()))
        args.owned_index_patch.write_bytes(b''.join(changes))


if __name__=='__main__':main()
