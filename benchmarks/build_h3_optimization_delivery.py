"""Extend the frozen cu128 capsule only with the reviewed optimization changes."""
import argparse
import difflib
import hashlib
import io
import json
from pathlib import Path
import re
import subprocess
import tarfile
from build_kaggle_light_delivery import payload_from_distribution,write_distribution

ROOT=Path(__file__).resolve().parents[1]
DIRTY=('patches/h3_operator_policy.py','patches/comfy_tp_runtime.py','patches/comfy_sampler.py',
       'custom_nodes/minimax_h3_tp2/__init__.py')
FILES=('patches/dit_column_linear.py','patches/h3_tensor_flow.py','patches/h3_vae_operators.py',
    'patches/tp2_disk_bake.py','patches/tp_dit.py','patches/tp_qwen3vl.py',
    'patches/tilelang_fp32_rms_rope.py','patches/h3_operator_aot.py','patches/h3_operator_dependencies.py','patches/kitchen_cu128_candidate.py',
    'benchmarks/test_cu128_candidate.py','benchmarks/test_h3_optimization_batch.py',
    'benchmarks/case_cu128_dispatch.py','benchmarks/case_h3_optimization.py',
    'benchmarks/profile_bake_pipeline.py','benchmarks/run_h3_optimization_batch.py',
    'benchmarks/tp2_public_dit_assets.py')


def owned_changes(name,text):
    """Replay only this task's hunks over frozen or committed runtime sources."""
    text=text.replace('\r\n','\n')
    def replace(old,new):
        nonlocal text
        assert text.count(old)==1,(name,old[:60]);text=text.replace(old,new)
    if name.endswith('h3_operator_policy.py'):
        replace("    column_linear_backend: str = 'cuda_int8_tilelang'\n", "    column_linear_backend: str = 'cuda_int8_tilelang'\n    dit_rms_rope: str = 'tilelang_fp32'\n    aot_precompile: bool = True\n")
        replace('    def __post_init__(self):\n',"    def __post_init__(self):\n        if self.dit_rms_rope not in ('native','tilelang_fp32'):raise ValueError('Unknown DiT RMS-RoPE backend')\n")
    elif name.endswith('comfy_tp_runtime.py'):
        replace('    """Two persistent NCCL processes. One in-flight command, bounded deadline."""\n',
                '    """Two persistent NCCL processes. One in-flight command, bounded deadline."""\n    operator_aot = True\n')
        replace("            elif command in ('refine', 'denoise'):\n", "            elif command == 'prepare_dit':\n                from patches.h3_operator_aot import prepare\n                result=prepare(int(args[0]),self.rank,self.config.operators)\n                self.events.append(dict(phase='dit_aot',**result))\n                return result\n            elif command in ('refine', 'denoise'):\n")
    elif name.endswith('comfy_sampler.py'):
        replace("        self.service = service\n        flow=getattr(service.config,'tensor_flow',None)","        self.service = service\n        self._aot_shapes=set()\n        flow=getattr(service.config,'tensor_flow',None)")
        method='''
    def prepare_payload(self,payload):
        if not getattr(self.service,'operator_aot',False):return
        layout=(payload or {}).get('layout')
        if layout is None or not getattr(getattr(self.service.config,'operators',None),'aot_precompile',False):return
        rows=layout.seq_len
        if rows not in self._aot_shapes:
            self.service.call('prepare_dit',rows)
            self._aot_shapes.add(rows)
'''
        replace("        return packet(self.service.call('refine', context), context.device)\n", "        return packet(self.service.call('refine', context), context.device)\n"+method)
        replace("        options.pop('wrappers', None)\n","        options.pop('wrappers', None)\n        self.prepare_payload(kwargs.get('minimax_payload'))\n")
        replace('                return super().extra_conds(**kwargs)\n',"                out=super().extra_conds(**kwargs)\n                self.diffusion_model.prepare_payload(out['minimax_payload'].cond)\n                return out\n")
    elif name.endswith('minimax_h3_tp2/__init__.py'):
        line=next(row for row in text.splitlines(True) if row.lstrip().startswith("'column_linear_backend':"))
        assert line.rstrip().endswith('}}')
        replace(line,line.rstrip()[:-2]+",\n"+
            "                        'dit_rms_rope':(['native','tilelang_fp32'],{'default':'tilelang_fp32','tooltip':'Default: T4-qualified FP32 fused RMS-RoPE for supported packed QKV. Native remains an explicit choice and unsupported-layout/device fallback.'}),\n"+
            "                        'aot_precompile':('BOOLEAN',{'default':True,'tooltip':'Prepare the exact selected TE/DiT custom kernels before execution. Shipped Kitchen/Torch kernels are already compiled. Preparation time is separate from steady state.'})}}\n")
        replace("column_linear_backend='cuda_int8_tilelang',**retired_options):",
                "column_linear_backend='cuda_int8_tilelang',dit_rms_rope='tilelang_fp32',aot_precompile=True,**retired_options):")
        replace('fc2_quantizer=fc2_quantizer,column_linear_backend=column_linear_backend))',
                'fc2_quantizer=fc2_quantizer,column_linear_backend=column_linear_backend,\n                dit_rms_rope=dit_rms_rope,aot_precompile=aot_precompile))')
    else:raise ValueError(name)
    return text.encode('utf-8')


def main():
    p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--owned-index-patch',type=Path);args=p.parse_args()
    raw=args.base.read_bytes();digest=hashlib.sha256(raw).hexdigest();assert args.base.name==digest+'.tar.gz'
    with tarfile.open(fileobj=io.BytesIO(raw),mode='r:gz') as archive:distribution={m.name:archive.extractfile(m).read() for m in archive}
    original=payload_from_distribution(distribution);files=dict(original)
    for name in FILES:files[name]=(ROOT/name).read_bytes()
    for name in DIRTY:files[name]=owned_changes(name,original[name].decode('utf-8'))
    assert all(files[n]==v for n,v in original.items() if n not in (*FILES,*DIRTY))
    assert all(files[n]==v for n,v in original.items() if n.startswith('ComfyUI/'))
    assert not any(re.search(rb'(?:hf_[A-Za-z0-9]{25,}|gh[pousr]_[A-Za-z0-9]{30,})',v) for v in files.values())
    write_distribution(args.output,distribution,files,parent=digest)
    (args.output/'preservation.json').write_text(json.dumps(dict(parent_sha256=digest,
        frozen_comfyui_and_core_pins_preserved=True,unrelated_dirty_changes_not_copied=True,
        overlays={n:hashlib.sha256(files[n]).hexdigest() for n in (*FILES,*DIRTY)},
        defaults='Scoped VAE CUDA INT8; exact TE TileLang epilogue; all-selected-custom-op preparation; pipelined bake',
        fp32_rope='Qualified physical SM75 default, with explicit Native and unsupported-contract fallbacks'),indent=2))
    if args.owned_index_patch:
        changes=[]
        for name in DIRTY:
            head=subprocess.check_output(['git','show','HEAD:'+name],cwd=ROOT).decode('utf-8').replace('\r\n','\n')
            edited=owned_changes(name,head).decode('utf-8')
            changes.extend(difflib.unified_diff(head.splitlines(True),edited.splitlines(True),fromfile='a/'+name,tofile='b/'+name))
        args.owned_index_patch.write_bytes(''.join(changes).encode('utf-8'))


if __name__=='__main__':main()
