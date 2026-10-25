"""Launch only the locked local ComfyUI, retaining the old attention API."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from patches.legacy_runtime import validate_source,validate_packages,COMFY_REVISION

ROOT=Path(__file__).resolve().parents[2]


def comfy_command(comfy_root,extra=(),*,isolated_ui=False):
    """Normal UI by default; node isolation is an explicit smoke-test option."""
    command=[sys.executable,str(comfy_root/'main.py'),'--cuda-device','all',
        '--disable-auto-launch','--disable-dynamic-vram',
        '--extra-model-paths-config',str(ROOT/'h3_legacy_model_paths.yaml')]
    if isolated_ui:
        command.extend(['--disable-partner-nodes','--disable-all-custom-nodes',
            '--whitelist-custom-nodes','minimax_h3_tp2'])
    return command+list(extra)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--preview',action='store_true',help='Local one-GPU check/UI; TP2 execution still requires two GPUs and NCCL.')
    parser.add_argument('--check',action='store_true',help='Check source/dependencies/API without starting a server')
    parser.add_argument('--comfy-root',type=Path,default=ROOT/'ComfyUI')
    parser.add_argument('--isolated-ui',action='store_true',help='Smoke tests only: disable partner nodes and whitelist the TP2 custom node')
    args,extra=parser.parse_known_args();extra=extra[1:] if extra[:1]==['--'] else extra
    comfy_root=Path(validate_source(args.comfy_root))
    os.environ['COMFY_ROOT']=str(comfy_root)
    sys.path[:0]=[str(comfy_root),str(ROOT)]
    # Do not let ComfyUI's parser consume this launcher's --preview/--check.
    sys.argv=sys.argv[:1]
    import torch
    hardware=[dict(name=torch.cuda.get_device_name(i),capability=list(torch.cuda.get_device_capability(i))) for i in range(torch.cuda.device_count())]
    if not args.preview:
        if len(hardware)!=2 or not torch.distributed.is_nccl_available():
            raise RuntimeError('Physical TP2 requires two CUDA GPUs and NCCL. This host cannot run TP2; use --preview only for local checks/UI.')
        if any(x['capability']!=[7,5] for x in hardware):raise RuntimeError('This deployment profile targets two SM75 T4 GPUs')
    packages=validate_packages(preview=args.preview)
    from patches.legacy_runtime import validate_legacy_attention_api
    api=validate_legacy_attention_api()
    if not args.preview:
        from patches.h3_operator_dependencies import configure_compiler
        from patches.h3_operator_policy import H3OperatorPolicy
        configure_compiler(H3OperatorPolicy())
    print(json.dumps(dict(runtime_profile='user-conda-comfyui',comfy_revision=COMFY_REVISION,root=str(comfy_root),python=sys.executable,
        preview=args.preview,packages=packages,hardware=hardware,attention_api=api),indent=2),flush=True)
    if args.check:return
    os.environ.update(NCCL_P2P_DISABLE='1',NCCL_IB_DISABLE='1',NCCL_MIN_NCHANNELS='4',NCCL_MAX_NCHANNELS='4')
    os.environ.pop('NCCL_PROTO',None)
    # Dynamic VRAM remains disabled because TP workers own the weight window.
    argv=comfy_command(comfy_root,extra,isolated_ui=args.isolated_ui)
    raise SystemExit(subprocess.call(argv,cwd=comfy_root))


if __name__=='__main__':main()
