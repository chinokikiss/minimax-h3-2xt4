"""Build CPU-only Kaggle Dataset mount SHA256 readback jobs for Stage C.

Only prepare the job after independently capturing the pre-upload receipt hash
and a provider-authenticated private Dataset version. No polling or submission.
"""
import argparse
import ast
import base64
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = '20261103'
OWNER = 'pamcks'
PARTS = ('shared',) + tuple(f'hop_{i:02d}' for i in range(8))
VERIFIER = ROOT / 'benchmarks' / 'verify_ptd_v2_stage_c_remote.py'


def dataset_id(part, *, ioab=False):
    if part not in PARTS:
        raise ValueError('Unknown Stage C part')
    if ioab and part not in ('hop_01', 'hop_02'):
        raise ValueError('Experimental IOAB Dataset exists only for Hop01/02')
    suffix = '-ioab' if ioab else ''
    return f'{OWNER}/h3-ptd-v2-{part.replace("_", "-")}{suffix}-{CAMPAIGN}'


def build(output, *, part, version, receipt_sha256, recipe_sha256, ioab=False):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError('Preserve previous immutable readback job')
    handle = dataset_id(part, ioab=ioab)
    if type(version) is not int or version < 1:
        raise ValueError('Dataset version must be positive')
    for label, value in (('receipt', receipt_sha256), ('recipe', recipe_sha256)):
        if not re.fullmatch(r'[a-f0-9]{64}', value):
            raise ValueError(label + ' source hash must be SHA256 hex')
    pin = f'{handle}/versions/{version}'
    source = VERIFIER.read_bytes()
    digest = hashlib.sha256(source).hexdigest()
    encoded = base64.b64encode(source).decode('ascii')
    slug = handle.split('/')[1]
    paths = [
        '/kaggle/input/' + slug,
        '/kaggle/input/datasets/' + handle,
        '/kaggle/input/datasets/' + slug,
        '/kaggle/input/' + handle,
    ]
    snippet = f"""
import base64,hashlib,json,pathlib,sys,traceback,types
PART={part!r}
HANDLE={handle!r}
PIN={pin!r}
PREUPLOAD_RECEIPT={receipt_sha256!r}
RECIPE_SHA256={recipe_sha256!r}
PATHS={paths!r}
VERIFY_SOURCE=base64.b64decode({encoded!r})
VERIFY_SHA256={digest!r}
OUTPUT=pathlib.Path('/kaggle/working/h3_ptd_v2_readback_{part}')
OUTPUT.mkdir(parents=True,exist_ok=False)
state=dict(format='h3-ptd-v2-stage-c-remote-readback-job-v1',
    part=PART,dataset=HANDLE,version_pin=PIN,status='starting',
    verifier_sha256=VERIFY_SHA256,read_only=True,expected_layers=50)
def save():
    (OUTPUT/'result.json').write_text(json.dumps(state,indent=2)+'\\n',encoding='utf-8')
save()
try:
    if hashlib.sha256(VERIFY_SOURCE).hexdigest()!=VERIFY_SHA256:
        raise ValueError('Embedded SHA256 verifier differs from local builder')
    mounts=[pathlib.Path(path) for path in PATHS if pathlib.Path(path).is_dir()]
    state['candidate_mounts']=[str(path) for path in mounts]
    if len(mounts)!=1:
        raise FileNotFoundError('Exactly one attached Kaggle Dataset root required')
    root=mounts[0]
    payload=root/'payload'
    if not payload.is_dir():
        raise FileNotFoundError('Kaggle tar did not expose payload subdirectory')
    state['resolved_mount']=str(root)
    module=types.ModuleType('stage_c_independent_readback')
    exec(compile(VERIFY_SOURCE,'stage_c_independent_readback.py','exec'),module.__dict__)
    record=module.verify(payload,PART,PREUPLOAD_RECEIPT,RECIPE_SHA256,PIN)
    (OUTPUT/'part_readback_proof.json').write_text(json.dumps(record,indent=2)+'\\n',encoding='utf-8')
    state.update(status='passed_remote_sha256',payload_root=str(payload),
        verified_files=record['payload_files'],verified_bytes=record['payload_bytes'],
        payload_tree_sha256=record['payload_tree_sha256'],
        version_identity_caveat='Mount bytes and operator-pinned version; provider version requires independent authentication')
except BaseException as error:
    state.update(status='failed',error_type=type(error).__name__)
    (OUTPUT/'exception_type.txt').write_text(type(error).__name__+'\\n',encoding='utf-8')
    raise
finally:
    save()
    print('STAGE_C_REMOTE_READBACK',PART,state['status'],flush=True)
"""
    ast.parse(snippet)
    notebook = dict(nbformat=4, nbformat_minor=5,
        metadata=dict(kernelspec=dict(display_name='Python 3',language='python',name='python3')),
        cells=[dict(cell_type='code',execution_count=None,metadata={},outputs=[],
                    source=snippet.splitlines(keepends=True))])
    suffix = '-ioab' if ioab else ''
    metadata = dict(id=f'{OWNER}/h3-ptd-v2-readback-{part.replace("_", "-")}{suffix}-{CAMPAIGN}',
        title=f'H3 PTD v2 Readback {part.replace("_", " ")}{suffix} {CAMPAIGN}',
        code_file='experiment.ipynb',language='python',kernel_type='notebook',
        is_private=True,enable_gpu=False,enable_tpu=False,enable_internet=False,
        docker_image_pinning_type='original',dataset_sources=[handle],
        kernel_sources=[],competition_sources=[],model_sources=[])
    output.mkdir(parents=True)
    nb=output/'experiment.ipynb'
    nb.write_text(json.dumps(notebook),encoding='utf-8')
    (output/'kernel-metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    binding=dict(format='h3-ptd-v2-stage-c-mount-readback-delivery-v1',
        status='prepared_not_submitted',part=part,private=True,read_only=True,
        experimental_ioab=ioab,attached_dataset=handle,operator_claimed_version_pin=pin,
        provider_version_authentication='required_before_submission',
        source_receipt_sha256=receipt_sha256,recipe_sha256=recipe_sha256,
        verifier_sha256=digest,notebook_sha256=hashlib.sha256(nb.read_bytes()).hexdigest())
    (output/'source_binding.json').write_text(json.dumps(binding,indent=2),encoding='utf-8')
    print(json.dumps(dict(output=str(output),part=part,
        notebook_sha256=binding['notebook_sha256'],verifier_sha256=digest)),flush=True)
    return binding


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--part',required=True)
    p.add_argument('--version',type=int,required=True)
    p.add_argument('--ioab',action='store_true',help='Opt in to experimental Hop01/02 IOAB private Dataset identities')
    p.add_argument('--receipt-sha256',required=True)
    p.add_argument('--recipe-sha256',required=True)
    args=p.parse_args()
    build(args.output,part=args.part,version=args.version,
          receipt_sha256=args.receipt_sha256,recipe_sha256=args.recipe_sha256,
          ioab=args.ioab)


if __name__=='__main__':
    main()
