"""Build a fresh-CPU Kaggle mounted readback for the Stage C tar layout probe.

Requires the private synthetic Dataset to finish processing first. No status
polling and no H3 model downloads. Never claim pass from create acknowledgment.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path

DATASET = 'pamcks/h3-ptd-v2-c0-tar-probe-20261103'
SLUG = 'h3-ptd-v2-c0-tar-probe-20261103'
WORKER = r'''
import hashlib,json,pathlib,traceback
ROOT=pathlib.Path('/kaggle/input/h3-ptd-v2-c0-tar-probe-20261103')
OUT=pathlib.Path('/kaggle/working/h3_ptd_v2_c0_tar_readback')
OUT.mkdir(parents=True,exist_ok=False)
state=dict(format='h3-ptd-v2-c0-tar-readback-v3',
    status='starting',source='pamcks/h3-ptd-v2-c0-tar-probe-20261103',
    expected_payload_files=203,synthetic_only=True,read_only=True)
def save(): (OUT/'result.json').write_text(json.dumps(state,indent=2),encoding='utf-8')
save()
try:
    mount_root=pathlib.Path('/kaggle/input')
    state['mounted_slugs']=[p.name for p in sorted(mount_root.iterdir()) if p.is_dir()][:25] if mount_root.is_dir() else []
    nested_root=mount_root/'datasets'
    state['nested_owners']=[p.name for p in sorted(nested_root.iterdir()) if p.is_dir()][:25] if nested_root.is_dir() else []
    owner_root=nested_root/'pamcks'
    state['nested_owner_datasets']=[p.name for p in sorted(owner_root.iterdir()) if p.is_dir()][:25] if owner_root.is_dir() else []
    # Kaggle also mounts attached datasets below /kaggle/input/datasets/OWNER/SLUG.
    paths=[mount_root/'h3-ptd-v2-c0-tar-probe-20261103',
        nested_root/'pamcks'/'h3-ptd-v2-c0-tar-probe-20261103',
        nested_root/'h3-ptd-v2-c0-tar-probe-20261103',
        mount_root/'pamcks'/'h3-ptd-v2-c0-tar-probe-20261103']
    found=[p for p in paths if p.is_dir()]
    state['matching_mounts']=[str(p) for p in found]
    if len(found)!=1:
        raise FileNotFoundError('Expected exactly one mounted private Dataset root; found '+str(len(found)))
    ROOT=found[0]
    state['resolved_mount']=str(ROOT)
    state['top_level']=[p.name for p in sorted(ROOT.iterdir())][:100]
    receipts=list(ROOT.rglob('receipt.json'))
    state['receipt_candidates']=[p.relative_to(ROOT).as_posix() for p in receipts]
    if len(receipts)!=1 or receipts[0].parent.name!='payload':
        state['status']='not_directly_mountable'
        raise RuntimeError('Kaggle tar did not expose direct payload/receipt.json layout')
    payload=receipts[0].parent
    receipt=json.loads(receipts[0].read_text(encoding='utf-8'))
    if receipt.get('format')!='synthetic-directory-probe':
        raise ValueError('Unexpected C0 probe receipt format')
    expected=receipt.get('files')
    if not isinstance(expected,dict) or len(expected)!=202:
        raise ValueError('Synthetic receipt inventory incomplete')
    files={p.name:p for p in payload.iterdir() if p.is_file() and p.name!='receipt.json'}
    if set(files)!=set(expected) or len(list(payload.iterdir()))!=203:
        raise ValueError('Mounted payload inventory mismatch')
    total_bytes=0
    for name,path in sorted(files.items()):
        with path.open('rb') as stream:
            digest=hashlib.file_digest(stream,'sha256').hexdigest()
        if digest!=expected[name]:
            raise ValueError('Mounted synthetic file SHA256 mismatch: '+name)
        total_bytes+=path.stat().st_size
    state.update(status='passed_direct_payload_mount',
        payload_root=str(payload),
        receipt_sha256=hashlib.sha256(receipts[0].read_bytes()).hexdigest(),
        verified_payload_files=len(files),payload_bytes=total_bytes,
        note='Verifies synthetic mounted tar layout and SHA256, not large upload reliability')
except BaseException as error:
    if state['status']=='starting': state['status']='failed'
    state['error_type']=type(error).__name__
    (OUT/'exception_type.txt').write_text(type(error).__name__+'\n',encoding='utf-8')
    raise
finally:
    save()
    print('PTD_V2_C0_READBACK',state['status'],flush=True)
'''

def build(output):
    output=Path(output).resolve()
    if output.exists(): raise FileExistsError('Preserve C0 readback job')
    ast.parse(WORKER)
    output.mkdir(parents=True)
    notebook=dict(nbformat=4,nbformat_minor=5,
        metadata=dict(kernelspec=dict(display_name='Python 3',language='python',name='python3')),
        cells=[dict(cell_type='code',execution_count=None,metadata={},outputs=[],
            source=WORKER.splitlines(keepends=True))])
    nb=output/'experiment.ipynb'
    nb.write_text(json.dumps(notebook),encoding='utf-8')
    kernel_id='pamcks/h3-ptd-v2-c0-tar-readback-20261103'
    meta=dict(id=kernel_id,title='H3 PTD v2 C0 Tar Readback 20261103',
        code_file='experiment.ipynb',language='python',kernel_type='notebook',
        is_private=True,enable_gpu=False,enable_tpu=False,enable_internet=False,
        docker_image_pinning_type='original',
        dataset_sources=[DATASET],kernel_sources=[],competition_sources=[],model_sources=[])
    (output/'kernel-metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    (output/'source_binding.json').write_text(json.dumps(dict(
        format='h3-ptd-v2-c0-tar-readback-delivery-v1',
        synthetic_only=True,read_only=True,
        depends_on_created_dataset=DATASET,
        notebook_sha256=hashlib.sha256(nb.read_bytes()).hexdigest(),
        status='prepared_not_submitted'),indent=2),encoding='utf-8')
    print(json.dumps(dict(output=str(output),kernel_id=kernel_id,
        notebook_sha256=hashlib.sha256(nb.read_bytes()).hexdigest())),flush=True)
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();build(a.output)
if __name__=='__main__':main()
