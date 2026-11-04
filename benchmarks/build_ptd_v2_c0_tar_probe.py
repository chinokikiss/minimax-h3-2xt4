"""Build a small private Kaggle Stage C directory-publication probe.

This writes a new CPU notebook only. The notebook creates a private synthetic
Dataset, without model weights or credentials in source, and does not poll.
Its final request acknowledgment is NOT a remote readback qualification.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path

WORKER = r'''
import hashlib,json,os,pathlib,time,traceback
ROOT=pathlib.Path('/kaggle/temp/h3_ptd_v2_c0_tar_probe')
OUT=pathlib.Path('/kaggle/working/h3_ptd_v2_c0_tar_probe')
DATASET_ID='pamcks/h3-ptd-v2-c0-tar-probe-20261103'
OUT.mkdir(parents=True,exist_ok=False)
state=dict(format='h3-ptd-v2-c0-tar-probe-v1',
    status='starting',private=True,model_weights=False,
    dataset_id=DATASET_ID,operation='create_once_no_status_polling',
    payload_files=0,payload_bytes=0,file_sha256={})
def save():
    (OUT/'result.json').write_text(json.dumps(state,indent=2),encoding='utf-8')
save()
try:
    from kaggle.api.kaggle_api_extended import KaggleApi
    api=KaggleApi()
    api.authenticate()
    ROOT.mkdir(parents=True,exist_ok=False)
    payload=ROOT/'payload'
    payload.mkdir()
    # 202 files + a receipt mimic the actual part directory shape.
    for i in range(202):
        path=payload/f'dit.block{i:03d}.synthetic'
        h=hashlib.sha256()
        with path.open('wb') as stream:
            for block in range(2):
                data=hashlib.sha256(f'PTD-C0-{i}-{block}'.encode()).digest()*2048
                stream.write(data);h.update(data)
        state['file_sha256'][path.name]=h.hexdigest()
        state['payload_bytes']+=path.stat().st_size
    receipt=payload/'receipt.json'
    receipt.write_text(json.dumps(dict(format='synthetic-directory-probe',
        files=state['file_sha256']),sort_keys=True),encoding='utf-8')
    state['receipt_sha256']=hashlib.sha256(receipt.read_bytes()).hexdigest()
    state['payload_files']=203
    (ROOT/'dataset-metadata.json').write_text(json.dumps(dict(
        id=DATASET_ID,title='H3 PTD v2 C0 private tar directory probe',
        licenses=[dict(name='CC0-1.0')])),encoding='utf-8')
    (ROOT/'README.txt').write_text(
        'Synthetic upload-layout test only. No MiniMax weights or personal data.\n',
        encoding='utf-8')
    state['status']='uploading_private_tar'
    save()
    started=time.monotonic()
    response=api.dataset_create_new(str(ROOT),public=False,quiet=True,
        convert_to_csv=False,dir_mode='tar')
    state['upload_seconds']=time.monotonic()-started
    state['api_status']=getattr(response,'status',None)
    state['api_error_type']='present' if getattr(response,'error',None) else 'none'
    state['dataset_url']=getattr(response,'url',None)
    if str(state['api_status']).lower()!='ok' or getattr(response,'error',None):
        raise RuntimeError('Kaggle private Dataset create request rejected')
    state['status']='request_accepted_remote_readback_pending'
    state['note']='Processing/version and mounted payload layout are NOT verified by create acknowledgement'
    print('PTD_V2_C0_PRIVATE_CREATE_ACK',DATASET_ID,flush=True)
except BaseException as error:
    state['status']='failed'
    state['error_type']=type(error).__name__
    (OUT/'exception.txt').write_text(traceback.format_exc(),encoding='utf-8')
    raise
finally:
    save()
'''
def build(output, *, kernel_id='pamcks/h3-ptd-v2-c0-tar-probe-20261103'):
    output=Path(output).resolve()
    if output.exists():
        raise FileExistsError('Refusing to overwrite Stage C C0 probe')
    if kernel_id.split('/')[0]!='pamcks':
        raise ValueError('Unexpected Kaggle owner')
    ast.parse(WORKER)
    output.mkdir(parents=True)
    notebook=dict(nbformat=4,nbformat_minor=5,metadata=dict(
        kernelspec=dict(display_name='Python 3',language='python',name='python3'),
        language_info=dict(name='python')),
        cells=[dict(cell_type='code',execution_count=None,metadata={},
                    outputs=[],source=WORKER.splitlines(keepends=True))])
    nb=output/'experiment.ipynb'
    nb.write_text(json.dumps(notebook),encoding='utf-8')
    metadata=dict(id=kernel_id,title='H3 PTD v2 C0 Tar Probe 20261103',
        code_file='experiment.ipynb',language='python',kernel_type='notebook',
        is_private=True,enable_gpu=False,enable_tpu=False,enable_internet=True,
        docker_image_pinning_type='original',dataset_sources=[],
        kernel_sources=[],competition_sources=[],model_sources=[])
    (output/'kernel-metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    (output/'source_binding.json').write_text(json.dumps(dict(
        format='h3-ptd-v2-c0-tar-probe-delivery-v1',
        model_weights=False, private=True, dataset_create_only=True,
        no_status_polling=True, kernel_id=kernel_id,
        notebook_sha256=hashlib.sha256(nb.read_bytes()).hexdigest()),indent=2),encoding='utf-8')
    print(json.dumps(dict(output=str(output),kernel_id=kernel_id,
        notebook_sha256=hashlib.sha256(nb.read_bytes()).hexdigest())))
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    build(args.output)
if __name__=='__main__':
    main()
