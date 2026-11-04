"""Build one CPU-only Kaggle mount SHA256 readback job for eight Stage C Hop parts.

Requires independently authenticated current Dataset versions, the completed
GPU batch receipt, and original pre-upload source receipts. This module
prepares an immutable Notebook but does not submit or mutate Kaggle Datasets.
"""
import argparse
import ast
import base64
import hashlib
import json
from pathlib import Path
import re

from benchmarks.build_ptd_v2_stage_c_readback_job import VERIFIER, dataset_id

PARTS = tuple(f'hop_{h:02d}' for h in range(8))
FORMAT = 'h3-ptd-v2-stage-c-eight-hop-readback-delivery-v1'


def _digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def build(batch_result, output, *, versions):
    batch_result, output = Path(batch_result).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError('Never overwrite immutable Kaggle readback jobs')
    if set(versions) != set(PARTS) or any(type(v) is not int or v < 1 for v in versions.values()):
        raise ValueError('All eight independently authenticated Kaggle Dataset versions are required')
    batch = json.loads(batch_result.read_text(encoding='utf-8'))
    recipe = batch.get('source_recipe_sha256')
    if (batch.get('format') != 'h3-ptd-v2-stage-c-bounded-batch-v1'
            or batch.get('status') != 'upload_requests_accepted_remote_readback_pending'
            or batch.get('hops') != list(range(8))
            or set(batch.get('published', {})) != set(PARTS)
            or not isinstance(recipe, str) or not re.fullmatch('[a-f0-9]{64}', recipe)):
        raise ValueError('Batch is not the completed eight-Hop upload-ACK release')
    specs = []
    for part in PARTS:
        entry = batch['published'][part]
        handle = dataset_id(part)
        receipt_file = batch_result.parent / f'{part}_receipt.json'
        admission_file = batch_result.parent / f'{part}_local_admission.json'
        admission = json.loads(admission_file.read_text(encoding='utf-8'))
        receipt_sha = _digest(receipt_file)
        if (entry.get('dataset_id') != handle
                or entry.get('status') != 'create_request_accepted_processing_unverified'
                or entry.get('receipt_sha256') != receipt_sha
                or entry.get('payload_bytes') != 21179555374
                or admission.get('part') != part
                or admission.get('receipt_sha256') != receipt_sha
                or admission.get('recipe_sha256') != recipe
                or admission.get('payload_files') != 202
                or admission.get('payload_bytes') != entry['payload_bytes']
                or not re.fullmatch('[a-f0-9]{64}', admission.get('payload_tree_sha256', ''))):
            raise ValueError('Missing or mismatched source-bound pre-upload proof: ' + part)
        specs.append(dict(part=part, dataset=handle,
                          version_pin=f'{handle}/versions/{versions[part]}',
                          receipt_sha256=receipt_sha,
                          recipe_sha256=recipe,
                          payload_bytes=entry['payload_bytes'],
                          payload_files=202,
                          payload_tree_sha256=admission['payload_tree_sha256']))
    source = VERIFIER.read_bytes()
    source_sha = hashlib.sha256(source).hexdigest()
    encoded = base64.b64encode(source).decode('ascii')
    notebook_code = f"""
import base64,hashlib,json,pathlib,types
SPECS={specs!r}
SOURCE=base64.b64decode({encoded!r})
SOURCE_SHA256={source_sha!r}
OUT=pathlib.Path('/kaggle/working/h3_ptd_v2_stage_c_eight_hop_readback_20261103')
OUT.mkdir(parents=True,exist_ok=False)
state=dict(format='h3-ptd-v2-stage-c-eight-hop-mount-readback-v1',
    status='starting',read_only=True,parts={{}},source_verifier_sha256=SOURCE_SHA256,
    expected_parts=len(SPECS),expected_payload_bytes=sum(p['payload_bytes'] for p in SPECS),
    provider_version_authentication='performed separately before job build')
def save():
    (OUT/'result.json').write_text(json.dumps(state,indent=2)+'\\n',encoding='utf-8')
save()
try:
    if hashlib.sha256(SOURCE).hexdigest()!=SOURCE_SHA256:
        raise ValueError('Frozen independent SHA256 verifier capsule mismatch')
    module=types.ModuleType('stage_c_independent_remote_readback')
    exec(compile(SOURCE,'stage_c_independent_remote_readback.py','exec'),module.__dict__)
    mounted={{}}
    for spec in SPECS:
        handle=spec['dataset']
        slug=handle.split('/')[1]
        candidates=[
            pathlib.Path('/kaggle/input')/slug,
            pathlib.Path('/kaggle/input/datasets')/handle,
            pathlib.Path('/kaggle/input/datasets')/slug,
            pathlib.Path('/kaggle/input')/handle]
        found=[p for p in candidates if p.is_dir()]
        if len(found)!=1 or not (found[0]/'payload').is_dir():
            raise FileNotFoundError('Cannot identify exact mounted payload for '+spec['part'])
        mounted[spec['part']]=found[0]/'payload'
    state['status']='auditing'
    save()
    for spec in SPECS:
        part=spec['part']
        try:
            proof=module.verify(mounted[part],part,spec['receipt_sha256'],
                                spec['recipe_sha256'],spec['version_pin'])
            if (proof['payload_bytes']!=spec['payload_bytes']
                    or proof['payload_files']!=spec['payload_files']
                    or proof['payload_tree_sha256']!=spec['payload_tree_sha256']):
                raise ValueError('Remote immutable payload tree differs from original bake admission')
            (OUT/(part+'_readback_proof.json')).write_text(
                json.dumps(proof,indent=2)+'\\n',encoding='utf-8')
            state['parts'][part]=dict(status='passed_remote_sha256',
                dataset=spec['dataset'],version_pin=spec['version_pin'],
                payload_files=proof['payload_files'],payload_bytes=proof['payload_bytes'],
                payload_tree_sha256=proof['payload_tree_sha256'])
            print('STAGE_C_EIGHT_HOP_SHA256_PASS',part,proof['payload_bytes'],flush=True)
        except BaseException as error:
            state['parts'][part]=dict(status='failed',
                dataset=spec['dataset'],error_type=type(error).__name__)
            print('STAGE_C_EIGHT_HOP_SHA256_FAILED',part,type(error).__name__,flush=True)
        save()
    if len(state['parts'])==len(SPECS) and all(
            part.get('status')=='passed_remote_sha256' for part in state['parts'].values()):
        state['status']='passed_remote_sha256'
    else:
        state['status']='failed'
        raise RuntimeError('At least one Hop remote SHA256 failed')
except BaseException as error:
    state['status']='failed'
    state['error_type']=type(error).__name__
    raise
finally:
    save()
    print('STAGE_C_EIGHT_HOP_SHA256_FINAL',state['status'],flush=True)
"""
    ast.parse(notebook_code)
    notebook = dict(nbformat=4, nbformat_minor=5,
                    metadata=dict(kernelspec=dict(display_name='Python 3',
                                                  language='python',name='python3')),
                    cells=[dict(cell_type='code',execution_count=None,
                                metadata={},outputs=[],
                                source=notebook_code.splitlines(keepends=True))])
    output.mkdir(parents=True)
    nb = output / 'experiment.ipynb'
    nb.write_text(json.dumps(notebook),encoding='utf-8')
    metadata = dict(
        id='pamcks/h3-ptd-v2-stage-c-eight-hop-readback-20261103',
        title='H3 PTD v2 Stage C Eight Hop Remote SHA256 20261103',
        code_file='experiment.ipynb',language='python',kernel_type='notebook',
        is_private=True,enable_gpu=False,enable_tpu=False,
        enable_internet=False,docker_image_pinning_type='original',
        dataset_sources=[s['dataset'] for s in specs],
        kernel_sources=[],competition_sources=[],model_sources=[])
    (output/'kernel-metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    binding=dict(format=FORMAT,status='prepared_not_submitted',
                 notebook_sha256=hashlib.sha256(nb.read_bytes()).hexdigest(),
                 source_verifier_sha256=source_sha,
                 batch_result_sha256=_digest(batch_result),
                 parts=specs,remote_sha256='pending',cpu_only=True,
                 private_only=True,read_only=True,
                 no_status_polling=True,provider_version_authentication='observed before build')
    (output/'source_binding.json').write_text(json.dumps(binding,indent=2),encoding='utf-8')
    return binding


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch-result',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--version',type=int,required=True,
                        help='Already authenticated current Dataset version for ALL eight Hops')
    args=parser.parse_args()
    binding=build(args.batch_result,args.output,
                  versions={part:args.version for part in PARTS})
    print(json.dumps(dict(status=binding['status'],parts=len(binding['parts']),
                          notebook_sha256=binding['notebook_sha256'],
                          verifier_sha256=binding['source_verifier_sha256']),
                     indent=2),flush=True)


if __name__=='__main__':
    main()
