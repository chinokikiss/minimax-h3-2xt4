"""Preserve completed v3 Native outputs in a private Kaggle input artifact."""
import hashlib
import json
from pathlib import Path
import requests
from kaggle.api.kaggle_api_extended import KaggleApi, ApiListKernelSessionOutputRequest

ROOT=Path(__file__).resolve().parent
SLUG='chinokiki9/minimax-h3-pipeline-residency-audit'
DATASET='chinokiki9/minimax-h3-native-audit-reference-v3'


def main():
    target=ROOT/'kaggle_native_audit_reference';target.mkdir(exist_ok=True)
    api=KaggleApi();api.authenticate()
    with api.build_kaggle_client() as client:
        request=ApiListKernelSessionOutputRequest()
        request.user_name,request.kernel_slug=SLUG.split('/')
        files=[]
        while True:
            response=client.kernels.kernels_api_client.list_kernel_session_output(request)
            files.extend(response.files or [])
            if not response.next_page_token:break
            request.page_token=response.next_page_token
    wanted=['pipeline_audit_results/native_result.json']+[
        'pipeline_audit_results/native_reference/'+name+'.pt'
        for name in ('conditioning','video','audio','pixels','waveform')]
    identity=dict(kernel=SLUG,version=3,dataset=DATASET,files={})
    for name in wanted:
        item=next((f for f in files if f.file_name==name),None)
        if item is None:raise RuntimeError('Completed Native output missing: '+name)
        path=target/Path(name).name
        try:
            with requests.get(item.url,stream=True,timeout=(15,45)) as response:
                if response.status_code!=200:raise RuntimeError('Reference download HTTP '+str(response.status_code))
                digest=hashlib.sha256();size=0
                with path.open('wb') as stream:
                    for chunk in response.iter_content(2**20):
                        stream.write(chunk);digest.update(chunk);size+=len(chunk)
        except requests.RequestException as error:
            raise RuntimeError('Reference download failed: '+type(error).__name__) from None
        identity['files'][path.name]=dict(bytes=size,sha256=digest.hexdigest())
        print('PRESERVED',path.name,size,flush=True)
    result=json.loads((target/'native_result.json').read_text())
    if not (result['passed'] and result['physical'] and result['layers']==50 and result['steps']==4):
        raise RuntimeError('Only completed physical Native outputs may be reused')
    for name,sha in result['source_sha256'].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=sha:
            raise RuntimeError('Native source changed: '+name)
    (target/'reference_identity.json').write_text(json.dumps(identity,indent=2))
    (target/'dataset-metadata.json').write_text(json.dumps(dict(
        id=DATASET,title='MiniMax H3 Native Audit Reference v3',
        subtitle='Private numerical reference for the full pipeline TP audit',
        description='Completed physical dual-T4 Native reference: 50-layer TE and DiT, four sampler evaluations, 512x512 and 124 frames. Source and artifact hashes are preserved. These files are numerical validation references, not standalone model weights.',
        licenses=[dict(name='other')]),indent=2))
    print('PRIVATE ARTIFACT READY',target,flush=True)


if __name__=='__main__':main()
