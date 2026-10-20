"""Version-bound bounded artifact retrieval; never expose signed URLs."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import time
import requests
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import ApiListKernelSessionOutputRequest


def main():
    p=argparse.ArgumentParser();p.add_argument('kernel');p.add_argument('--version',required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--pattern',required=True);args=p.parse_args()
    api=KaggleApi();api.authenticate();request=ApiListKernelSessionOutputRequest()
    request.user_name,request.kernel_slug=args.kernel.split('/');request.version_label='v'+args.version.removeprefix('v');request.page_size=200
    args.output.mkdir(parents=True,exist_ok=True);manifest={'kernel':args.kernel,'requested_version':args.version,'files':[]}
    total=0
    with api.build_kaggle_client() as client:
        for _ in range(8):
            page=client.kernels.kernels_api_client.list_kernel_session_output(request)
            for item in page.files or []:
                if not re.search(args.pattern,item.file_name):continue
                destination=(args.output/item.file_name).resolve()
                if not destination.is_relative_to(args.output.resolve()):raise RuntimeError('Provider path leaves output directory')
                destination.parent.mkdir(parents=True,exist_ok=True)
                for attempt in range(3):
                    try:
                        size=0
                        with requests.get(item.url,stream=True,timeout=(15,60)) as response:
                            response.raise_for_status()
                            with destination.with_suffix(destination.suffix+'.partial').open('wb') as stream:
                                for chunk in response.iter_content(256*1024):
                                    size+=len(chunk)
                                    if size>200*2**20 or total+size>400*2**20:raise RuntimeError('Artifact collection exceeded bounded byte budget')
                                    stream.write(chunk)
                        destination.with_suffix(destination.suffix+'.partial').replace(destination)
                        total+=size;break
                    except requests.RequestException as error:
                        if attempt==2:raise RuntimeError('Artifact download failed: '+item.file_name+' '+type(error).__name__) from None
                        time.sleep(2)
                with destination.open('rb') as stream:sha=hashlib.file_digest(stream,'sha256').hexdigest()
                manifest['files'].append({'file':item.file_name,'bytes':size,'sha256':sha})
                print('COLLECTED',item.file_name,size,flush=True)
            token=page.next_page_token
            if not token:break
            request.page_token=token
    (args.output/'download_manifest.json').write_text(json.dumps(manifest,indent=2))


if __name__=='__main__':main()
