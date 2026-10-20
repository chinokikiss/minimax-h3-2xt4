"""Read version-bound experiment status without printing signed artifact URLs."""
import argparse
import json
from pathlib import Path
import requests
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import ApiListKernelSessionOutputRequest


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--version',type=int,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    api=KaggleApi();api.authenticate();request=ApiListKernelSessionOutputRequest()
    request.user_name='chinokiki9';request.kernel_slug='h3-quality-first-generation'
    request.version_label=f'v{args.version}';request.page_size=200
    inventory=[];status=None
    with api.build_kaggle_client() as client:
        for _ in range(8):
            page=client.kernels.kernels_api_client.list_kernel_session_output(request)
            for item in page.files or []:
                inventory.append(item.file_name)
                if item.file_name=='generation_screen/status.json':
                    response=requests.get(item.url,timeout=(15,60));response.raise_for_status();status=response.json()
            if not page.next_page_token:break
            request.page_token=page.next_page_token
    args.output.write_text(json.dumps({'requested_version':args.version,'experiment':status,'artifact_names':inventory},indent=2))
    print(json.dumps({'version':args.version,'status':status['status'] if status else None,
        'runs':status.get('runs',[]) if status else [],'hardware':status.get('hardware',{}) if status else {},
        'blocker':status.get('blocker') if status else None},indent=2))


if __name__=='__main__':main()
