"""Read the sampler gate's current session and download only its evidence."""
import argparse
import json
from pathlib import Path
import requests
import threading
from kaggle.api.kaggle_api_extended import KaggleApi, ApiListKernelSessionOutputRequest

SLUG = 'chinokiki9/minimax-h3-comfyui-sampler-gate'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--download', action='store_true')
    parser.add_argument('--output',type=Path,default=Path('kaggle_comfy_sampler_output'))
    options = parser.parse_args()
    root = options.output.resolve();root.mkdir(parents=True,exist_ok=True)
    api = KaggleApi();api.authenticate()
    status = api.kernels_status(SLUG)
    print('STATUS',status.status,status.failure_message or '',flush=True)
    if 'RUNNING' in str(status.status):
        def stream():
            try:
                with (root/'live.log').open('w',encoding='utf-8') as log:
                    for event in api.kernels_logs_stream(SLUG):
                        data=event.get('data','');log.write(data);log.flush()
                        if any(s in data for s in ('SAMPLER','VERIFIED','NATIVE','TP ','Native ComfyUI','Traceback','Error:')):
                            print(data.rstrip(),flush=True)
            except Exception as error:
                print('Log stream unavailable:',type(error).__name__,flush=True)
        thread=threading.Thread(target=stream,daemon=True);thread.start();thread.join(8)
        # Completed-session output can still refer to the preceding version
        # while a new version runs. Only consume live logs during execution.
        return
    with api.build_kaggle_client() as client:
        request = ApiListKernelSessionOutputRequest()
        request.user_name,request.kernel_slug = SLUG.split('/')
        files = []
        printed_log = False
        while True:
            response = client.kernels.kernels_api_client.list_kernel_session_output(request)
            files.extend(response.files or [])
            if response.log and not printed_log:
                printed_log = True
                (root/'session.log').write_text(response.log,encoding='utf-8')
                try:
                    text=''.join(e.get('data','') for e in json.loads(response.log))
                except (ValueError,TypeError):text=response.log
                meaningful = [line for line in text.splitlines() if any(s in line for s in ('SAMPLER','VERIFIED','NATIVE','TP ','Native ComfyUI','Traceback','Error:'))]
                print('\n'.join(meaningful[-18:]),flush=True)
            if not response.next_page_token:break
            request.page_token = response.next_page_token
    if options.download:
        for item in files:
            if not item.file_name.startswith('sampler_results/') or item.file_name.endswith('.pt'):
                continue
            target = (root/item.file_name).resolve()
            if not target.is_relative_to(root):raise RuntimeError('Output path escapes evidence directory')
            try:
                response = requests.get(item.url,timeout=(10,30))
                if response.status_code!=200:raise RuntimeError('Output HTTP '+str(response.status_code))
                target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(response.content)
                print('SAVED',item.file_name,len(response.content),flush=True)
            except requests.RequestException:
                raise RuntimeError('Evidence download failed; signed URL omitted') from None


if __name__=='__main__':main()
