"""Bounded SDK live-log read; exclude signed URLs and credential fields."""
import argparse
import re
import threading
import time
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelSessionLogsStreamRequest

parser=argparse.ArgumentParser();parser.add_argument('kernel');parser.add_argument('--output',type=Path,required=True);parser.add_argument('--version');args=parser.parse_args()
api=KaggleApi();api.authenticate()
request=ApiGetKernelSessionLogsStreamRequest();request.user_name,request.kernel_slug=args.kernel.split('/')
request.wait_for_logs_url_seconds=10
if args.version:request.version_label='v'+args.version.removeprefix('v')
redact=lambda s:re.sub(r'(token|upload_id|signature)=[^&\s]+',r'\1=[redacted]',re.sub(r'https?://\S+','[URL]',s))
try:
    with api.build_kaggle_client() as client:
        response=client.kernels.kernels_api_client.get_kernel_session_logs_stream(request)
        timer=threading.Timer(20,response.close);timer.daemon=True;timer.start()
        start=time.monotonic();total=0
        try:
            with args.output.open('w',encoding='utf8') as log:
                for line in response.iter_lines(chunk_size=128,decode_unicode=True):
                    if not line:continue
                    if isinstance(line,bytes):line=line.decode('utf8',errors='replace')
                    clean=redact(line);log.write(clean+'\n');log.flush();total+=len(clean)
                    if any(s in clean for s in ['GENERATION_STAGE','GENERATION_BLOCKED','VERIFIED','DENOISE','Traceback','Error']):print(clean[:1200],flush=True)
                    if total>512000 or time.monotonic()-start>20:break
        finally:timer.cancel();response.close()
except Exception as error:print('LIVE_LOG_UNAVAILABLE',type(error).__name__,getattr(getattr(error,'response',None),'status_code',None),flush=True)
