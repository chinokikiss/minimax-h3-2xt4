"""Bounded coordinator: Kaggle uploads all weights; local traffic is metadata.

Run with HF_TOKEN inherited in memory, never written to a configuration file.
The process exits on Kaggle failure, deadline, or verified publication.
"""
import argparse
import json
from pathlib import Path
import time


def main():
    from kaggle.api.kaggle_api_extended import KaggleApi, ApiListKernelSessionOutputRequest
    import requests
    from kaggle_lfs_transport import finalize, read
    p = argparse.ArgumentParser()
    p.add_argument('--slug', required=True); p.add_argument('--ticket', required=True)
    p.add_argument('--output', required=True); p.add_argument('--timeout', type=int, default=9000)
    a = p.parse_args(); root = Path(a.output).resolve(); root.mkdir(parents=True, exist_ok=True)
    api = KaggleApi(); api.authenticate(); deadline = time.monotonic()+a.timeout
    previous = None; network_errors = 0
    while time.monotonic() < deadline:
        try:
            status = str(api.kernels_status(a.slug).status)
            network_errors = 0
        except Exception as error:
            network_errors += 1
            print('STATUS request error', type(error).__name__, 'attempt', network_errors, flush=True)
            if network_errors >= 3: raise RuntimeError('Repeated Kaggle status failure') from None
            time.sleep(45); continue
        if status != previous:
            print('STATUS', status, flush=True); previous = status
        if any(x in status for x in ('ERROR','CANCEL','ABORT')):
            raise RuntimeError('Kaggle terminated without completed object upload')
        if 'COMPLETE' in status:
            break
        time.sleep(45)
    else: raise RuntimeError('Publication coordinator deadline exceeded')
    prefix = 'pipeline_audit_results/publication_evidence/'
    with api.build_kaggle_client() as client:
        request = ApiListKernelSessionOutputRequest()
        request.user_name, request.kernel_slug = a.slug.split('/')
        selected = []
        while True:
            response = client.kernels.kernels_api_client.list_kernel_session_output(request)
            selected.extend(f for f in response.files or []
                            if f.file_name.startswith(prefix) and f.file_name.endswith('.json'))
            if not response.next_page_token: break
            request.page_token = response.next_page_token
    if len(selected) != 203:
        raise RuntimeError('Expected 200 manifests, receipts, plan and bundle')
    total = 0
    for item in selected:
        target = (root/item.file_name.removeprefix(prefix)).resolve()
        if not target.is_relative_to(root): raise RuntimeError('Evidence path escapes root')
        # Strictly JSON evidence. Never fetch or upload safetensors here.
        with requests.get(item.url, timeout=(10,60), stream=True) as response:
            if response.status_code != 200: raise RuntimeError('Evidence download failed')
            data = bytearray()
            for chunk in response.iter_content(65536):
                data.extend(chunk)
                if len(data) > 4*2**20: raise RuntimeError('Unexpected metadata file size')
            json.loads(data); total += len(data)
            if total > 32*2**20: raise RuntimeError('Unexpected total metadata size')
            target.write_bytes(data)
    print('VERIFIED downloaded JSON evidence', len(selected), 'bytes', total, flush=True)
    finalize(a.ticket, root)


if __name__ == '__main__':
    try: main()
    except Exception as error:
        # Network exceptions can include signed URLs; do not print their text.
        print('ERROR publication coordinator', type(error).__name__, flush=True)
        raise SystemExit(1) from None
