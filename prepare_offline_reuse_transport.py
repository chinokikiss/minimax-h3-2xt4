"""Issue private, expiring read capabilities for the verified HF revision.

Only HEAD/JSON metadata is accessed locally. All weight downloads run on Kaggle.
"""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import time
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parent


def main():
    import httpx
    from huggingface_hub import get_hf_file_metadata, hf_hub_url, set_client_factory
    set_client_factory(lambda: httpx.Client(follow_redirects=True, timeout=httpx.Timeout(60,connect=20)))
    folder = ROOT/'semantic_results/kaggle_hf_publication_v6'
    publication = json.loads((folder/'publication.json').read_text())
    audit = json.loads((folder/'publication_audit.json').read_text())
    bundle = json.loads((folder/'bundle.json').read_text())
    receipts = json.loads((folder/'upload_receipts.json').read_text())
    if not (publication['complete'] and audit['passed'] and audit['commit']==publication['commit']
            and hashlib.sha256((folder/'bundle.json').read_bytes()).hexdigest()==publication['bundle_sha256']):
        raise ValueError('Verified immutable publication required')
    def authorize(item):
        name, expected = item
        for attempt in range(3):
            try:
                metadata = get_hf_file_metadata(hf_hub_url(publication['repo'], name, revision=publication['commit']))
                break
            except httpx.TransportError as error:
                if attempt==2:raise
                print('AUTHORIZATION RETRY',type(error).__name__,flush=True)
                time.sleep(2*(attempt+1))
        query = parse_qs(urlparse(metadata.location).query)
        expires = int(query.get('Expires',['0'])[0])
        if (metadata.size != expected['bytes'] or metadata.etag != expected['sha256'] or
                urlparse(metadata.location).scheme != 'https' or expires < time.time()+2700):
            raise ValueError('Expected matching, fresh object-scoped HF download URL')
        stem = name.removesuffix('.safetensors')
        manifest = json.loads((folder/(stem+'.json')).read_text())
        return name, dict(bytes=expected['bytes'],sha256=expected['sha256'],url=metadata.location,
                         expires=expires,manifest=manifest)
    with ThreadPoolExecutor(max_workers=4) as pool:
        objects = dict(pool.map(authorize, receipts['objects'].items()))
    output = ROOT/'kaggle_direct_loading_audit/offline_download_ticket.json'
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(dict(repo=publication['repo'],revision=publication['commit'],
        bundle_sha256=publication['bundle_sha256'],bundle=bundle,objects=objects,authorized_at=time.time())))
    print('AUTHORIZED private immutable download objects',len(objects),
          'minimum validity seconds',int(min(x['expires'] for x in objects.values())-time.time()),flush=True)


if __name__ == '__main__':
    try: main()
    except Exception as error:
        print('ERROR download authorization',type(error).__name__,flush=True)
        raise SystemExit(1) from None
