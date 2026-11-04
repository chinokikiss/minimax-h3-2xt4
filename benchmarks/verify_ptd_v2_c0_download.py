"""Verify a Kaggle-downloaded synthetic Stage C directory probe ZIP offline.

This is a provider download readback, not a mounted NFS readback. It does not
query Kaggle or manufacture a publication version pin.
"""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile


def verify(archive, expected_receipt_sha256):
    archive=Path(archive)
    with zipfile.ZipFile(archive) as container:
        names=container.namelist()
        if len(names)!=len(set(names)) or len(names)!=204:
            raise ValueError('Expected 204 unique Kaggle archive members')
        if any(name.startswith('/') or '\\' in name or
               any(part in ('.','..','') for part in name.split('/'))
               for name in names):
            raise ValueError('Unsafe Kaggle ZIP path')
        receipt=container.read('payload/receipt.json')
        actual=hashlib.sha256(receipt).hexdigest()
        if actual!=expected_receipt_sha256:
            raise ValueError('Remote receipt differs from source C0 receipt')
        record=json.loads(receipt)
        if record.get('format')!='synthetic-directory-probe':
            raise ValueError('Unexpected C0 receipt')
        expected=record['files']
        if len(expected)!=202 or set(names)!=(set('payload/'+x for x in expected) |
                                               {'payload/receipt.json','README.txt'}):
            raise ValueError('Remote archive missing/extraneous files')
        size=0
        for name,digest in expected.items():
            h=hashlib.sha256()
            with container.open('payload/'+name) as stream:
                for block in iter(lambda: stream.read(2**20),b''):
                    h.update(block)
                    size+=len(block)
            if h.hexdigest()!=digest:
                raise ValueError('Remote synthetic file mismatch: '+name)
    if size!=26476544:
        raise ValueError('Remote synthetic payload size mismatch')
    return dict(status='verified_provider_download_not_mount',
                format='h3-ptd-v2-c0-downloaded-zip-readback-v1',
                payload_files=203,payload_bytes=size,
                archive_files=len(names),receipt_sha256=actual,
                archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
                note='Tar directory retained as payload/ in downloaded Dataset; GPU/NFS mount remains unqualified')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive',required=True,type=Path)
    p.add_argument('--receipt-sha256',required=True)
    p.add_argument('--output',required=True,type=Path)
    args=p.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    value=verify(args.archive,args.receipt_sha256)
    args.output.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(value,indent=2))


if __name__=='__main__':main()
