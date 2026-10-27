"""Deliver exact public references to private benchmark notebooks, without auth."""
import base64
import hashlib
from pathlib import Path
import re


def private_reference_prefix(rows, source, destination):
    payload = {}
    for row in rows:
        name = row['file']
        if Path(name).name != name or name in payload:
            raise ValueError('Reference filenames must be unique basenames')
        raw = (Path(source) / name).read_bytes()
        if len(raw) != row['bytes'] or hashlib.sha256(raw).hexdigest() != row['sha256']:
            raise ValueError('Reference bytes do not match the frozen input contract')
        payload[name] = base64.b85encode(raw).decode('ascii')
    # Build code from structured literals; never substitute inside encoded data.
    return (
        'import base64,hashlib,pathlib\n'
        f'PRIVATE_REFERENCES={payload!r}\n'
        f'REFERENCE_ROWS={rows!r}\n'
        f'refdir=pathlib.Path({str(destination)!r});refdir.mkdir(parents=True,exist_ok=True)\n'
        "for row in REFERENCE_ROWS:\n"
        " raw=base64.b85decode(PRIVATE_REFERENCES.pop(row['file']))\n"
        " assert len(raw)==row['bytes'] and hashlib.sha256(raw).hexdigest()==row['sha256']\n"
        " (refdir/row['file']).write_bytes(raw)\n"
        'del PRIVATE_REFERENCES,raw\n'
    )


def mirror_reference_prefix(rows, urls, destination):
    names = [row['file'] for row in rows]
    if len(names) != len(set(names)) or any(Path(name).name != name for name in names):
        raise ValueError('Reference filenames must be unique basenames')
    if set(urls) != set(names) or any(not re.fullmatch(
            r'https://raw\.githubusercontent\.com/[^/]+/[^/]+/[0-9a-f]{40}/[^?#]+', url)
            for url in urls.values()):
        raise ValueError('Reference mirrors require immutable raw GitHub commit URLs')
    return (
        'import hashlib,pathlib,urllib.request\n'
        f'REFERENCE_ROWS={rows!r}\n'
        f'REFERENCE_MIRRORS={urls!r}\n'
        f'refdir=pathlib.Path({str(destination)!r});refdir.mkdir(parents=True,exist_ok=True)\n'
        'for row in REFERENCE_ROWS:\n'
        " with urllib.request.urlopen(REFERENCE_MIRRORS[row['file']],timeout=90) as response:\n"
        "  raw=response.read(row['bytes']+1)\n"
        " if len(raw)!=row['bytes'] or hashlib.sha256(raw).hexdigest()!=row['sha256']:\n"
        "  raise ValueError('Mirrored reference does not match original bytes: '+row['file'])\n"
        " (refdir/row['file']).write_bytes(raw)\n"
        'del raw\n'
    )
