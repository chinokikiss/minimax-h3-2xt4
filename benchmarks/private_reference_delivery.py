"""Stage exact public reference bytes inside a private notebook, without auth."""
import base64
import hashlib
from pathlib import Path


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
