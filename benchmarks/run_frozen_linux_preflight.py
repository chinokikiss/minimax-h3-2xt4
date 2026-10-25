"""CPU-only, no-install audit before allocating a T4 to the frozen H3 stack.

Reads package publisher metadata; never changes the Windows lock, installs
packages, or downloads model weights. A green notebook is not a qualification:
consume preflight.json's status and individual blockers.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from email.parser import Parser
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import sys
import time
from urllib.parse import quote
from urllib.request import urlopen


def fetch(url):
    with urlopen(url, timeout=30) as response:
        return response.read()


def audit_requirement(text, packages, environment):
    from packaging.requirements import Requirement
    from packaging.utils import canonicalize_name
    requirement = Requirement(text)
    if requirement.marker and not requirement.marker.evaluate(environment):
        return None
    name = canonicalize_name(requirement.name)
    version = packages.get(name)
    if version is None:
        return dict(kind='unlocked_linux_dependency', requirement=text)
    if requirement.specifier and not requirement.specifier.contains(version):
        return dict(kind='locked_version_conflict', requirement=text,
                    locked_package=name, locked_version=version)
    return None


def audit_package(item, environment):
    from packaging.tags import Tag, compatible_tags, cpython_tags
    from packaging.utils import parse_wheel_filename
    name, version = item
    url = 'https://pypi.org/pypi/' + quote(name) + '/' + quote(version, safe='') + '/json'
    try:
        data = json.loads(fetch(url))
        # Target Linux x86_64 CPython 3.11, independent of notebook Python.
        platforms = ['manylinux_2_%d_x86_64' % minor for minor in range(17, 40)]
        platforms += ['manylinux2014_x86_64', 'manylinux2010_x86_64', 'linux_x86_64']
        tags = set(cpython_tags((3, 11), platforms=platforms))
        tags.update(compatible_tags((3, 11), interpreter='cp311', platforms=platforms))
        tags.add(Tag('py3', 'none', 'any'))
        wheels = []
        for wheel in data['urls']:
            if wheel['packagetype'] != 'bdist_wheel' or wheel.get('yanked'):
                continue
            _, _, _, wheel_tags = parse_wheel_filename(wheel['filename'])
            if tags.intersection(wheel_tags):
                wheels.append(dict(filename=wheel['filename'], url=wheel['url'],
                                   sha256=wheel['digests']['sha256']))
        row=dict(name=name, version=version, metadata_url=url,
                 requires_dist=data['info'].get('requires_dist') or [],
                 linux_cp311_candidates=wheels,
                 qualification='Metadata only; wheel ABI and kernels untested')
        if wheels:
            # PyPI's project JSON can omit dependencies (notably torchaudio).
            # Read the selected Linux wheel's PEP 658 metadata as published.
            metadata_url=wheels[0]['url']+'.metadata'
            try:
                metadata=fetch(metadata_url)
                info=Parser().parsestr(metadata.decode())
                if info['Version']!=version:
                    raise RuntimeError('Wheel metadata version mismatch')
                row.update(metadata_url=metadata_url,
                           metadata_sha256=hashlib.sha256(metadata).hexdigest(),
                           requires_dist=info.get_all('Requires-Dist', []))
            except Exception as error:
                row['wheel_metadata_error']=str(error)
        return row
    except Exception as error:
        return dict(name=name, version=version, metadata_url=url,
                    error=type(error).__name__ + ': ' + str(error))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    spec = importlib.util.spec_from_file_location('frozen_setup', args.bundle/'frozen_kaggle_setup.py')
    frozen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(frozen)
    frozen.require_cpu_setup()
    # prepare_source expects an existing extraction root.
    source = args.output/'source'; source.mkdir(exist_ok=True)
    manifest = frozen.prepare_source(args.bundle, source)
    from packaging.markers import default_environment
    environment = default_environment()
    environment.update(platform_system='Linux', sys_platform='linux', os_name='posix',
                       platform_machine='x86_64', python_version='3.11',
                       python_full_version=manifest['python_version'], extra='')
    receipt = dict(schema=1, started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                   scope='CPU-only frozen Linux preflight; no installs, GPU allocation or model inference',
                   host=dict(python=platform.python_version(), platform=platform.platform()),
                   manifest_sha256=frozen.manifest_digest(manifest),
                   source_archive_sha256=manifest['source_archive_sha256'],
                   packages=manifest['packages'], blockers=[], metadata=[])
    if sys.platform != 'linux':
        receipt['blockers'].append(dict(kind='wrong_host', actual=sys.platform))
    locked_minor='.'.join(manifest['python_version'].split('.')[:2])
    if f'{sys.version_info.major}.{sys.version_info.minor}' != locked_minor:
        receipt['blockers'].append(dict(kind='host_python_minor_mismatch', expected=locked_minor,
                                        actual=platform.python_version()))
    # The CUDA Torch/vision builds live on their publisher's index, not PyPI.
    ordinary = [(n,v) for n,v in manifest['packages'].items() if '+cu' not in v]
    with ThreadPoolExecutor(max_workers=6) as pool:
        receipt['metadata'] = list(pool.map(lambda item:audit_package(item, environment), ordinary))
    for name, version in manifest['packages'].items():
        if '+cu' not in version:
            continue
        import html
        import re
        index=f'https://download.pytorch.org/whl/cu132/{name}/'
        try:
            links=re.findall(r'href="([^"]+)"', fetch(index).decode())
            candidates=[html.unescape(link) for link in links
                        if (quote(version, safe='') in link or version in link)
                        and 'cp311-cp311-manylinux' in link and 'x86_64.whl' in link]
            if len(candidates) != 1:
                raise RuntimeError('Expected exactly one pinned cp311 Linux wheel, found '+str(len(candidates)))
            url=candidates[0].split('#')[0]
            metadata_bytes=fetch(url+'.metadata')
            info=Parser().parsestr(metadata_bytes.decode())
            if info['Version'] != version:
                raise RuntimeError('Publisher metadata version mismatch')
            receipt['metadata'].append(dict(name=name, version=version,
                metadata_url=url+'.metadata', metadata_sha256=hashlib.sha256(metadata_bytes).hexdigest(),
                requires_dist=info.get_all('Requires-Dist', []),
                linux_cp311_candidates=[dict(url=candidates[0])]))
        except Exception as error:
            receipt['metadata'].append(dict(name=name,version=version,error=str(error),metadata_url=index))
    for row in receipt['metadata']:
        if 'error' in row:
            receipt['blockers'].append(dict(kind='metadata_unavailable', package=row['name'], reason=row['error']))
            continue
        if not row['linux_cp311_candidates']:
            receipt['blockers'].append(dict(kind='no_linux_cp311_wheel', package=row['name'], version=row['version']))
        for requirement in row['requires_dist']:
            problem=audit_requirement(requirement, manifest['packages'], environment)
            if problem:
                receipt['blockers'].append(dict(parent=row['name'], **problem))
    receipt['status']='blocked' if receipt['blockers'] else 'metadata_consistent_not_abi_qualified'
    receipt['finished_utc']=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    (args.output/'preflight.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    print(json.dumps({k:receipt[k] for k in ('status','host','blockers')},indent=2),flush=True)
    return 2 if receipt['blockers'] else 0


if __name__=='__main__':
    raise SystemExit(main())
