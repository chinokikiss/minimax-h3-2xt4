"""Resolve official CUDA 13 wheel metadata into a fully hashed Linux lock.

No wheel installation or local environment modification. Non-platform
dependencies must be satisfied by the preserved core/native inventories.
"""
import argparse
import email
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import urllib.parse
import urllib.request
from packaging.requirements import Requirement
from packaging.tags import cpython_tags, compatible_tags
from packaging.utils import canonicalize_name, parse_wheel_filename
from packaging.version import Version


def fetch(url):
    with urllib.request.urlopen(url, timeout=60) as response:
        return response.read()


class Links(HTMLParser):
    def __init__(self):
        super().__init__(); self.links = []

    def handle_starttag(self, tag, attrs):
        if tag == 'a':
            self.links.extend(v for k, v in attrs if k == 'href')


def build(base, core):
    environment = dict(implementation_name='cpython', implementation_version='3.13.15',
        os_name='posix', platform_machine='x86_64', platform_release='', platform_system='Linux',
        platform_version='', python_full_version='3.13.15', python_version='3.13',
        sys_platform='linux', extra='')
    platforms = [f'manylinux_2_{n}_x86_64' for n in range(39, 4, -1)] + ['manylinux2014_x86_64', 'linux_x86_64']
    tags = set(cpython_tags((3, 13), ['cp313'], platforms)) | set(compatible_tags((3, 13), 'cp313', platforms))
    rows, queue, checked = {}, [], set()
    versions = dict(base, **core)
    for name, version in (('torch','2.11.0'), ('torchvision','0.26.0'), ('torchaudio','2.11.0')):
        index = f'https://download.pytorch.org/whl/cu130/{name}/'
        parser = Links(); parser.feed(fetch(index).decode())
        links = [urllib.parse.urljoin(index, x) for x in parser.links
                 if f'{name}-{version}%2Bcu130-cp313-cp313-manylinux_2_28_x86_64.whl' in x]
        if len(links) != 1:
            raise RuntimeError('Ambiguous official wheel: '+name)
        url, fragment = links[0].split('#', 1)
        metadata = fetch(url+'.metadata')
        message = email.message_from_bytes(metadata)
        rows[name] = dict(version=version+'+cu130', url=url, sha256=fragment.removeprefix('sha256='),
                          metadata_sha256=hashlib.sha256(metadata).hexdigest())
        versions[name] = version+'+cu130'
        queue.extend((Requirement(r), {''}) for r in message.get_all('Requires-Dist', []))
    retained = {}
    while queue:
        req, extras = queue.pop(0)
        if req.marker and not any(req.marker.evaluate(dict(environment, extra=e)) for e in extras):
            continue
        name = canonicalize_name(req.name)
        selected = rows.get(name, {}).get('version')
        if selected:
            if not req.specifier.contains(selected):
                raise ValueError('Conflicting CUDA requirement: '+str(req))
            continue
        platform = name.startswith(('nvidia-', 'cuda-')) or name == 'triton'
        if not platform:
            value = versions.get(name)
            if value is None or not req.specifier.contains(value):
                raise RuntimeError('Preserved dependency does not satisfy '+str(req)+': '+str(value))
            retained[name] = value
            continue
        info = json.loads(fetch('https://pypi.org/pypi/'+name+'/json'))
        candidates = sorted((Version(v) for v, files in info['releases'].items()
                             if files and not Version(v).is_prerelease and req.specifier.contains(v)))
        if not candidates:
            raise RuntimeError('No official release for '+str(req))
        # Use the lowest satisfying stable release for ranges; metadata's exact
        # versions and wildcard versions constrain the CUDA Toolkit family.
        version = str(candidates[0])
        detail = json.loads(fetch(f'https://pypi.org/pypi/{name}/{version}/json'))
        wheels = [w for w in detail['urls'] if w['packagetype']=='bdist_wheel' and not w['yanked']
                  and parse_wheel_filename(w['filename'])[3] & tags]
        if not wheels:
            raise RuntimeError('No CP313 Linux wheel: '+name+'=='+version)
        wheel = sorted(wheels, key=lambda w: w['filename'])[0]
        rows[name] = dict(version=version, url=wheel['url'], sha256=wheel['digests']['sha256'])
        for child in detail['info'].get('requires_dist') or []:
            queue.append((Requirement(child), set(req.extras) | {''}))
        checked.add(name)
    return dict(format='h3-vae-cu130-v1', python='3.13', platform='Linux-x86_64',
                torch='2.11.0+cu130', cuda='13.0', packages=rows, retained_dependencies=retained,
                authorization='2026-10-28: isolated VAE dispatch attribution only; DiT retains native CUDA 12.8')


def main():
    p=argparse.ArgumentParser(); p.add_argument('--native-receipt',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True); args=p.parse_args()
    receipt=json.loads(args.native_receipt.read_text())
    result=build(receipt['native_base_packages'],receipt['core_pins'])
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2))
    print(json.dumps(dict(packages=len(result['packages']),sha256=hashlib.sha256(args.output.read_bytes()).hexdigest())))


if __name__=='__main__':main()
