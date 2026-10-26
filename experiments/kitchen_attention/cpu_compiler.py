"""Extract NVIDIA's exact CUDA12.8.1 build tools into a temporary CPU directory.

No pip/conda install, native environment change, runtime replacement or GPU use.
Hashes are NVIDIA's redistrib_12.8.1.json linux-x86_64 publisher digests.
"""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tarfile
import urllib.request


ARCHIVES={
    'cuda_nvcc':('12.8.93','9961b3484b6b71314063709a4f9529654f96782ad39e72bf1e00f070db8210d3',79015464),
    'cuda_cudart':('12.8.90','8d566b5fe745c46842dc16945cf36686227536decd2302c372be86da37faca68',1354240),
    'cuda_cccl':('12.8.90','0740e9e01e4f15e17c5ab8d68bba4f8ec0eb6b84edccba4ac45112d2d2174e4b',928524),
}


def prepare(root):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    staging=root/'archives';staging.mkdir(exist_ok=True)
    receipt=[]
    for name,(version,digest,size) in ARCHIVES.items():
        filename=f'{name}-linux-x86_64-{version}-archive.tar.xz'
        url=f'https://developer.download.nvidia.com/compute/cuda/redist/{name}/linux-x86_64/{filename}'
        target=staging/filename
        with urllib.request.urlopen(url,timeout=60) as response,target.open('wb') as stream:
            shutil.copyfileobj(response,stream)
        assert target.stat().st_size==size and hashlib.sha256(target.read_bytes()).hexdigest()==digest
        unpack=staging/name;unpack.mkdir()
        with tarfile.open(target,'r:xz') as archive:archive.extractall(unpack,filter='data')
        folders=[p for p in unpack.iterdir() if p.is_dir()];assert len(folders)==1
        source=folders[0]
        for path in source.iterdir():
            if path.is_dir():shutil.copytree(path,root/path.name,dirs_exist_ok=True,symlinks=True)
            else:shutil.copyfile(path,staging/(name+'_'+path.name))
        receipt.append(dict(name=name,version=version,url=url,sha256=digest,bytes=size))
    # NVIDIA redist archives use lib; nvcc's conventional profile can use lib64.
    if (root/'lib').exists() and not (root/'lib64').exists():(root/'lib64').symlink_to('lib',target_is_directory=True)
    version=subprocess.check_output([str(root/'bin/nvcc'),'--version'],text=True,timeout=10)
    assert 'V12.8.93' in version
    (root/'toolchain_receipt.json').write_text(json.dumps(dict(archives=receipt,compiler=version),indent=2)+'\n')
    return root/'bin/nvcc',receipt
