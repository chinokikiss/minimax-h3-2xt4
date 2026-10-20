"""Execute on the provider, preserving actual profiler availability/permission evidence."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import hashlib
import tarfile
import urllib.request

OUT = Path('/kaggle/working/t4_profiler_probe')
OUT.mkdir(exist_ok=True)
result = {'commands': [], 'started': time.time()}


def save():
    (OUT / 'probe.json').write_text(json.dumps(result, indent=2))


def run(args, name, timeout=120):
    begin = time.monotonic()
    try:
        p = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
        record = dict(argv=args, exit_code=p.returncode, stdout=p.stdout, stderr=p.stderr)
    except Exception as e:
        record = dict(argv=args, exception=repr(e))
    record['seconds'] = time.monotonic() - begin
    result['commands'].append({'name': name, **record})
    (OUT / (name + '.json')).write_text(json.dumps(record, indent=2))
    print(name, json.dumps(record)[-9000:], flush=True)
    save()
    return record


import torch
result['hardware'] = {'torch': torch.__version__, 'cuda': torch.version.cuda,
    'gpus': [{'name': torch.cuda.get_device_name(i),
              'capability': list(torch.cuda.get_device_capability(i)),
              'memory': torch.cuda.get_device_properties(i).total_memory}
             for i in range(torch.cuda.device_count())]}
save()
print('HARDWARE', result['hardware'], flush=True)
assert len(result['hardware']['gpus']) == 2, result['hardware']
assert all(g['capability'] == [7, 5] and 'T4' in g['name'] for g in result['hardware']['gpus'])
run(['nvidia-smi', '-q'], 'nvidia_smi')
run(['id'], 'identity')
run(['cat', '/proc/driver/nvidia/params'], 'driver_policy')
run(['apt-cache', 'search', 'nsight'], 'packages')


def find_tool(name):
    found = shutil.which(name)
    if found:
        return found
    for base in ['/opt/nvidia', '/usr/local/cuda', '/usr/local/cuda-12.8']:
        if Path(base).exists():
            for p in Path(base).rglob(name):
                if p.is_file() and os.access(p, os.X_OK):
                    return str(p)


for name, package in [('nsys', 'nsight-systems'), ('ncu', 'nsight-compute')]:
    tool = find_tool(name)
    if not tool:
        policy = run(['apt-cache', 'policy', package], name + '_package_policy')
        if 'Candidate:' in policy.get('stdout', '') and 'Candidate: (none)' not in policy['stdout']:
            run(['apt-get', 'install', '-y', package], name + '_install', 420)
            tool = find_tool(name)
    if not tool:
        # Official CUDA redistributions; no driver changes or privilege bypass.
        archive_name = {
            'nsys': 'nsight_systems-linux-x86_64-2025.1.3.140-archive.tar.xz',
            'ncu': 'nsight_compute-linux-x86_64-2025.1.0.14-archive.tar.xz',
        }[name]
        distribution = 'nsight_systems' if name == 'nsys' else 'nsight_compute'
        url = f'https://developer.download.nvidia.com/compute/cuda/redist/{distribution}/linux-x86_64/{archive_name}'
        local = Path('/tmp/h3_profiler_tools') / name
        local.mkdir(parents=True, exist_ok=True)
        archive = local / archive_name
        download = {'url': url, 'started': time.time()}
        try:
            with urllib.request.urlopen(url, timeout=60) as response, archive.open('wb') as stream:
                while block := response.read(8 * 1024 * 1024):
                    stream.write(block)
                    if time.time() - download['started'] > 480:
                        raise TimeoutError('Profiler download exceeded 480 seconds')
            download['bytes'] = archive.stat().st_size
            with archive.open('rb') as stream:
                download['sha256'] = hashlib.file_digest(stream, 'sha256').hexdigest()
            with tarfile.open(archive) as bundle:
                bundle.extractall(local, filter='data')
            tool = next((str(p) for p in local.rglob(name)
                         if p.is_file() and os.access(p, os.X_OK)), None)
        except Exception as e:
            download['exception'] = repr(e)
        result[name + '_download'] = download
        save()
    result[name] = {'path': tool, 'available': bool(tool)}
    save()
    if tool:
        run([tool, '--version'], name + '_version')

demo = OUT / 'cuda_probe.py'
demo.write_text('''import torch
a=torch.randn(1024,1024,device="cuda"); b=torch.randn_like(a)
for _ in range(4): c=a@b
torch.cuda.synchronize()
torch.cuda.nvtx.range_push("probe_steady_matmul")
for _ in range(3): c=a@b
torch.cuda.synchronize()
torch.cuda.nvtx.range_pop()
print(float(c[0,0]))
''')
if result['nsys']['path']:
    rec = run([result['nsys']['path'], 'profile', '--trace=cuda,nvtx', '--sample=none',
               '--cpuctxsw=none', '--force-overwrite=true', '-o', str(OUT / 'cuda_trace'),
               sys.executable, str(demo)], 'nsys_capture', 240)
    result['nsys']['trace_success'] = rec.get('exit_code') == 0 and (OUT / 'cuda_trace.nsys-rep').exists()
if result['ncu']['path']:
    rec = run([result['ncu']['path'], '--set', 'basic', '--launch-skip', '4',
               '--launch-count', '1', '--force-overwrite', '-o', str(OUT / 'cuda_counters'),
               sys.executable, str(demo)], 'ncu_capture', 240)
    result['ncu']['counters_success'] = (rec.get('exit_code') == 0
        and (OUT / 'cuda_counters.ncu-rep').exists()
        and 'ERR_NVGPUCTRPERM' not in rec.get('stdout', '') + rec.get('stderr', ''))
result['completed'] = time.time()
save()
print('PROBE_RESULT', json.dumps(result), flush=True)
