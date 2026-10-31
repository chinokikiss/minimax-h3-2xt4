"""One bounded Real-H3 attention microgate: stage exact assets, bake Turbo, capture first DiT attention, replay operators."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import threading
import time
import traceback


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def download_with_retry(operation, name, on_retry, *, sleep=time.sleep):
    """Resume transient HF transfers; never retry integrity or authorization errors."""
    import httpx
    import requests
    for attempt in range(3):
        try:
            return operation()
        except Exception as error:
            status = getattr(getattr(error, 'response', None), 'status_code', None)
            transient = (status in (408, 429, 500, 502, 503, 504) or
                (status is None and isinstance(error, (httpx.TransportError,
                    requests.ConnectionError, requests.Timeout, ConnectionError, TimeoutError))))
            if not transient or attempt == 2:
                raise
            delay = (5, 15)[attempt]
            on_retry(dict(file=name, failed_attempt=attempt+1, http_status=status,
                error_type=type(error).__name__, backoff_seconds=delay))
            sleep(delay)


def release_owned_dit_payloads(scratch, root):
    scratch, root = Path(scratch).resolve(), Path(root).resolve()
    if not root.is_relative_to(scratch) or not (scratch / '.owned_by_turbo_e2e').is_file():
        raise ValueError('Refuse deletion outside this job-owned download directory')
    paths = [root / f'dit.block{i:02d}.rank{r}.safetensors' for i in range(50) for r in (0, 1)]
    if not all(p.is_file() and p.resolve().is_relative_to(root) for p in paths):
        raise ValueError('Incomplete/nonlocal owned base payloads')
    released = sum(p.stat().st_size for p in paths)
    for path in paths:
        path.unlink()
    return released


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--plan',type=Path)
    parser.add_argument('--scratch',type=Path,default=Path('/kaggle/temp/h3_l1_turbo_05mp'))
    parser.add_argument('--e2e-timeout',type=int,default=2700)
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    if not 60<=args.e2e_timeout<=7200:raise ValueError('Bounded generation timeout required')
    output.mkdir(parents=True, exist_ok=True)
    state = dict(status='starting', commands=[], downloads={}, full_pipeline=False,
                 scope='Real-H3 first-attention capture/replay only; no full denoise trajectory or media generation',
                 online_lora=False, request_count=1)
    def save():
        (output / 'batch_results.json').write_text(json.dumps(state, indent=2))
    def mark(stage, **extra):
        state.update(status=stage, **extra)
        save()
        print('TURBO_STAGE', stage, flush=True)
    def run(argv, label, cap):
        mark(label)
        begin = time.monotonic()
        with (output / (label + '.log')).open('w') as stream:
            process = subprocess.Popen(argv, cwd=source, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, text=True, start_new_session=True, bufsize=1)
            def relay():
                for line in process.stdout:
                    stream.write(line)
                    stream.flush()
                    if line.startswith(('TP2_', 'L1_', 'REF2VA_', 'Traceback')):
                        print(line, end='', flush=True)
            thread = threading.Thread(target=relay, daemon=True)
            thread.start()
            try:
                process.wait(timeout=cap)
            finally:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait(timeout=5)
                thread.join(timeout=5)
                state['commands'].append(dict(stage=label, seconds=time.monotonic()-begin,
                                               exit_code=process.returncode))
                save()
            if process.returncode:
                raise RuntimeError(label + ' failed; see saved stage log')
    stopped = threading.Event()
    def heartbeat():
        while not stopped.wait(60):
            print('TURBO_HEARTBEAT', state['status'], flush=True)
    threading.Thread(target=heartbeat, daemon=True).start()
    try:
        import torch
        import importlib.metadata as md
        from patches.legacy_runtime import validate_packages, validate_source
        validate_packages()
        validate_source(source / 'ComfyUI')
        devices = [dict(name=torch.cuda.get_device_name(i),
            capability=list(torch.cuda.get_device_capability(i)),
            uuid=str(torch.cuda.get_device_properties(i).uuid)) for i in range(torch.cuda.device_count())]
        if (len(devices) != 2 or len({d['uuid'] for d in devices}) != 2
                or not all(d['capability'] == [7, 5] and 'T4' in d['name'] for d in devices)):
            raise RuntimeError('Two actual SM75 Tesla T4 GPUs required')
        compiler = Path(shutil.which('nvcc') or '')
        if not compiler.is_file() or 'release 12.8' not in subprocess.check_output([str(compiler), '--version'], text=True):
            raise RuntimeError('Validated preinstalled CUDA12.8 nvcc required')
        os.environ.update(CUDACXX=str(compiler.resolve()), CUDA_HOME=str(compiler.resolve().parent.parent),
            CUDA_PATH=str(compiler.resolve().parent.parent), HF_HUB_DISABLE_XET='1',
            HF_HUB_DISABLE_PROGRESS_BARS='1', HF_HUB_DOWNLOAD_TIMEOUT='90',
            OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2',
            MALLOC_ARENA_MAX='2', MALLOC_TRIM_THRESHOLD_='67108864')
        state['hardware'] = dict(devices=devices, torch=torch.__version__, cuda=torch.version.cuda,
            kitchen=md.version('comfy-kitchen'), tilelang=md.version('tilelang'))
        plan = json.loads((args.plan or source/'turbo_plan.json').read_text())
        if plan.get('workload_contract')=='turbo_1mp_10s':
            from benchmarks.turbo_challenge_contract import validate
            validate(plan)
        assets = json.loads((source / 'bundle_assets.json').read_text())
        identity = json.loads((source / 'l1_model_identity.json').read_text())
        references = json.loads((source / 'l1_assets_identity.json').read_text())['images']
        scratch = args.scratch
        scratch.mkdir(exist_ok=False)
        (scratch / '.owned_by_turbo_e2e').write_text('Staged public base downloads for this one Real-H3 microgate')
        root = scratch / 'base_bundle'
        root.mkdir()
        for path in (source / 'bundle_provenance').glob('*.json'):
            shutil.copyfile(path, root / path.name)
        if sha(root / 'bundle.json') != assets['publication']['bundle_sha256']:
            raise ValueError('Frozen base bundle index mismatch')
        images = []
        for row in references:
            path = source / 'bench_assets/l1_current' / row['file']
            if row.get('url'):
                from benchmarks.qualify_anime_inputs import download_image
                path.parent.mkdir(parents=True, exist_ok=True)
                if not path.exists():download_image(row, path.parent)
            if path.stat().st_size != row['bytes'] or sha(path) != row['sha256']:
                raise ValueError('Frozen reference image mismatch')
            images.append(str(path))
        state['request'] = plan
        state['reference_images'] = references
        save()
        from huggingface_hub import hf_hub_download
        def record_download_retry(event):
            state.setdefault('download_retries', []).append(event)
            print('TP2_DOWNLOAD_RETRY', json.dumps(event), flush=True)
        def download(repo, revision, name, expected, directory):
            path = Path(download_with_retry(
                lambda: hf_hub_download(repo, name, revision=revision, token=False, local_dir=str(directory)),
                name, record_download_retry))
            if path.stat().st_size != expected['bytes'] or sha(path) != expected['sha256']:
                raise ValueError('Immutable weight checksum mismatch: ' + name)
            return path
        def prepare_kind(kind):
            mark('download_' + kind + '_tp2')
            objects = [(n, row) for n, row in assets['objects'].items() if n.startswith(kind + '.')]
            if len(objects) != 100:
                raise ValueError('Exactly100 base rank shards required per model')
            needed = sum(row['bytes'] for _, row in objects) + assets['auxiliary'][kind]['bytes'] + 2*2**30
            if shutil.disk_usage(scratch).free < needed:
                raise OSError('Insufficient stage disk headroom')
            def one(item):
                name, expected = item
                path = download(assets['publication']['repo'], assets['publication']['commit'], name, expected, root)
                return dict(file=name, bytes=path.stat().st_size, sha256=expected['sha256'])
            rows = []
            with ThreadPoolExecutor(max_workers=4) as pool:
                futures = [pool.submit(one, item) for item in objects]
                for future in as_completed(futures):
                    rows.append(future.result())
                    if len(rows) % 10 == 0:
                        print('TP2_DOWNLOAD_VERIFIED', kind, len(rows), '/100', flush=True)
                        state['downloads'][kind] = rows
                        save()
            mark('assemble_' + kind + '_auxiliary')
            from checkpoint_ranges import assemble
            import requests
            aux_plan = assets['auxiliary'][kind]
            original = assets['source_identity'][kind]
            url = 'https://huggingface.co/' + assets['model_repository'] + '/resolve/' + assets['model_revision'] + '/' + original['remote_file']
            def read(start, end):
                expected_range=f'bytes {start}-{end-1}/{original["bytes"]}'
                expected_bytes=end-start
                last=None
                for attempt in range(3):
                    try:
                        with requests.get(url, params={'download':'true'},
                                headers={'Range':f'bytes={start}-{end-1}',
                                         'Accept-Encoding':'identity'},
                                stream=True, timeout=(15, 90)) as response:
                            actual_range=response.headers.get('Content-Range')
                            if response.status_code != 206 or actual_range != expected_range:
                                raise ValueError(
                                    f'Auxiliary immutable range mismatch: status={response.status_code} '
                                    f'content_range={actual_range!r} expected={expected_range!r}')
                            value=bytearray()
                            for chunk in response.iter_content(4*2**20):
                                value.extend(chunk)
                                if len(value)>expected_bytes:
                                    raise ValueError('Oversized auxiliary range')
                            if len(value)!=expected_bytes:
                                raise ValueError(
                                    f'Short auxiliary range: got={len(value)} expected={expected_bytes}')
                            return value
                    except (requests.RequestException,ValueError) as error:
                        last=error
                        if attempt==2:
                            break
                        print('AUX_RANGE_RETRY',kind,start,end,attempt+1,type(error).__name__,str(error),flush=True)
                        time.sleep(1<<attempt)
                raise ValueError(f'Auxiliary immutable range failed after 3 attempts: {last}') from last
            auxiliary = scratch / (kind + '_aux.safetensors')
            assemble(aux_plan, auxiliary, read)
            from patches.tp2_bundle import DESCRIPTOR
            descriptor = scratch / (kind + '_bundle.json')
            descriptor.write_text(json.dumps(dict(format=DESCRIPTOR, kind=kind, bundle_root=root.name,
                bundle_sha256=assets['publication']['bundle_sha256'],
                source={k:original[k] for k in ('bytes', 'sha256')},
                native_header=aux_plan['native_header'], native_header_sha256=aux_plan['native_header_sha256'],
                auxiliary_file=auxiliary.name, auxiliary_sha256=aux_plan['sha256']), indent=2))
            state['downloads'][kind] = rows
            save()
            return descriptor
        dit = prepare_kind('dit')
        mark('download_replacement_turbo')
        turbo = plan['turbo_lora']
        lora = download(turbo['repository'], turbo['revision'], turbo['file'], turbo, scratch / 'adapter')
        state['turbo_download'] = {k:turbo[k] for k in ('repository', 'revision', 'file', 'bytes', 'sha256')}
        save()
        baked = scratch / 'baked_tp2'
        needed = sum(r['bytes'] for n,r in assets['objects'].items() if n.startswith('dit.')) + assets['auxiliary']['dit']['bytes'] + 2*2**30
        if shutil.disk_usage(scratch).free < needed:
            raise OSError('Insufficient disk for atomic persistent bake')
        run([sys.executable, '-u', '-m', 'patches.tp2_disk_bake', '--base', str(dit), '--lora', str(lora), '1.0',
             '--output', str(baked), '--devices', 'cuda:0', 'cuda:1'], 'offline_disk_bake', 1800)
        bake_receipt = json.loads((baked / 'bake_receipt.json').read_text())
        if (bake_receipt['verified_native_targets'] != plan['turbo_lora']['targets']
                or bake_receipt['devices'] != ['cuda:0', 'cuda:1']):
            raise ValueError('Incomplete Native dual-GPU bake verification')
        shutil.copyfile(baked / 'bake_receipt.json', output / 'offline_disk_bake_receipt.json')
        state['persistent_bake'] = dict(path=str(baked), receipt_sha256=sha(baked / 'bake_receipt.json'),
            bytes=bake_receipt['persistent_baked_bytes'], verified_native_targets=bake_receipt['verified_native_targets'],
            persistence_scope='Job disk cache survives model/window eviction; media/receipts are notebook outputs',
            inference_remerge=False, online_lora=False)
        state['released_job_owned_dit_download_bytes'] = release_owned_dit_payloads(scratch, root)
        save()
        te = prepare_kind('te')
        paths = dict(te=str(te), dit=str(baked / 'dit_baked.json'))
        mark('download_vaes')
        for kind in ('video',):
            expected = identity[kind]
            path = download(assets['model_repository'], assets['model_revision'], expected['remote_file'], expected, scratch / 'vaes')
            paths[kind] = str(path)
        request = plan | dict(paths=paths, images=images, output=str(output / 'generation'), lora_stack=[])
        (output / 'request_config.json').write_text(json.dumps(request, indent=2))
        run([sys.executable, '-u', str(source / 'benchmarks/real_h3_attention_microgate.py'),
             '--config', str(output / 'request_config.json')], 'real_h3_attention_microgate', args.e2e_timeout)
        result = json.loads((output / 'generation/result.json').read_text())
        if result['status'] != 'complete':
            raise RuntimeError('Real-H3 attention microgate incomplete')
        mark('complete', microgate_summary=result.get('summary'),
             scope='First real post-RoPE DiT attention only; no generation/decode')
    except BaseException as error:
        state.update(status='failed', error=type(error).__name__ + ': ' + str(error))
        (output / 'exception.txt').write_text(traceback.format_exc())
        save()
        raise
    finally:
        stopped.set()


if __name__ == '__main__':
    main()
