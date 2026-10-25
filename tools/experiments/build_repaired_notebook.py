"""Package one bounded benchmark of repaired TE/DiT blocks and full video VAE."""
import base64
import hashlib
import io
import json
from pathlib import Path
import tarfile
import sys

ROOT = Path(__file__).resolve().parents[2]
SLUG = 'chinokiki9/minimax-h3-repaired-dual-t4'


def main():
    proof = json.loads((ROOT / 'semantic_results/repaired_t4_inputs.json').read_text())
    if not proof['passed'] or len(proof['cases']) != 20:
        raise RuntimeError('Exact-input local dual-rank execution must pass')
    for name, expected in proof['source_sha256'].items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != expected:
            raise RuntimeError('Source changed after planned-input validation: ' + name)
    if hashlib.sha256((ROOT / 'repaired_fixtures/fixture_manifest.json').read_bytes()).hexdigest() != proof['fixture_manifest_sha256']:
        raise RuntimeError('Fixtures changed after validation')
    sys.path.insert(0, str(ROOT / 'benchmarks'))
    from checkpoint_ranges import make_plan
    plans = {kind: make_plan(Path('D:/minimaxH3') / original,
        ROOT / f'repaired_fixtures/{kind}_block0.safetensors', remote) for kind, original, remote in [
        ('te', 'qwen3vl_32b_minimax_h3_int8_convrot.safetensors', 'text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors'),
        ('dit', 'MiniMax_H3_Ref2VA_pruned_int8_convrot.safetensors', 'diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors')]}
    (ROOT / 'repaired_fixtures/checkpoint_plans.json').write_text(json.dumps(plans, indent=2))
    paths = set((ROOT / 'patches').glob('*.py'))
    paths.update(ROOT / 'benchmarks' / name for name in [
        'retest.py', 'repaired_t4.py', 'retest_slab.py', 'test_slab_semantics.py',
        'fixture_inputs.py', 'checkpoint_ranges.py'])
    paths.update(ROOT / name for name in ['semantic_results/repaired_t4_inputs.json',
        'kaggle_retest_output/final/aggregate_results.json'])
    paths.update(ROOT / f'kaggle_retest_output/cu130/retest_results/round_{r}/video_tiling/results.json' for r in range(3))
    paths.update(ROOT / 'repaired_fixtures' / name for name in [
        'fixture_manifest.json', 'checkpoint_plans.json',
        'input_768x768_1F.pt', 'input_768x1344_2F.pt'])
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode='w:gz') as tar:
        for path in sorted(paths):
            tar.add(path, arcname=path.relative_to(ROOT).as_posix())
    payload = base64.b64encode(stream.getvalue()).decode()
    code = '''import base64, io, json, os, pathlib, signal, subprocess, sys, tarfile, time
ROOT = pathlib.Path('/kaggle/working/repaired_source')
ROOT.mkdir(exist_ok=True)
archive = base64.b64decode(PAYLOAD)
pathlib.Path('/kaggle/working/repaired_source.tar.gz').write_bytes(archive)
with tarfile.open(fileobj=io.BytesIO(archive), mode='r:gz') as tar:
    tar.extractall(ROOT, filter='data')
started = time.monotonic()
setup_deadline = started + 420
manifest = {'status': 'setup', 'setup_budget_seconds': 420, 'gpu_budget_seconds': 720,
            'automatic_retry': False, 'jobs': [], 'scope': 'trained TE/DiT blocks; full raw video decoder and approximate slabs'}
manifest_path = pathlib.Path('/kaggle/working/repaired_manifest.json')
def save():
    manifest_path.write_text(json.dumps(manifest, indent=2))
def command(args):
    remaining = setup_deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError('Seven-minute setup budget exhausted')
    subprocess.run(args, check=True, timeout=remaining)
save()
try:
    fixtures = ROOT / 'repaired_fixtures'
    specs = json.loads((fixtures / 'fixture_manifest.json').read_text())
    command([sys.executable, '-m', 'pip', 'install', '--quiet',
             'comfy-kitchen==0.2.37', 'comfy-aimdo==0.5.5', 'safetensors==0.8.0',
             'huggingface-hub==1.12.0', 'einops', 'transformers', 'torchsde', 'sentencepiece'])
    command([sys.executable, '-m', 'pip', 'install', '--quiet', '--upgrade',
             'torch==2.11.0', 'torchvision==0.26.0', 'torchaudio==2.11.0',
             '--index-url', 'https://download.pytorch.org/whl/cu130'])
    comfy = ROOT / 'ComfyUI'
    command(['git', 'clone', '--filter=blob:none', 'https://github.com/Comfy-Org/ComfyUI.git', str(comfy)])
    command(['git', '-C', str(comfy), 'checkout', 'd49e888586dd8ae012c0667b33466b815fee07f7'])
    os.environ['COMFY_ROOT'] = str(comfy)
    os.environ['HF_HUB_DISABLE_XET'] = '1'
    os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
    os.environ['NCCL_P2P_DISABLE'] = '1'
    os.environ['NCCL_IB_DISABLE'] = '1'
    # Extract only the byte ranges corresponding to the locally proved blocks.
    # Exact input counters avoid CPU RNG differences across Torch releases.
    prepare = """import sys, json, pathlib, torch
root = pathlib.Path(ROOT_VALUE)
sys.path.insert(0, str(root / 'benchmarks'))
from checkpoint_ranges import download
from fixture_inputs import block_input
from retest import state, digest_tensor
fixtures = root / 'repaired_fixtures'
plans = json.loads((fixtures / 'checkpoint_plans.json').read_text())
specs = json.loads((fixtures / 'fixture_manifest.json').read_text())
for kind in ['te', 'dit']:
    path = fixtures / specs['checkpoints'][kind]['file']
    download(plans[kind], path)
    curve = state(path, 'adaln_t_table')[''] if kind == 'dit' else None
    for row in specs['inputs'][kind]:
        data = block_input(kind, row['length'], row['seed'], curve)
        if digest_tensor(data['input']) != row['input_sha256']:
            raise RuntimeError('Portable input fingerprint mismatch')
        torch.save(data, fixtures / row['file'])
    print('VERIFIED trained block and inputs', kind, flush=True)
for row in specs['inputs']['video']:
    x = torch.load(fixtures / ('input_' + row['name'] + '.pt'), weights_only=True, map_location='cpu')
    if digest_tensor(x) != row['input_sha256']:
        raise RuntimeError('Saved video input fingerprint mismatch')
    torch.save({'input': x}, fixtures / row['file'])
""".replace('ROOT_VALUE', repr(str(ROOT)))
    command([sys.executable, '-u', '-c', prepare])
    download = "from huggingface_hub import hf_hub_download; import pathlib; p=hf_hub_download('Comfy-Org/MiniMax-H3', 'vae/minimax_h3_video_vae_int8_convrot.safetensors', revision='e5eb578a89295337b8ff433a035929ce0279e0b6'); pathlib.Path('/kaggle/working/video_checkpoint_path.txt').write_text(p)"
    command([sys.executable, '-c', download])
    video = pathlib.Path('/kaggle/working/video_checkpoint_path.txt').read_text()
    manifest.update(status='running', setup_seconds=time.monotonic() - started)
    save()
    gpu_deadline = time.monotonic() + 720
    def job(name, args, budget):
        remaining = min(budget, gpu_deadline - time.monotonic())
        if remaining <= 0:
            raise TimeoutError('Twelve-minute GPU budget exhausted')
        out = pathlib.Path('/kaggle/working/repaired_results') / name
        out.mkdir(parents=True, exist_ok=True)
        args += ['--output', str(out), '--budget-seconds', str(max(1, int(remaining)))]
        print('START', name, 'budget', int(remaining), flush=True)
        start = time.monotonic()
        with (out / 'execution.log').open('w') as log:
            process = subprocess.Popen(args, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                returncode = process.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
                returncode = 124
            # A failed rank/parent must not leave NCCL workers consuming GPUs.
            if returncode:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
        print((out / 'execution.log').read_text(), flush=True)
        manifest['jobs'].append({'name': name, 'returncode': returncode, 'seconds': time.monotonic() - start})
        save()
        return returncode
    for suite, budget in [('te', 120), ('dit', 180), ('video', 240)]:
        checkpoint = video if suite == 'video' else str(fixtures / specs['checkpoints'][suite]['file'])
        job(suite, [sys.executable, '-u', str(ROOT / 'benchmarks/repaired_t4.py'),
            '--suite', suite, '--checkpoint', checkpoint, '--fixtures', str(fixtures)], budget)
    # The freshly measured full raw outputs provide same-allocation slab
    # quality references, eliminating historical mount/name ambiguity.
    import shutil
    references = ROOT / 'slab_references'
    references.mkdir(exist_ok=True)
    reference_manifest = []
    for row in specs['inputs']['video']:
        name = row['name']
        for filename in ['input_' + name + '.pt', 'native_' + name + '_output.pt']:
            origin = fixtures / filename if filename.startswith('input_') else pathlib.Path('/kaggle/working/repaired_results/video') / filename
            shutil.copyfile(origin, references / filename)
    # Hashing tensor bytes uses the benchmark runtime, in a separate process.
    reference_code = """import sys, pathlib, json, torch
root = pathlib.Path(ROOT_VALUE)
sys.path.insert(0, str(root / 'benchmarks'))
from retest import digest_tensor
references = root / 'slab_references'
rows = []
for path in sorted(references.glob('*.pt')):
    x = torch.load(path, weights_only=True, map_location='cpu')
    rows.append({'file': path.name, 'tensor_sha256': digest_tensor(x), 'shape': list(x.shape), 'dtype': str(x.dtype), 'source': 'same-allocation raw native reference / saved original input'})
(references / 'manifest.json').write_text(json.dumps(rows, indent=2))
""".replace('ROOT_VALUE', repr(str(ROOT)))
    subprocess.run([sys.executable, '-c', reference_code], check=True, timeout=max(1, gpu_deadline - time.monotonic()))
    job('slabs', [sys.executable, '-u', str(ROOT / 'benchmarks/retest_slab.py'),
        '--checkpoint', video, '--references', str(references), '--variant', '64_128',
        '--warmups', '1', '--iterations', '5'], 180)
    manifest.update(status='complete' if all(j['returncode'] == 0 for j in manifest['jobs']) else 'partial_failure',
                    total_seconds=time.monotonic() - started)
    save()
    if manifest['status'] != 'complete':
        raise RuntimeError('One or more suites failed; independent suites were still attempted within the budget')
except Exception as exc:
    manifest.update(status='failed', error=type(exc).__name__ + ': ' + str(exc), total_seconds=time.monotonic() - started)
    save()
    raise
'''.replace('PAYLOAD', repr(payload))
    compile(code, 'repaired-t4-notebook', 'exec')
    notebook = {'nbformat': 4, 'nbformat_minor': 5, 'metadata': {'kernelspec': {
        'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}}, 'cells': [
        {'id': 'scope', 'cell_type': 'markdown', 'metadata': {}, 'source': [
            '# Repaired dual-T4 benchmark\nActual trained TE/DiT block 0, complete raw video decoder, 64/128 slabs. Exact-input local simulation passed. One allocation, seven-minute setup/twelve-minute GPU budgets, no automatic retry.\n']},
        {'id': 'run', 'cell_type': 'code', 'metadata': {}, 'execution_count': None,
         'outputs': [], 'source': code.splitlines(keepends=True)}]}
    folder = ROOT / 'kaggle_repaired'
    folder.mkdir(exist_ok=True)
    (folder / 'repaired.ipynb').write_text(json.dumps(notebook, indent=1), encoding='utf-8')
    (folder / 'kernel-metadata.json').write_text(json.dumps({
        'id': SLUG, 'title': 'MiniMax H3 Repaired Dual T4', 'code_file': 'repaired.ipynb',
        'language': 'python', 'kernel_type': 'notebook', 'is_private': True,
        'enable_gpu': True, 'enable_tpu': False, 'enable_internet': True,
        'dataset_sources': [],
        'competition_sources': [], 'kernel_sources': [], 'model_sources': []}, indent=2), encoding='utf-8')
    print(folder)


if __name__ == '__main__':
    main()
