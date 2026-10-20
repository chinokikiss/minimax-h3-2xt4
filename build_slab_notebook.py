"""Package one new slab-only run after verifying local simulation evidence."""
import base64
import hashlib
import io
import json
from pathlib import Path
import tarfile

ROOT = Path(__file__).resolve().parent
SLUG = "chinokiki9/minimax-h3-slab-only"
COMFY_SHA = "d49e888586dd8ae012c0667b33466b815fee07f7"


def main():
    local = json.loads((ROOT / "slab_results/trained_local_semantics.json").read_text())
    logical = json.loads((ROOT / "semantic_results/logical_dual_gpu.json").read_text())
    if not logical["passed"] or not local["trained_checkpoint"] or len(local["cases"]) != 13:
        raise RuntimeError("Trained local simulations must pass before packaging")
    if not local.get("passed") or not all(
        c["rank_decodes_bit_exact_vs_sequential"]
        and c["matching_sequential_slabs"]["rel_l2"] <= 1e-5
        and c["matching_sequential_slabs"]["max_abs"] <= 5e-4
        for c in local["cases"]):
        raise RuntimeError("Slab scheduling simulation failed")
    for proof in [local, logical]:
        for name, expected in proof["source_sha256"].items():
            source = ROOT / name.replace("\\", "/")
            if hashlib.sha256(source.read_bytes()).hexdigest() != expected:
                raise RuntimeError(f"Source changed after local validation: {name}")
    paths = set((ROOT / "patches").glob("*.py"))
    paths.update(ROOT / "benchmarks" / name for name in [
        "retest.py", "retest_slab.py", "test_slab_semantics.py"])
    paths.update(ROOT / name for name in [
        "slab_results/trained_local_semantics.json", "semantic_results/logical_dual_gpu.json",
        "slab_results/saved_references/manifest.json", "kaggle_retest_output/final/aggregate_results.json"])
    paths.update(ROOT / f"kaggle_retest_output/cu130/retest_results/round_{r}/video_tiling/results.json"
                 for r in range(3))
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w:gz") as tar:
        for path in sorted(paths):
            tar.add(path, arcname=path.relative_to(ROOT).as_posix())
    payload = base64.b64encode(stream.getvalue()).decode()
    code = '''import base64, hashlib, io, json, os, pathlib, shutil, subprocess, sys, tarfile, time
ROOT = pathlib.Path('/kaggle/working/slab_source')
ROOT.mkdir(exist_ok=True)
archive = base64.b64decode(PAYLOAD)
pathlib.Path('/kaggle/working/slab_source.tar.gz').write_bytes(archive)
with tarfile.open(fileobj=io.BytesIO(archive), mode='r:gz') as tar:
    tar.extractall(ROOT, filter='data')
setup_started = time.monotonic()
setup_deadline = setup_started + 300
manifest = {'status': 'setup', 'setup_budget_seconds': 300, 'gpu_budget_seconds': 600,
            'new_modes_only': True, 'variants': ['64', '128'], 'automatic_retry': False}
manifest_path = pathlib.Path('/kaggle/working/slab_manifest.json')
def save():
    manifest_path.write_text(json.dumps(manifest, indent=2))
def command(args):
    remaining = setup_deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError('Five-minute setup budget exhausted')
    subprocess.run(args, check=True, timeout=remaining)
save()
try:
    # Reuse the completed allocation's saved tensors, never regenerate seeds.
    references = ROOT / 'slab_results/saved_references'
    for row in json.loads((references / 'manifest.json').read_text()):
        # Each historical round has the same basename. Select the recorded
        # round/suite path, not an arbitrary first match across the mount.
        hits = [p for p in pathlib.Path('/kaggle/input').rglob(row['file'])
                if p.as_posix().endswith('/' + row['source'])]
        if len(hits) != 1:
            raise RuntimeError('Expected exactly one mounted historical tensor: ' + row['file'])
        shutil.copyfile(hits[0], references / row['file'])
    command([sys.executable, '-m', 'pip', 'install', '--quiet',
             'comfy-kitchen==0.2.37', 'comfy-aimdo==0.5.5', 'safetensors==0.8.0',
             'huggingface-hub==1.12.0', 'einops', 'transformers', 'torchsde', 'sentencepiece'])
    command([sys.executable, '-m', 'pip', 'install', '--quiet', '--upgrade',
             'torch==2.11.0', 'torchvision==0.26.0', 'torchaudio==2.11.0',
             '--index-url', 'https://download.pytorch.org/whl/cu130'])
    comfy = ROOT / 'ComfyUI'
    command(['git', 'clone', '--filter=blob:none', 'https://github.com/Comfy-Org/ComfyUI.git', str(comfy)])
    command(['git', '-C', str(comfy), 'checkout', COMFY_SHA])
    os.environ['COMFY_ROOT'] = str(comfy)
    os.environ['HF_HUB_DISABLE_XET'] = '1'
    os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
    # Download only the locally verified 2.8GB Video VAE; no TE/DiT/audio models.
    download = "from huggingface_hub import hf_hub_download; import pathlib; p=hf_hub_download('Comfy-Org/MiniMax-H3', 'vae/minimax_h3_video_vae_int8_convrot.safetensors', revision='e5eb578a89295337b8ff433a035929ce0279e0b6'); pathlib.Path('/kaggle/working/checkpoint_path.txt').write_text(p)"
    command([sys.executable, '-c', download])
    checkpoint = pathlib.Path('/kaggle/working/checkpoint_path.txt').read_text()
    manifest.update(status='running', setup_seconds=time.monotonic() - setup_started)
    save()
    out = pathlib.Path('/kaggle/working/slab_results/t4_64_128')
    out.mkdir(parents=True, exist_ok=True)
    args = [sys.executable, '-u', str(ROOT / 'benchmarks/retest_slab.py'),
            '--checkpoint', checkpoint, '--references', str(references), '--output', str(out),
            '--variant', '64_128', '--warmups', '1', '--iterations', '5', '--budget-seconds', '600']
    with (out / 'execution.log').open('w') as log:
        process = subprocess.Popen(args, stdout=log, stderr=subprocess.STDOUT)
        try:
            returncode = process.wait(timeout=600)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
            raise TimeoutError('Ten-minute slab experiment budget exhausted')
    print((out / 'execution.log').read_text(), flush=True)
    manifest.update(status='complete' if returncode == 0 else 'failed', returncode=returncode)
    save()
    if returncode:
        raise RuntimeError('Slab-only process failed; inspect persisted results')
except Exception as exc:
    manifest.update(status='failed', error=type(exc).__name__ + ': ' + str(exc))
    save()
    raise
'''.replace("PAYLOAD", repr(payload)).replace("COMFY_SHA", repr(COMFY_SHA))
    compile(code, "slab-only-notebook", "exec")
    folder = ROOT / "kaggle_slab"
    folder.mkdir(exist_ok=True)
    notebook = {"nbformat": 4, "nbformat_minor": 5, "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
        "cells": [{"id": "scope", "cell_type": "markdown", "metadata": {}, "source": [
            "# New slab-only experiment\nLocally verified trained computation. One job, two workloads, 64/128 overlap; 5-minute setup and 10-minute GPU cap. Reuses saved historical tensors and timings.\n"]},
            {"id": "slab-only", "cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
             "source": code.splitlines(keepends=True)}]}
    (folder / "slab.ipynb").write_text(json.dumps(notebook, indent=1), encoding="utf-8")
    metadata = {"id": SLUG, "title": "MiniMax H3 Slab Only", "code_file": "slab.ipynb",
        "language": "python", "kernel_type": "notebook", "is_private": True,
        "enable_gpu": True, "enable_tpu": False, "enable_internet": True,
        "dataset_sources": [], "competition_sources": [], "model_sources": [],
        "kernel_sources": ["chinokiki9/minimax-h3-cuda13-retest"]}
    (folder / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(folder)


if __name__ == "__main__":
    main()
