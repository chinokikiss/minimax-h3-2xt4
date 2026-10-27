"""Hash-bound private-input gate and decoder-only T4 batch; no Kaggle API calls."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time
import traceback


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--distribution', type=Path, required=True)
    parser.add_argument('--private-input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cpu-preflight', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    state = dict(status='starting', full_pipeline=False, video_decoder_only=True)
    stopped = threading.Event()
    def save():
        (args.output/'batch_results.json').write_text(json.dumps(state, indent=2))
    def heartbeat():
        while not stopped.wait(60):
            print('VAE_BATCH_HEARTBEAT', state['status'], flush=True)
    threading.Thread(target=heartbeat, daemon=True).start()
    save()
    try:
        manifest = json.loads((args.distribution/'delivery_manifest.json').read_text())
        state['source_capsule_sha256'] = manifest['capsule_sha256']
        locator = json.loads(args.private_input.read_text())
        assert locator['private'] and locator['repo_type'] == 'dataset'
        state['status'] = 'private_input_download';save()
        from huggingface_hub import hf_hub_download
        from kaggle_secrets import UserSecretsClient
        token = None
        for name in ('HF_TOKEN','HF_token'):
            try:
                token = UserSecretsClient().get_secret(name)
                break
            except Exception:
                pass
        if not token:
            raise RuntimeError('Private H3 latents require HF_TOKEN or the explicitly authorized private benchmark auth')
        latents = Path(hf_hub_download(locator['repo'], locator['file'], repo_type='dataset',
            revision=locator['revision'], token=token, cache_dir='/kaggle/temp/h3_vae_private_inputs'))
        del token
        assert latents.stat().st_size == locator['bytes'] and sha(latents) == locator['sha256']
        import torch
        z = torch.load(latents, map_location='cpu', weights_only=True)['video']
        assert list(z.shape) == locator['shape'] == [1,24,107,34,60]
        tensor_sha = hashlib.sha256(z.contiguous().view(torch.uint8).numpy()).hexdigest()
        assert tensor_sha == locator['tensor_sha256'] == '8fb3296f9dfc00641698e1a1aa26acb7f0fa98675b89b597dc95f10e64a9fa64'
        del z
        state['input'] = dict(file_sha256=locator['sha256'], tensor_sha256=tensor_sha, shape=locator['shape'])
        if args.cpu_preflight:
            state.update(status='cpu_input_ready', cuda_execution=False)
            return
        from patches.legacy_runtime import validate_packages, validate_source
        validate_source(args.source/'ComfyUI');validate_packages()
        hardware = [dict(name=torch.cuda.get_device_name(i),capability=list(torch.cuda.get_device_capability(i)),
                         uuid=str(torch.cuda.get_device_properties(i).uuid)) for i in range(torch.cuda.device_count())]
        assert len(hardware) == 2 and len({d['uuid'] for d in hardware}) == 2
        assert all(d['capability'] == [7,5] and 'T4' in d['name'] for d in hardware)
        state['hardware'] = hardware
        nvcc = Path(shutil.which('nvcc') or '')
        assert nvcc.is_file() and 'release 12.8' in subprocess.check_output([str(nvcc),'--version'],text=True)
        os.environ.update(CUDACXX=str(nvcc.resolve()),CUDA_HOME=str(nvcc.resolve().parent.parent),
            CUDA_PATH=str(nvcc.resolve().parent.parent),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',
            HF_HUB_DISABLE_XET='1',HF_HUB_DISABLE_PROGRESS_BARS='1',HF_HUB_DOWNLOAD_TIMEOUT='90')
        state['status'] = 'video_checkpoint_download';save()
        begin = time.monotonic()
        video = Path(hf_hub_download('Comfy-Org/MiniMax-H3',
            'vae/minimax_h3_video_vae_int8_convrot.safetensors',
            revision='e5eb578a89295337b8ff433a035929ce0279e0b6', token=False,
            cache_dir='/kaggle/temp/h3_vae_checkpoint'))
        assert video.stat().st_size == 2811065184
        assert sha(video) == '52a2c8c73583c86e4f41cdcce3a6ad0ea562987bc0bf3d60a0cef5f5c8e60c0e'
        state['model_download_and_verification_seconds'] = time.monotonic()-begin
        state['status'] = 'decoder_comparison';save()
        begin = time.monotonic()
        command = [sys.executable,'-u',str(args.source/'benchmarks/run_vae_temporal_compare.py'),
            '--video',str(video),'--latents',str(latents),'--output',str(args.output/'comparison'),'--repeats','3']
        with (args.output/'comparison.log').open('w') as stream:
            # The outer supervisor owns and bounds the complete process group.
            process = subprocess.Popen(command,cwd=args.source,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
            for line in process.stdout:
                stream.write(line);stream.flush()
                if line.startswith(('VAE_','Traceback')):
                    print(line,end='',flush=True)
            code = process.wait()
        state['comparison_seconds'], state['comparison_exit_code'] = time.monotonic()-begin, code
        if code:
            raise RuntimeError('VAE decoder comparison failed; inspect the retained stage evidence')
        result = json.loads((args.output/'comparison/result.json').read_text())
        assert result['status'] == 'complete' and len(result['cases']) == 6
        state.update(status='complete',summary=result['summary'],
            temporal_speedup_over_query=result['temporal_speedup_over_query'],
            all_outputs_byte_exact=result['all_outputs_byte_exact'])
    except BaseException as error:
        state.update(status='failed',error=repr(error))
        (args.output/'exception.txt').write_text(traceback.format_exc())
        raise
    finally:
        stopped.set();save()
        print('VAE_BATCH_FINAL',state['status'],flush=True)


if __name__ == '__main__':
    main()
