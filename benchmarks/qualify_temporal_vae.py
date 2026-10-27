"""Local trained decoder correctness gate; logical ranks are not a T4 benchmark."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'ComfyUI'), str(ROOT/'benchmarks')]
import torch
from qualify_video_decode_memory import VideoService
from logical_dual_gpu import isolate_cublas_workspaces
from patches.comfy_sampler import ParallelDecodeHandoff
from patches.staged_vae import StagedVAE


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', type=Path, required=True)
    parser.add_argument('--latents', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    import comfy.ldm.minimax.vae
    import comfy.model_management as management
    import comfy_kitchen as ck
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    ck.backends.cuda._max_dynamic_shared_memory_per_block = lambda unused: 49152
    isolate_cublas_workspaces()
    service = VideoService()
    # VideoService's parent check covers both raw-tile and whole-clip dispatch.
    vae = StagedVAE(args.video, service)
    service.parent = vae.vae
    state = dict(status='running', physical_t4=False, hardware=torch.cuda.get_device_name(),
                  scope='Trained Video decoder on two logical ranks of one local GPU; correctness only', cases=[])
    try:
        saved = torch.load(args.latents, map_location='cpu', weights_only=True)['video']
        for length, height, width in ((2,16,16),(7,16,16),(12,16,16),(12,34,60)):
            z = saved[:,:,:length,:height,:width].contiguous()
            reference = vae.decode(z).cpu()
            service.trained = vae.first_stage_model
            for mode in ('query','temporal'):
                start = time.monotonic()
                wrapper = ParallelDecodeHandoff(vae,service,args.video,mode)
                actual = wrapper.decode(z).cpu()
                torch.cuda.synchronize()
                row = dict(tokens=length, shape=list(z.shape), mode=mode, bit_exact=torch.equal(actual,reference),
                    max_abs=float((actual-reference).abs().max()), finite=bool(torch.isfinite(actual).all()),
                    strategy=wrapper.last_strategy, fallback=wrapper.last_fallback,
                    diagnostic_seconds=time.monotonic()-start)
                state['cases'].append(row)
                args.output.write_text(json.dumps(state,indent=2))
                print(json.dumps(row),flush=True)
                assert row['bit_exact'] and row['finite'] and row['fallback'] is None
        state['status'] = 'passed'
    except BaseException as error:
        state.update(status='failed',error=repr(error))
        raise
    finally:
        vae.release();service.close();management.unload_all_models()
        args.output.write_text(json.dumps(state,indent=2))


if __name__ == '__main__':
    main()
