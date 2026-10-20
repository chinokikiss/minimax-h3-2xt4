"""Actual Native ComfyUI sampler vs persistent logical TP2 on trained weights.

Compact audiovisual references and Native vision tower; correctness only.
Use --layers 50 for the complete trained decoder/denoiser checkpoint.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import threading
import traceback
import subprocess

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT/'ComfyUI', ROOT/'.retest-deps', ROOT/'benchmarks'):
    sys.path.insert(0, str(path))

import torch
from torch import nn
import retest
from full_models import StreamedBlock
from ref2va_workloads import request
from logical_dual_gpu import LogicalDualGPU, isolate_cublas_workspaces
from patches.comfy_tp_runtime import RuntimeConfig, RankRuntime, native_te, native_dit, packet
from patches.comfy_sampler import create_model, create_clip, DecodeHandoff
from patches.host_weight_cache import HostBudget, GiB
from patches.windowed_tp import CheckpointReader
from patches.optimized_tp import enable


class LogicalService:
    def __init__(self, config):
        self.config, self.lock, self.closed = config, threading.RLock(), False
        self.comm = LogicalDualGPU(timeout=config.timeout_seconds)
        import patches.tp_qwen3vl as te
        import patches.tp_dit as dit
        te.dist = dit.dist = self.comm
        self.budget = HostBudget(config.host_bytes, reserve_bytes=GiB)
        self.ranks = [RankRuntime(config, r, 'cuda:0', self.comm, self.budget) for r in range(2)]

    def call(self, command, *args, **kwargs):
        with self.lock:
            args, kwargs = packet(args, 'cpu'), packet(kwargs,'cpu')
            def run(rank):
                output = self.ranks[rank].dispatch(command,args,kwargs)
                output = packet(output,'cpu')
                torch.cuda.current_stream().synchronize()
                return output
            results = self.comm.run([lambda r=r:run(r) for r in range(2)])
            return results if command == 'stats' else results[0]

    def close(self):
        self.call('close')
        self.closed = True


def sample(model, positive, negative, case, interrupt=False):
    import comfy.samplers
    from comfy.nested_tensor import NestedTensor
    data, _ = request('dit', 68, 1700, 'cpu')
    latent = NestedTensor([x*.1 for x in data[0]])
    generator = torch.Generator().manual_seed(1729)
    noise = NestedTensor([torch.randn(x.shape, generator=generator) for x in data[0]])
    masks = [torch.ones_like(x) for x in data[0]]
    masks[0][..., :2,:2] = .25
    masks[1][..., :2] = 0
    guider = comfy.samplers.CFGGuider(model)
    guider.set_conds(positive,negative)
    guider.set_cfg(case['cfg'])
    sigmas = comfy.samplers.calculate_sigmas(model.model.model_sampling, 'simple', case['steps'])
    callbacks = []
    def callback(i,x0,x,total):
        callbacks.append((i,total,[list(t.shape) for t in x0.unbind()]))
        if interrupt:
            import comfy.model_management
            raise comfy.model_management.InterruptProcessingException()
    result = guider.sample(noise, latent, comfy.samplers.sampler_object(case['sampler']),
        sigmas, denoise_mask=NestedTensor(masks), disable_pbar=True, seed=1729,
        callback=callback)
    return result.cpu(), callbacks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--layers', type=int, default=2)
    parser.add_argument('--physical', action='store_true')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--checkpoints', type=Path)
    parser.add_argument('--include-video', action='store_true')
    parser.add_argument('--mode', choices=('all','reference','verify'), default='all')
    args = parser.parse_args()
    target = args.output or ROOT/f'semantic_results/comfy_sampler_{args.layers}/result.json'
    target.parent.mkdir(parents=True,exist_ok=True)
    reference_path = target.parent/'native_reference.pt'
    if args.mode == 'all':
        # Reference CPU allocator high-water storage must not compete with
        # persistent workers on a 32 GB host. Terminate the reference process
        # before starting the actual integrated request; never relax admission.
        base = [sys.executable,'-u',str(Path(__file__).resolve()),*sys.argv[1:]]
        subprocess.run([*base,'--mode','reference'],check=True,timeout=900)
        subprocess.run([*base,'--mode','verify'],check=True,timeout=900)
        return
    rows = []
    sources = list((ROOT/'patches').glob('*.py'))+[Path(__file__),ROOT/'benchmarks/logical_dual_gpu.py']
    result = {'passed':False, 'physical':args.physical, 'layers':args.layers,
        'scope':'Native ComfyUI conditioning/vision and actual res_multistep/heun CFG masked audiovisual sampling; compact references; no latency claim',
        'cases':rows, 'source_sha256':{p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}}
    def save():
        target.write_text(json.dumps(result,indent=2),encoding='utf-8')
    save()
    enable('scalemax')
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    if not args.physical:
        import comfy_kitchen as ck
        ck.backends.cuda._max_dynamic_shared_memory_per_block = lambda x:49152
        isolate_cublas_workspaces()
    paths = json.loads(args.checkpoints.read_text()) if args.checkpoints else {
        'te':'D:/minimaxH3/qwen3vl_32b_minimax_h3_int8_convrot.safetensors',
        'dit':'D:/minimaxH3/MiniMax_H3_Ref2VA_pruned_int8_convrot.safetensors',
        'video':'D:/minimaxH3/minimax_h3_video_vae_int8_convrot.safetensors'}
    config = RuntimeConfig(paths['te'], paths['dit'], layers=args.layers,
        weight_bytes=4*GiB if not args.physical else 6*GiB, reserve_bytes=256*2**20,
        timeout_seconds=120 if not args.physical else 300,
        te_weight_bytes=2*GiB if not args.physical else None)
    from comfy.text_encoders.minimax import MiniMaxH3Tokenizer
    tokenizer = MiniMaxH3Tokenizer()
    image = torch.linspace(0,1,64*64*3).reshape(1,64,64,3)
    tokens = tokenizer.tokenize_with_weights('A small bird moves.', images=[image])
    ref_tokens = tokenizer.tokenize_with_weights('A small bird moves.', minimax_ref_items=[
        {'type':'image','data':image}, {'type':'video','data':torch.cat([image,1-image]),
         'timestamps':[0.,.5]}])
    negative_tokens = tokenizer.tokenize_with_weights('blur')
    # Native vision tower, Native metadata expansion and trained streamed text.
    _, options = request('dit',68,1700,'cpu')
    extra = {'minimax_refs':options['minimax_payload']['refs'],
        'minimax_visual_cond_noise_aug':.999, 'minimax_audio_cond_noise_aug':1.0}
    cases = [{'sampler':'res_multistep','steps':2,'cfg':1.}]
    if args.layers < 50:
        cases.append({'sampler':'heun','steps':2,'cfg':2.})
    import comfy.model_management
    if args.mode == 'reference':
        print('NATIVE conditioning', args.layers, flush=True)
        text = native_te(CheckpointReader(config.te_path),args.layers).cuda()
        text.qwen3vl_32b.transformer.model.layers = nn.ModuleList([
            StreamedBlock('te',config.te_path,i,'cuda:0') for i in range(args.layers)])
        text.set_clip_options({'execution_device':torch.device('cuda:0')})
        with torch.inference_mode():
            conditioning = packet(text.encode_token_weights(tokens),'cpu')
            ref_conditioning = packet(text.encode_token_weights(ref_tokens),'cpu')
            negative = packet(text.encode_token_weights(negative_tokens),'cpu')
        del text
        retest.cleanup()
        positive = [[ref_conditioning[0], {**ref_conditioning[2], **extra}]]
        negative_cond = [[negative[0], {**negative[2], **extra}]]
        class Placeholder:
            lock = threading.RLock()
            closed = False
        native = create_model(Placeholder())
        native.remove_wrappers_with_key('outer_sample','minimax_tp2_session')
        native.model.diffusion_model = native_dit(CheckpointReader(config.dit_path)).cuda()
        native.model.diffusion_model.blocks = nn.ModuleList([
            StreamedBlock('dit',config.dit_path,i,'cuda:0') for i in range(args.layers)])
        goldens = []
        for case in cases:
            print('NATIVE sampler',case,flush=True)
            with torch.inference_mode():
                output,callbacks = sample(native,positive,negative_cond,case)
                goldens.append((output.unbind(),callbacks))
        comfy.model_management.unload_all_models()
        del native
        retest.cleanup()
        pixels = None
        if args.include_video:
            import comfy.sd
            print('NATIVE ComfyUI VAE reference',flush=True)
            with torch.inference_mode(False), torch.no_grad():
                vae = comfy.sd.VAE(sd=CheckpointReader(paths['video']).read(''),
                    device=torch.device('cuda:0'),dtype=torch.float16)
            with torch.inference_mode():
                pixels = vae.decode(goldens[0][0][0]).cpu()
        torch.save(dict(conditioning=conditioning,ref_conditioning=ref_conditioning,
            negative=negative,goldens=goldens,pixels=pixels,layers=args.layers,
            source_sha256=result['source_sha256']),reference_path)
        return
    reference = torch.load(reference_path,map_location='cpu',weights_only=False)
    if reference['layers'] != args.layers or reference['source_sha256'] != result['source_sha256']:
        raise RuntimeError('Native reference was produced by a different source/configuration')
    conditioning,ref_conditioning,negative = [reference[k] for k in ('conditioning','ref_conditioning','negative')]
    from comfy.nested_tensor import NestedTensor
    goldens = [(NestedTensor(tensors),callbacks) for tensors,callbacks in reference['goldens']]
    if args.physical:
        from patches.comfy_tp_runtime import PhysicalTPService
        service = PhysicalTPService(config)
    else:
        service = LogicalService(config)
    try:
        clip = create_clip(service)
        print('TP Native CLIP',flush=True)
        with torch.inference_mode():
            output = packet(clip.encode_from_tokens(tokens, return_dict=True),'cpu')
        rows.append({'name':'TE vision DeepStack conditioning', **retest.metrics(output['cond'],conditioning[0])})
        if not torch.equal(conditioning[2]['minimax_token_tags'],output['minimax_token_tags']):
            raise AssertionError('Native image/text token tags changed')
        with torch.inference_mode():
            ref_output = packet(clip.encode_from_tokens(ref_tokens, return_dict=True),'cpu')
            negative_output = packet(clip.encode_from_tokens(negative_tokens, return_dict=True),'cpu')
        rows.append({'name':'TE image/video DeepStack conditioning',
            **retest.metrics(ref_output['cond'],ref_conditioning[0])})
        rows.append({'name':'TE negative conditioning', **retest.metrics(negative_output['cond'],negative[0])})
        if not torch.equal(ref_conditioning[2]['minimax_token_tags'],ref_output['minimax_token_tags']):
            raise AssertionError('Native video token tags changed')
        positive = [[ref_output.pop('cond'), {**ref_output, **extra}]]
        negative_cond = [[negative_output.pop('cond'), {**negative_output, **extra}]]
        model = create_model(service)
        for case, (golden, gold_callbacks) in zip(cases,goldens):
            print('TP Native sampler',case,flush=True)
            with torch.inference_mode():
                out, callbacks = sample(model,positive,negative_cond,case)
            for i,(expected, actual) in enumerate(zip(golden.unbind(),out.unbind())):
                rows.append({'name':f"{case['sampler']} cfg={case['cfg']} stream={i}",
                    **retest.metrics(actual,expected)})
            if callbacks != gold_callbacks:
                raise AssertionError('Native callbacks changed')
        if args.layers < 50:
            print('TP cancellation and next sample',flush=True)
            try:
                with torch.inference_mode():
                    sample(model,positive,negative_cond,cases[0],interrupt=True)
            except comfy.model_management.InterruptProcessingException:
                pass
            else:
                raise AssertionError('Sampler callback did not interrupt')
            with torch.inference_mode():
                repeat,_ = sample(model,positive,negative_cond,cases[0])
            for i,(expected,actual) in enumerate(zip(goldens[0][0].unbind(),repeat.unbind())):
                rows.append({'name':f'post-cancellation next request stream={i}',**retest.metrics(actual,expected)})
            result['cancellation_passed'] = True
        if args.include_video:
            import comfy.sd
            print('Native ComfyUI VAE decode and RAM handoff',flush=True)
            with torch.inference_mode(False), torch.no_grad():
                vae = comfy.sd.VAE(sd=CheckpointReader(paths['video']).read(''),
                    device=torch.device('cuda:0'),dtype=torch.float16)
            with torch.inference_mode():
                z = goldens[0][0].unbind()[0]
                handed = DecodeHandoff(vae,service).decode(z).cpu()
            rows.append({'name':'Native ComfyUI VAE decode and handoff', **retest.metrics(handed,reference['pixels'])})
            comfy.model_management.unload_all_models()
            del vae, handed
            retest.cleanup()
            with torch.inference_mode():
                next_te = clip.encode_from_tokens(negative_tokens,return_dict=True)
            rows.append({'name':'TE after completed VAE handoff', **retest.metrics(next_te['cond'],negative[0])})
            service.call('begin_sample')
            service.call('end_sample')
        result['runtime'] = service.call('stats')
        if any(r['sample_active'] or r['phase'] is not None for r in result['runtime']):
            raise AssertionError('Sampler completion leaked GPU residency')
        result['passed'] = all(row.get('valid') and row['rel_l2'] <=1e-5
            and row['cosine'] >= .999999 and row['max_abs'] <=.005 for row in rows)
        save()
    finally:
        service.close()
        result['host_budget_after_close'] = service.budget.snapshot()
        save()
    if not result['passed']:
        raise AssertionError('Native sampler comparison failed')


if __name__ == '__main__':
    main()
