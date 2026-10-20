"""Run actual parallel algorithms as two CUDA ranks on one 5060 Ti.

Only rank placement and collective transport are substituted. Native CUDA
operators and independent rank tensors remain real. No cluster submission.
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
for path in [ROOT, ROOT / 'ComfyUI', ROOT / '.retest-deps']:
    sys.path.insert(0, str(path))
import json
import time
import hashlib
import argparse
import torch
import comfy.ops
from benchmarks.logical_dual_gpu import LogicalDualGPU, isolate_cublas_workspaces
from benchmarks.retest import metrics
from comfy.text_encoders.llama import Qwen3VL_32BConfig, TransformerBlock, precompute_freqs_cis
from comfy.ldm.minimax.model import DiTBlock
from comfy.ldm.modules.attention import optimized_attention_for_device
import patches.tp_qwen3vl as te_tp
import patches.sp_int8_qwen3vl as te_sp
import patches.tp_dit as dit_tp
import patches.sp_int8_dit as dit_sp
import patches.tp_video_vae as video_parallel


def quant_state(shapes, scale):
    state = {}
    config = torch.tensor(list(json.dumps({'format': 'int8_tensorwise', 'convrot': True,
                                         'convrot_groupsize': 256}).encode()), dtype=torch.uint8)
    for name, shape in shapes.items():
        state[name + '.weight'] = torch.randint(-8, 9, shape, dtype=torch.int8)
        state[name + '.weight_scale'] = torch.linspace(scale * .8, scale * 1.2, shape[0])[:, None]
        state[name + '.comfy_quant'] = config.clone()
    return state


def make_te(cls, full_size=False):
    config = Qwen3VL_32BConfig(hidden_size=512, intermediate_size=1024,
                             num_attention_heads=8, num_key_value_heads=4)
    config.head_dim = 64
    config.rope_dims = [12, 10, 10]
    if full_size:
        config = Qwen3VL_32BConfig()
    h, f, heads, kv, d = config.hidden_size, config.intermediate_size, config.num_attention_heads, config.num_key_value_heads, config.head_dim
    native = TransformerBlock(config, 0, device='cpu', dtype=torch.float16,
        ops=comfy.ops.mixed_precision_ops({'mixed_ops': True}, torch.float16))
    sd = quant_state({'self_attn.q_proj': (heads*d, h), 'self_attn.k_proj': (kv*d, h),
        'self_attn.v_proj': (kv*d, h), 'self_attn.o_proj': (h, heads*d),
        'mlp.gate_proj': (f, h), 'mlp.up_proj': (f, h), 'mlp.down_proj': (h, f)}, .01 if not full_size else .002)
    for name, size in {'input_layernorm': h, 'post_attention_layernorm': h,
                      'self_attn.q_norm': d, 'self_attn.k_norm': d}.items():
        sd[name + '.weight'] = torch.linspace(.8, 1.2, size).half()
    native.load_state_dict(sd, strict=True)
    ranks = [cls(h, f, heads, kv, d, rank=r, world_size=2, device='cuda:0').cuda() for r in range(2)]
    for rank in ranks:
        for name in ['input_layernorm', 'post_attention_layernorm', 'q_norm', 'k_norm']:
            key = name if 'layernorm' in name else 'self_attn.' + name
            getattr(rank, name).weight.data.copy_(sd[key + '.weight'])
        for name in ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']:
            key = ('self_attn.' if name in ['q_proj', 'k_proj', 'v_proj', 'o_proj'] else 'mlp.') + name
            getattr(rank, name).load_shard_from_full(sd[key + '.weight'], sd[key + '.weight_scale'])
        rank.eval().requires_grad_(False)
    return native.cuda().eval().requires_grad_(False), ranks, config


def make_dit(cls, full_size=False):
    h, heads, d, f = (5376, 56, 128, 14336) if full_size else (512, 8, 64, 1024)
    native = DiTBlock(h, heads, d, f, 8, 1e-5, 1e-5, apply_silu=False,
        adaln_dtype=torch.float32, device='cpu', dtype=torch.float32,
        operations=comfy.ops.mixed_precision_ops({'mixed_ops': True}, torch.float32))
    sd = quant_state({'attn.qkv_proj': (3*heads*d, h), 'attn.out_proj': (h, heads*d),
                      'mlp.fc1': (2*f, h), 'mlp.fc2': (h, f)}, .01 if not full_size else .002)
    for name, size in {'norm1': h, 'norm2': h, 'attn.q_norm': d, 'attn.k_norm': d}.items():
        sd[name + '.weight'] = torch.linspace(.8, 1.2, size)
    sd['adaln_proj.linear.weight'] = torch.randn(18*h, 8) * .05
    sd['adaln_proj.linear.bias'] = torch.randn(18*h) * .05
    native.load_state_dict(sd, strict=True)
    ranks = [cls(h, heads, d, f, 8, rank=r, world_size=2, device='cuda:0') for r in range(2)]
    for rank in ranks:
        for name in ['norm1', 'norm2', 'q_norm', 'k_norm']:
            key = name if name.startswith('norm') else 'attn.' + name
            getattr(rank, name).weight.data.copy_(sd[key + '.weight'])
        rank.adaln_proj.load_state_dict({'weight': sd['adaln_proj.linear.weight'], 'bias': sd['adaln_proj.linear.bias']})
        for name, key in {'qkv_proj': 'attn.qkv_proj', 'out_proj': 'attn.out_proj', 'fc1': 'mlp.fc1', 'fc2': 'mlp.fc2'}.items():
            getattr(rank, name).load_shard_from_full(sd[key + '.weight'], sd[key + '.weight_scale'])
        rank.eval().requires_grad_(False)
    return native.cuda().eval().requires_grad_(False), ranks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trained-model-dir', type=Path)
    args = parser.parse_args()
    start = time.perf_counter()
    # T4 has no TF32. Avoid Blackwell TF32 attention kernels introducing a
    # hardware-only difference between full and half head batch sizes.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    rows = []
    isolate_cublas_workspaces()
    comm = LogicalDualGPU()
    for module in [te_tp, te_sp, dit_tp, dit_sp, video_parallel]:
        module.dist = comm
    def record(name, out, ref, tolerance):
        result = metrics(out, ref)
        result['passed'] = result['valid'] and result['rel_l2'] <= tolerance
        rows.append({'case': name, **result})
        print(name, 'PASS' if result['passed'] else 'FAIL', f"L2={result.get('rel_l2', -1):.9g}", flush=True)
    with torch.no_grad():
        for strategy, cls in [('tp', te_tp.TPTransformerBlockINT8), ('sp', te_sp.SPTransformerBlockINT8)]:
            torch.manual_seed(919)
            native, ranks, config = make_te(cls)
            for batch, sequence in [(1, 32), (2, 37)]:
                x = torch.randn(batch, sequence, 512, device='cuda', dtype=torch.float16)
                # Unequal rank data/scales, nonuniform global positions, masked tail.
                positions = torch.arange(sequence, device='cuda').square()[None].expand(3, -1)
                freq = precompute_freqs_cis(64, positions, config.rope_theta, rope_dims=config.rope_dims,
                                          device=x.device, interleaved_mrope=True)
                mask = torch.full((sequence, sequence), torch.finfo(x.dtype).min / 4, device=x.device, dtype=x.dtype).triu(1)
                mask[:, -3:] = torch.finfo(x.dtype).min / 4
                attn = optimized_attention_for_device(x.device, mask=True, small_input=True)
                ref = native(x.clone(), mask, freq, attn)[0]
                def run(inp):
                    chunks = list(torch.tensor_split(inp, 2, dim=1)) if strategy == 'sp' else [inp, inp]
                    outputs = comm.run([lambda r=r: ranks[r](chunks[r].clone(), mask, freq) for r in range(2)])
                    return torch.cat(outputs, dim=1) if strategy == 'sp' else outputs[0]
                out = run(x)
                record(f'te_{strategy}_B{batch}_S{sequence}', out, ref, .001)
                changed = x.clone(); changed[:, sequence // 2:] *= -5
                other = run(changed)
                record(f'te_{strategy}_causal_B{batch}_S{sequence}', other[:, :sequence // 2], out[:, :sequence // 2], 0)
                changed = x.clone(); changed[:, -3:] *= -7
                other = run(changed)
                record(f'te_{strategy}_padding_B{batch}_S{sequence}', other[:, :-3], out[:, :-3], 0)
            del native, ranks
        if args.trained_model_dir:
            from benchmarks.retest import load_native_block, load_video_native
            trained_paths = {
                'te': args.trained_model_dir / 'qwen3vl_32b_minimax_h3_int8_convrot.safetensors',
                'dit': args.trained_model_dir / 'MiniMax_H3_Ref2VA_pruned_int8_convrot.safetensors'}
            for kind, path in trained_paths.items():
                native = load_native_block(kind, path)
                for strategy in ['tp', 'sp']:
                    cls = (te_tp.TPTransformerBlockINT8 if strategy == 'tp' else te_sp.SPTransformerBlockINT8) if kind == 'te' else (
                        dit_tp.TPDiTBlockINT8 if strategy == 'tp' else dit_sp.SPDiTBlockINT8)
                    ranks = [cls(rank=r, world_size=2, device='cuda:0').cuda().eval().requires_grad_(False) for r in range(2)]
                    for rank in ranks:
                        rank.load_from_full_block(native)
                    if kind == 'te':
                        x = torch.randn(1, 37, 5120, device='cuda', dtype=torch.float16)
                        config = Qwen3VL_32BConfig()
                        positions = torch.stack([torch.arange(37, device='cuda') ** exponent for exponent in [1, 2, 3]])
                        freq = precompute_freqs_cis(128, positions, config.rope_theta, rope_dims=config.rope_dims,
                                                  device=x.device, interleaved_mrope=True)
                        mask = torch.full((37, 37), torch.finfo(x.dtype).min / 4, device=x.device, dtype=x.dtype).triu(1)
                        mask[:, -3:] = torch.finfo(x.dtype).min / 4
                        ref = native(x.clone(), mask, freq, optimized_attention_for_device(x.device, mask=True, small_input=True))[0]
                        chunks = list(torch.tensor_split(x, 2, dim=1)) if strategy == 'sp' else [x, x]
                        outputs = comm.run([lambda r=r: ranks[r](chunks[r].clone(), mask, freq) for r in range(2)])
                        out = torch.cat(outputs, 1) if strategy == 'sp' else outputs[0]
                    else:
                        x = torch.randn(37, 5376, device='cuda', dtype=torch.float32)
                        t = torch.randn(2, 8, device='cuda', dtype=torch.float32)
                        segments = [(0, 7, 0), (7, 37, torch.arange(30, device='cuda') % 6)]
                        angle = torch.randn(1, 37, 1, 48, device='cuda')
                        c, s = angle.cos(), angle.sin()
                        rope = torch.stack([c, -s, s, c], -1).reshape(1, 37, 1, 48, 2, 2)
                        ref = native(x.clone(), t, segments, rope)
                        chunks = list(torch.tensor_split(x, 2, dim=0)) if strategy == 'sp' else [x, x]
                        outputs = comm.run([lambda r=r: ranks[r](chunks[r].clone(), t, segments, rope) for r in range(2)])
                        out = torch.cat(outputs, 0) if strategy == 'sp' else outputs[0]
                    record(f'trained_{kind}_{strategy}_block0_S37', out, ref, .00001)
                    del ranks
                del native
            path = args.trained_model_dir / 'minimax_h3_video_vae_int8_convrot.safetensors'
            for strategy in ['tp', 'sp']:
                native, sd = load_video_native(path, 'cuda:0')
                ranks = []
                patcher = video_parallel.patch_video_vae_tp if strategy == 'tp' else video_parallel.patch_video_vae_sp
                for r in range(2):
                    model, unused = load_video_native(path, 'cuda:0')
                    del unused
                    ranks.append(patcher(model, sd, r, 2, torch.device('cuda:0')))
                del sd
                for batch, temporal in [(1, 1), (2, 2)]:
                    z = torch.randn(batch, 24, temporal, 2, 3, device='cuda', dtype=torch.float16)
                    ref = native._decode_pixels(z)
                    outputs = comm.run([lambda r=r: ranks[r]._decode_pixels(z.clone()) for r in range(2)])
                    record(f'trained_video_{strategy}_36blocks_B{batch}_T{temporal}', outputs[0], ref, .001)
                del native, ranks, model
        # Actual checkpoint dimensions exercise dispatch branches absent from
        # tiny graphs, including TE's 25600-wide rotated-matmul quantizer.
        for strategy, cls in [('tp', te_tp.TPTransformerBlockINT8), ('sp', te_sp.SPTransformerBlockINT8)]:
            torch.manual_seed(959)
            native, ranks, config = make_te(cls, full_size=True)
            x = torch.randn(1, 17, config.hidden_size, device='cuda', dtype=torch.float16)
            positions = torch.arange(17, device='cuda').square()[None]
            freq = precompute_freqs_cis(config.head_dim, positions, config.rope_theta,
                rope_dims=config.rope_dims, device=x.device, interleaved_mrope=True)
            mask = torch.full((17, 17), torch.finfo(x.dtype).min / 4, device=x.device, dtype=x.dtype).triu(1)
            ref = native(x.clone(), mask, freq, optimized_attention_for_device(x.device, mask=True, small_input=True))[0]
            chunks = list(torch.tensor_split(x, 2, dim=1)) if strategy == 'sp' else [x, x]
            outputs = comm.run([lambda r=r: ranks[r](chunks[r].clone(), mask, freq) for r in range(2)])
            record(f'te_{strategy}_checkpoint_dimensions_S17', torch.cat(outputs, 1) if strategy == 'sp' else outputs[0], ref, .001)
            del native, ranks
        for strategy, cls in [('tp', dit_tp.TPDiTBlockINT8), ('sp', dit_sp.SPDiTBlockINT8)]:
            torch.manual_seed(969)
            native, ranks = make_dit(cls, full_size=True)
            x = torch.randn(17, 5376, device='cuda', dtype=torch.float32)
            t = torch.randn(2, 8, device='cuda', dtype=torch.float32)
            segments = [(0, 7, 0), (7, 17, torch.arange(10, device='cuda') % 6)]
            angle = torch.randn(1, 17, 1, 48, device='cuda')
            c, s = angle.cos(), angle.sin()
            rope = torch.stack([c, -s, s, c], -1).reshape(1, 17, 1, 48, 2, 2)
            ref = native(x.clone(), t, segments, rope)
            chunks = list(torch.tensor_split(x, 2, dim=0)) if strategy == 'sp' else [x, x]
            outputs = comm.run([lambda r=r: ranks[r](chunks[r].clone(), t, segments, rope) for r in range(2)])
            record(f'dit_{strategy}_checkpoint_dimensions_S17', torch.cat(outputs, 0) if strategy == 'sp' else outputs[0], ref, .00001)
            del native, ranks
        from comfy.text_encoders.llama import Llama2_
        from patches.pp_qwen3vl import PipelineParallelQwen3VL
        torch.manual_seed(949)
        te_block, unused, config = make_te(te_tp.TPTransformerBlockINT8)
        del unused
        config.num_hidden_layers, config.vocab_size = 2, 64
        config.final_norm = True
        config.fixed_kv = False
        ops = comfy.ops.mixed_precision_ops({'mixed_ops': True}, torch.float16)
        models = [Llama2_(config, device='cpu', dtype=torch.float16, ops=ops) for _ in range(2)]
        sd = {f'layers.{i}.{k}': v.cpu() for i in range(2) for k, v in te_block.state_dict().items()}
        sd['embed_tokens.weight'] = torch.randn(64, 512).half() * .1
        sd['norm.weight'] = torch.ones(512, dtype=torch.float16)
        for model in models:
            model.load_state_dict(sd, strict=True)
            model.cuda().eval().requires_grad_(False)
        native, placed = models
        pp = PipelineParallelQwen3VL(placed, 'cuda:0', 'cuda:0', split_layer=1)
        x = torch.randn(1, 17, 512, device='cuda', dtype=torch.float16)
        padding = torch.ones(1, 17, device='cuda'); padding[:, -2:] = 0
        positions = torch.stack([torch.arange(17, device='cuda') ** exponent for exponent in [1, 2, 3]])
        for selection in [0, 'all', [0, 1, 2]]:
            kwargs = dict(embeds=x.clone(), attention_mask=padding, position_ids=positions,
                          intermediate_output=selection)
            ref = native(None, **kwargs)
            out = pp(None, **{**kwargs, 'embeds': x.clone()})
            assert isinstance(out, tuple) and len(out) == len(ref) == 2
            record(f'te_pp_output_{selection}', out[0], ref[0], 0)
            record(f'te_pp_intermediate_{selection}', out[1], ref[1], 0)
        visual = torch.zeros(1, 17, device='cuda', dtype=torch.bool); visual[:, 3:6] = True
        deepstack = [torch.randn(3, 512, device='cuda', dtype=x.dtype) * .1 for _ in range(2)]
        kwargs = dict(embeds=x.clone(), deepstack_embeds=deepstack, visual_pos_masks=visual, position_ids=positions)
        ref = native(None, **kwargs)
        out = pp(None, **{**kwargs, 'embeds': x.clone()})
        record('te_pp_deepstack', out[0], ref[0], 0)
        ref = native(None, embeds=x.clone(), past_key_values=native.init_kv_cache(1, 18, x.device, x.dtype))
        out = pp(None, embeds=x.clone(), past_key_values=native.init_kv_cache(1, 18, x.device, x.dtype))
        record('te_pp_cache_prefill', out[0], ref[0], 0)
        tail = torch.randn(1, 1, 512, device='cuda', dtype=x.dtype)
        ref = native(None, embeds=tail.clone(), past_key_values=ref[2])
        out = pp(None, embeds=tail.clone(), past_key_values=out[2])
        record('te_pp_cache_decode', out[0], ref[0], 0)
        del models, native, placed, pp, te_block, sd
        from benchmarks.test_slab_semantics import make_fixture
        for strategy in ['tp', 'sp']:
            native, rank0 = make_fixture(heads=8)
            rank1, unused = make_fixture(heads=8)
            del unused
            sd = native.state_dict()
            patcher = video_parallel.patch_video_vae_tp if strategy == 'tp' else video_parallel.patch_video_vae_sp
            ranks = [patcher(model, sd, r, 2, torch.device('cuda:0')) for r, model in enumerate([rank0, rank1])]
            for batch, temporal, height, width in [(1, 1, 1, 3), (1, 1, 2, 3), (2, 2, 2, 3)]:
                z = torch.randn(batch, 24, temporal, height, width, device='cuda', dtype=torch.float16)
                ref = native._decode_pixels(z)
                outputs = comm.run([lambda r=r: ranks[r]._decode_pixels(z.clone()) for r in range(2)])
                record(f'video_{strategy}_B{batch}_T{temporal}_H{height}_W{width}', outputs[0], ref, .001)
                record(f'video_{strategy}_rank_agreement_B{batch}_T{temporal}_H{height}_W{width}', outputs[1], outputs[0], 0)
            del native, ranks, rank0, rank1, sd
        for strategy, cls in [('tp', dit_tp.TPDiTBlockINT8), ('sp', dit_sp.SPDiTBlockINT8)]:
            torch.manual_seed(929)
            native, ranks = make_dit(cls)
            for sequence in [32, 37]:
                x = torch.randn(sequence, 512, device='cuda', dtype=torch.float32)
                t = torch.randn(2, 8, device='cuda', dtype=torch.float32)
                # Cross-rank segments and per-token indices; all six modulation rows.
                segments = [(0, 7, 0), (7, sequence - 4, torch.arange(sequence - 11, device='cuda') % 6),
                            (sequence - 4, sequence, 5)]
                angle = torch.randn(1, sequence, 1, 24, device='cuda')
                c, s = angle.cos(), angle.sin()
                rope = torch.stack([c, -s, s, c], -1).reshape(1, sequence, 1, 24, 2, 2)
                ref = native(x.clone(), t, segments, rope)
                chunks = list(torch.tensor_split(x, 2, dim=0)) if strategy == 'sp' else [x, x]
                outputs = comm.run([lambda r=r: ranks[r](chunks[r].clone(), t, segments, rope) for r in range(2)])
                out = torch.cat(outputs, 0) if strategy == 'sp' else outputs[0]
                record(f'dit_{strategy}_S{sequence}', out, ref, .00001)
            from patches.pp_dit import PipelineParallelDiT
            stages = [DiTBlock(512, 8, 64, 1024, 8, 1e-5, 1e-5, apply_silu=False,
                adaln_dtype=torch.float32, device='cpu', dtype=torch.float32,
                operations=comfy.ops.mixed_precision_ops({'mixed_ops': True}, torch.float32)) for _ in range(2)]
            for block in stages:
                block.load_state_dict(native.state_dict(), strict=True)
                block.cuda().eval().requires_grad_(False)
            pp = PipelineParallelDiT(torch.nn.ModuleList(stages[:1]), torch.nn.ModuleList(stages[1:]),
                                     dev0='cuda:0', dev1='cuda:0')
            ref = native(native(x.clone(), t, segments, rope), t, segments, rope)
            out = pp(x.clone(), t, segments, rope)
            record(f'dit_pp_{strategy}_native_stack', out, ref, 0)
            del native, ranks
    result = {'gpu': torch.cuda.get_device_name(0), 'physical_gpus': torch.cuda.device_count(),
        'logical_ranks': 2, 'tf32_enabled': False, 'trained_model_dir': str(args.trained_model_dir) if args.trained_model_dir else None,
        'scope': 'native initialized graphs and, when supplied, trained TE/DiT block 0 and complete 36-block video decoder; two ranks on cuda:0; placement/transport simulated',
        'cases': rows, 'passed': all(r['passed'] for r in rows), 'wall_seconds': time.perf_counter() - start,
        'source_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in sorted(set((ROOT / 'patches').glob('*.py')) | {
                              Path(__file__), ROOT / 'benchmarks/logical_dual_gpu.py',
                              ROOT / 'benchmarks/retest.py', ROOT / 'benchmarks/test_slab_semantics.py'})}}
    path = ROOT / 'semantic_results/logical_dual_gpu.json'
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(result, indent=2))
    print('RESULT', result['passed'], 'seconds', result['wall_seconds'], flush=True)
    if not result['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
