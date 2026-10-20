"""Locate a trained VAE transfer mismatch; no timing sweep."""
import simulate_weight_residency as base
import dataclasses
import torch
import retest
from patches.weight_residency import ResidencyWindow, WeightSpec, tensor_leaves


def main():
    path = base.PATHS['video']
    torch.set_grad_enabled(False)
    with torch.inference_mode():
        with torch.inference_mode(False), torch.no_grad():
            native, sd = retest.load_video_native(path, 'cuda:0')
        del sd
        def prepare():
            host, sd = retest.load_video_native(path, 'cpu')
            del sd
            return host
        spec = WeightSpec('video', prepare, path.stat().st_size + 64*2**20)
        window = ResidencyWindow([spec], ['video'], 'cuda:0', max_weight_bytes=4*2**30,
            pinned_bytes=4*2**30, reserve_bytes=256*2**20)
        try:
            with window.use('video') as model:
                torch.cuda.synchronize()
                differences = []
                for name, source in native.named_parameters():
                    other = model.get_parameter(name)
                    a, b = list(tensor_leaves(source)), list(tensor_leaves(other))
                    if len(a) != len(b) or any(x.dtype != y.dtype or not torch.equal(x, y) for x, y in zip(a, b)):
                        differences.append(name)
                    if hasattr(source, '_params') and source._params.orig_dtype != other._params.orig_dtype:
                        differences.append(name+' orig_dtype')
                for name, source in native.named_buffers():
                    other = model.get_buffer(name)
                    if source.dtype != other.dtype or not torch.equal(source, other):
                        differences.append(name)
                print('PARAMETER_DIFFERENCES', len(differences), differences[:4], flush=True)
                name = 'decoder.transformer_blocks.0.attn.to_qkv.weight'
                for label, instance in [('native', native), ('window', model)]:
                    qt = instance.get_parameter(name)
                    print('QUANT', label, type(qt), qt.dtype, getattr(qt, '_params', None),
                          instance.decoder.transformer_blocks[0].attn.to_qkv.quant_format, flush=True)
                    print('QUANT_LEAVES', label, [(t.dtype, list(t.shape), t.flatten()[:4].tolist()) for t in tensor_leaves(qt)], flush=True)
                z = torch.arange(1*24*2*4*4, device='cuda:0').reshape(1,24,2,4,4).half()/1000
                saved, hooks = {}, []
                names = ['post_quant_conv', 'decoder.x_embedder', 'decoder.transformer_blocks.0.attn.to_qkv',
                         'decoder.transformer_blocks.0.attn.to_out', 'decoder.transformer_blocks.0.ff.w1',
                         'decoder.transformer_blocks.0', 'decoder.transformer_blocks.35', 'decoder.proj_out']
                for label, instance in [('native', native), ('window', model)]:
                    for name in names:
                        def capture(m, a, out, label=label, name=name):
                            saved[label, name] = out.detach().clone()
                        hooks.append(instance.get_submodule(name).register_forward_hook(capture))
                ref, out = native._decode_pixels(z), model._decode_pixels(z)
                print('OUTPUT', retest.metrics(out, ref), flush=True)
                for name in names:
                    if ('native',name) in saved and ('window',name) in saved:
                        print('INTERMEDIATE', name, retest.metrics(saved['window',name], saved['native',name]), flush=True)
                for hook in hooks:
                    hook.remove()
        finally:
            window.close()


if __name__ == '__main__':
    main()
