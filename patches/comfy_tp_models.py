"""Native model construction for TP2; no profiling or worker transport."""
import torch
from torch import nn
from patches.weight_residency import copy_module


def native_te(reader, layers, startup_state=None):
    """Full Native vision/front end, with decoder shells created without weights."""
    import comfy.ops
    import comfy.sd1_clip
    from comfy.text_encoders.minimax import MiniMaxQwen3VL, MiniMaxH3ClipModel, MiniMaxH3TEModel
    operations = comfy.ops.mixed_precision_ops({'mixed_ops': True}, torch.float16)

    def transformer(config, dtype, device, ops):
        model = MiniMaxQwen3VL({'num_hidden_layers': 0}, dtype, device, ops)
        state = reader.read('', exclude=('model.layers.',)) if startup_state is None else startup_state
        model.load_state_dict(state, strict=True)
        model.num_layers = layers
        return model

    class Clip(MiniMaxH3ClipModel):
        def __init__(self, device='cpu', dtype=torch.float16, **kwargs):
            comfy.sd1_clip.SDClipModel.__init__(self, device=device, layer='last',
                textmodel_json_config={}, dtype=dtype, special_tokens={'pad':151643},
                layer_norm_hidden_state=False, model_class=transformer,
                enable_attention_masks=False, return_attention_masks=False,
                model_options={'custom_operations': operations})
    return MiniMaxH3TEModel(device='cpu', dtype=torch.float16,
        model_options={'qwen3vl_32b_class': Clip}).eval().requires_grad_(False)


def native_dit(reader, rank=None):
    import comfy.ops
    from comfy.ldm.minimax.model import MiniMaxH3Model
    ops = comfy.ops.mixed_precision_ops({'mixed_ops': True}, torch.float32)
    model = MiniMaxH3Model(num_layers=0, adaln_curve_grid=1025, time_embed_dim=8,
        dtype=torch.float32, device='cpu', operations=ops)
    exclude = ('blocks.',)
    if rank == 1:
        names = ('condition_proj','token_refiner','video_patch_proj','audio_patch_proj')
        for name in names:
            setattr(model, name, nn.Identity())
        exclude += tuple(name+'.' for name in names)
    model.load_state_dict(reader.read('', exclude=exclude), strict=True)
    store=getattr(reader,'lora_store',None)
    if store is not None and rank!=1:store.attach_refiner(model)
    if rank!=1:
        from patches.h3_operator_policy import NativeRefinerAttention
        for block in model.token_refiner.blocks:block.attn=NativeRefinerAttention(block.attn)
    return model.eval().requires_grad_(False)


class StagedVision(nn.Module):
    """Release vision weights before the long text decoder owns its window."""
    def __init__(self, canonical, device):
        super().__init__()
        self.canonical, self.device = canonical, device

    def forward(self, *args, **kwargs):
        module = copy_module(self.canonical, lambda t:t.to(self.device))
        try:
            return module(*args, **kwargs)
        finally:
            # Native vision may retain converted FP32 projection weights;
            # their lifetime must end before the text blocks start streaming.
            torch.cuda.current_stream(self.device).synchronize()
            del module


class CPUStartupState(nn.Module):
    """Checkpoint tensors only: defer Native construction to actual demand."""
    def __init__(self, reader, names):
        super().__init__()
        self.tensors = {name:reader.read_tensor(name) for name in names}
