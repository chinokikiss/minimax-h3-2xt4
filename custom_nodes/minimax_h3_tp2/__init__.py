"""Loader and optional post-decode RAM scheduling for the validated INT8 stack."""
from pathlib import Path
import sys
import threading

ROOT = Path(__file__).resolve().parents[2]
if not (ROOT/'patches/comfy_tp_runtime.py').is_file():
    raise RuntimeError('Keep this node linked to its repository; do not copy __init__.py alone')
sys.path.insert(0, str(ROOT))

from patches.comfy_tp_runtime import RuntimeConfig, PhysicalTPService, GiB
from patches.h3_operator_policy import H3OperatorPolicy
from patches.comfy_sampler import create_model, create_clip, DecodeHandoff, ParallelDecodeHandoff

_lock = threading.RLock()
_service = None


class MiniMaxH3TP2Loader:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required':{
            'text_encoder':('STRING', {'default':'D:/minimaxH3/qwen3vl_32b_minimax_h3_int8_convrot.safetensors'}),
            'diffusion_model':('STRING', {'default':'D:/minimaxH3/MiniMax_H3_Ref2VA_pruned_int8_convrot.safetensors'}),
            'gpu_weight_window_gib':('FLOAT', {'default':6., 'min':3.5, 'max':10., 'step':.5}),
            'ram_cache_per_rank_gib':('FLOAT', {'default':3., 'min':3., 'max':4., 'step':.5})},
            'optional':{'idle_residency':('BOOLEAN',{'default':False}),
                        'lora_stack':('MINIMAX_TP2_LORAS',),
                        'tp_quantization':(['scalemax'],{'default':'scalemax'}),
                        'idle_weight_gib':('FLOAT',{'default':4.,'min':.5,'max':8.,'step':.5}),
                        'lora_row_partial_reduce':('BOOLEAN',{'default':False,'tooltip':'Approximate: reduce row-LoRA A partials instead of gathering the full input.'}),
                        'te_attention':(['tilelang_group32','native'],{'default':'tilelang_group32'}),
                        'dit_attention':(['kitchen','tilelang_drop','tilelang_scalar','tilelang_dense','native'],{'default':'kitchen'}),
                        'row_linear_backend':(['tilelang_packed','tilelang_scaled_fp16','native'],{'default':'tilelang_packed','tooltip':'Scaled FP16 is approximate; packed retains native INT32 SUM.'}),
                        'swiglu_padding':('BOOLEAN',{'default':True}),
                        'selective_b':('BOOLEAN',{'default':True,'tooltip':'Selected-B FC1 GEMM for the validated H3 rank128 LoRA geometry; other shapes use full B.'}),
                        'scalar_score_workspace_mib':('INT',{'default':96,'min':0,'max':512,'step':1,'tooltip':'Extra score workspace cap per rank. Set 0 for inline scalar summaries.'}),
                        'adaptive_lookahead':('BOOLEAN',{'default':False,'tooltip':'Adapt the GPU weight window between 1 and 2 layers using VRAM headroom.'}),
                        'fc2_quantizer':(['owned_fp32','native'],{'default':'owned_fp32','tooltip':'owned_fp32 reduces FC2 activation memory while retaining global row scales and INT32 SUM. Tested outputs are byte-exact; GEMM geometry differs. Applies to packed row-linear with SwiGLU padding; other backends keep their own path.'}),
                        'lora_mode':(['online','offline'],{'default':'online','tooltip':'Offline bakes whole layers on up to two GPUs, caches a new checkpoint, then runs without online LoRA. Base weights are never modified.'})}}
    RETURN_TYPES = ('MODEL', 'CLIP', 'MINIMAX_TP2_SESSION')
    FUNCTION = 'load'
    CATEGORY = 'MiniMax H3/TP2'

    @classmethod
    def IS_CHANGED(cls,**kwargs):
        return tuple((p,Path(p).stat().st_size,Path(p).stat().st_mtime_ns) for p,_ in kwargs.get('lora_stack',()))

    def load(self, text_encoder, diffusion_model, gpu_weight_window_gib=6., ram_cache_per_rank_gib=3.,
             idle_residency=False,idle_weight_gib=4.,lora_stack=(),tp_quantization='scalemax',
             lora_row_partial_reduce=False,lora_concat_a=False,te_attention='tilelang_group32',
             dit_attention='kitchen',row_linear_backend='tilelang_packed',swiglu_padding=True,
             selective_b=True,scalar_score_workspace_mib=96,adaptive_lookahead=False,fc2_quantizer='owned_fp32',lora_mode='online'):
        global _service
        if lora_mode not in ('online','offline'):raise ValueError('Unknown LoRA loading mode')
        if lora_mode=='offline' and lora_stack:
            from patches.legacy_runtime import validate_imported_source
            validate_imported_source()
            from patches.offline_lora import cached_bake
            with _lock:
                def release_active():
                    global _service
                    if _service is not None:_service.close();_service=None
                    from patches.comfy_sampler import reclaim_parent_models
                    import torch
                    for index in range(min(2,torch.cuda.device_count())):reclaim_parent_models(torch.device('cuda',index))
                diffusion_model=cached_bake(diffusion_model,tuple(lora_stack),before_bake=release_active)
                lora_stack=()  # The new checkpoint already contains every adapter.
        config = RuntimeConfig(str(Path(text_encoder).resolve()), str(Path(diffusion_model).resolve()),
            weight_bytes=int(gpu_weight_window_gib*GiB), cache_bytes=int(ram_cache_per_rank_gib*GiB),
            idle_residency=bool(idle_residency),prefetch_lifecycle=bool(idle_residency),
            idle_weight_bytes=int(idle_weight_gib*GiB),lora_stack=tuple(lora_stack),
            tp_quantization=tp_quantization,lora_row_partial_reduce=bool(lora_row_partial_reduce),
            lora_concat_a=bool(lora_concat_a),
            adaptive_lookahead=bool(adaptive_lookahead),
            operators=H3OperatorPolicy(
                te_attention=te_attention,
                dit_attention=dit_attention,
                row_linear_backend=row_linear_backend,
                swiglu_padding=bool(swiglu_padding),
                selective_b=bool(selective_b),
                scalar_score_workspace_mib=int(scalar_score_workspace_mib),
                fc2_quantizer=fc2_quantizer))
        with _lock:
            if _service is None or _service.closed or _service.config != config:
                if _service is not None:
                    _service.close()
                _service = PhysicalTPService(config)
            return create_model(_service), create_clip(_service), _service


class MiniMaxH3TP2LoRAStack:
    DESCRIPTION='Native-ordered online multi-LoRA by default. The loader optionally enables approximate RowParallel A partial reduction. Concat A is deprecated. B projection and scale/add order remain separate.'
    @classmethod
    def INPUT_TYPES(cls):
        return {'required':{'lora_path':('STRING',{'default':''}),
                            'strength':('FLOAT',{'default':1.,'min':-20.,'max':20.,'step':.05})},
                'optional':{'previous':('MINIMAX_TP2_LORAS',)}}
    RETURN_TYPES=('MINIMAX_TP2_LORAS',)
    FUNCTION='append'
    CATEGORY='MiniMax H3/TP2'

    def append(self,lora_path,strength,previous=()):
        path=Path(lora_path).resolve()
        if not path.is_file():raise FileNotFoundError(path)
        return (tuple(previous)+((str(path),float(strength)),),)


class MiniMaxH3PostDecodePrefetch:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required':{'vae':('VAE',), 'session':('MINIMAX_TP2_SESSION',)}}
    RETURN_TYPES = ('VAE',)
    FUNCTION = 'wrap'
    CATEGORY = 'MiniMax H3/TP2'

    def wrap(self, vae, session):
        return (DecodeHandoff(vae, session),)


class MiniMaxH3ParallelVAELoader:
    @classmethod
    def INPUT_TYPES(cls):
        import folder_paths
        return {'required':{
            'vae_name':(folder_paths.get_filename_list('vae'),),
            'session':('MINIMAX_TP2_SESSION',),
            'decode_mode':(['auto', 'native'], {'default':'auto'})}}
    RETURN_TYPES = ('VAE',)
    FUNCTION = 'load'
    CATEGORY = 'MiniMax H3/TP2'

    def load(self, vae_name, session, decode_mode='auto'):
        import folder_paths
        import nodes
        # Use the exact same file for Native encoding/public decode and ranks.
        path = folder_paths.get_full_path_or_raise('vae', vae_name)
        from patches.staged_vae import StagedVAE,checkpoint_kind
        from patches.windowed_tp import CheckpointReader
        if checkpoint_kind(CheckpointReader(path)) is not None:
            vae=StagedVAE(path,session)
        else:
            vae, = nodes.VAELoader().load_vae(vae_name)
        return (ParallelDecodeHandoff(vae, session, path, decode_mode),)


NODE_CLASS_MAPPINGS = {'MiniMaxH3TP2Loader':MiniMaxH3TP2Loader,
    'MiniMaxH3TP2LoRAStack':MiniMaxH3TP2LoRAStack,
    'MiniMaxH3PostDecodePrefetch':MiniMaxH3PostDecodePrefetch,
    'MiniMaxH3ParallelVAELoader':MiniMaxH3ParallelVAELoader}
