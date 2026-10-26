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
from patches.h3_tensor_flow import TensorFlowPolicy
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
                        'idle_weight_gib':('FLOAT',{'default':4.,'min':.5,'max':8.,'step':.5}),
                        'lora_row_partial_reduce':('BOOLEAN',{'default':True,'tooltip':'Default: project owned RowParallel LoRA A columns locally, then FP32 SUM the low-rank hidden state. Approximate reduction order; disable to gather the full input.'}),
                        'te_attention':(['kitchen_causal','tilelang_group32','native'],{'default':'kitchen_causal','tooltip':'Default: pinned Kitchen INT8 with the complete causal/padding/additive mask. Direct dispatch avoids GPU mask-classification overhead.'}),
                        'dit_attention':(['kitchen','tilelang_drop','tilelang_scalar','tilelang_dense','native'],{'default':'kitchen'}),
                        'row_linear_backend':(['tilelang_packed','tilelang_scaled_fp16','native'],{'default':'tilelang_scaled_fp16','tooltip':'Default: approximate power-of-two scaled FP16 SUM with fused FP32 dequantization. Select tilelang_packed for INT32 SUM.'}),
                        'scalar_score_workspace_mib':('INT',{'default':96,'min':0,'max':512,'step':1,'tooltip':'Extra score workspace cap per rank. Set 0 for inline scalar summaries.'}),
                        'adaptive_lookahead':('BOOLEAN',{'default':True,'tooltip':'Adapt the GPU weight window between 1 and 2 layers using VRAM headroom.'}),
                        'fc2_quantizer':(['owned_fp32','native','fused_stored','fused_recompute'],{'default':'fused_stored','tooltip':'Default: FP32 SwiGLU + regular ConvRot256 fusion, retaining rotated values until shared-MAX INT8 quantization. Approximate FP32 summation order. recompute saves memory; owned_fp32/native retain prior preparation. Applies to TE down and DiT FC2.'}),
                        'lora_mode':(['offline','online'],{'default':'offline','tooltip':'Default offline: TP2 JSON merges bounded blocks on CPU; safetensors creates a baked checkpoint. Prefer TP2 bundles for long sequences. Online is for fitting short/medium requests.'}),
                        'tensor_flow':(['fp16','legacy'],{'default':'fp16','tooltip':'Default FP16-fix uses scaled residual streams and FP32 statistics/time/head islands. Audio VAE stays FP32. Select legacy for the original FP32 flow.'}),
                        'residual_exponent':('INT',{'default':6,'min':0,'max':12,'tooltip':'TE/DiT FP16 residual divisor is 2**exponent. Default: 64.'}),
                        'vision_residual_exponent':('INT',{'default':0,'min':0,'max':12,'tooltip':'Optional vision gauge. Default 0 retains the Native FP16 vision arithmetic.'}),
                        'video_residual_exponent':('INT',{'default':0,'min':0,'max':12,'tooltip':'Optional Video VAE decoder gauge. Default 0 preserves its existing FP16 arithmetic.'}),
                        'video_decode_reserve_gib':('FLOAT',{'default':2.,'min':0.,'max':12.,'step':.5,'tooltip':'Extra VRAM headroom for Video query-parallel admission, separate from the DiT sampling reserve.'}),
                        'prefetch_lifecycle':('BOOLEAN',{'default':True,'tooltip':'Enable reuse-aware scheduling independently of idle speculation. Idle residency also enables this policy.'}),
                        'memory_pressure_aware':('BOOLEAN',{'default':True,'tooltip':'Use available host/cgroup headroom as well as the explicit RAM budget.'}),
                        'dit_reserve_gib':('FLOAT',{'default':9.,'min':0.,'max':12.,'step':.5,'tooltip':'Historical long-sequence headroom; also applies to TE. Tune separately for shorter workloads. Video decode uses its own reserve.'}),
                        'command_timeout_seconds':('INT',{'default':1500,'min':1,'max':3600,'tooltip':'Per-command worker deadline, including cold weight preparation.'}),
                        'rank_checkpoint_dir':('STRING',{'default':'','tooltip':'Optional verified contiguous rank-block directory; empty retains ordinary checkpoint reads.'}),
                        'lora_selective_qkv_b':('BOOLEAN',{'default':True,'tooltip':'Owned QKV B for supported online rank128 adapters; byte-exact and faster on tested T4 fixtures. Inactive offline; unsupported shapes retain full B. Not long-sequence online admission.'}),
                        'column_linear_backend':(['cuda_int8_tilelang','preallocated'],{'default':'cuda_int8_tilelang','tooltip':'Measured direct INT8 GEMM + ordered TileLang epilogue on aligned SM75 DiT columns. Other shapes/devices fall back. Preallocated remains the fallback/control.'})}}
    RETURN_TYPES = ('MODEL', 'CLIP', 'MINIMAX_TP2_SESSION')
    FUNCTION = 'load'
    CATEGORY = 'MiniMax H3/TP2'

    @classmethod
    def IS_CHANGED(cls,**kwargs):
        stack=kwargs.get('lora_stack',()) or ()
        return tuple((p,float(s),Path(p).stat().st_size,Path(p).stat().st_mtime_ns) if Path(p).is_file() else (p,float(s),0,0) for p,s in stack)

    def load(self, text_encoder, diffusion_model, gpu_weight_window_gib=6., ram_cache_per_rank_gib=3.,
             idle_residency=False,idle_weight_gib=4.,lora_stack=(),
             lora_row_partial_reduce=True,te_attention='kitchen_causal',
             dit_attention='kitchen',row_linear_backend='tilelang_scaled_fp16',
             scalar_score_workspace_mib=96,adaptive_lookahead=True,fc2_quantizer='fused_stored',lora_mode='offline',tensor_flow='fp16',residual_exponent=6,
             vision_residual_exponent=0,video_residual_exponent=0,video_decode_reserve_gib=2.,
             prefetch_lifecycle=True,memory_pressure_aware=True,dit_reserve_gib=9.,
             command_timeout_seconds=1500,rank_checkpoint_dir='',
             lora_selective_qkv_b=True,column_linear_backend='cuda_int8_tilelang',**retired_options):
        global _service
        from patches.h3_configuration import check_retired_loader_options
        check_retired_loader_options(retired_options)
        if column_linear_backend=='tilelang_exact':column_linear_backend='cuda_int8_tilelang'
        if lora_mode not in ('online','offline'):raise ValueError('Unknown LoRA loading mode')
        bundle_offline=lora_mode=='offline' and Path(diffusion_model).suffix.lower()=='.json'
        if lora_mode=='offline' and lora_stack and not bundle_offline:
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
            idle_residency=bool(idle_residency),prefetch_lifecycle=bool(prefetch_lifecycle or idle_residency),
            memory_pressure_aware=bool(memory_pressure_aware),reserve_bytes=int(dit_reserve_gib*GiB),
            timeout_seconds=int(command_timeout_seconds),
            rank_checkpoint_dir=str(Path(rank_checkpoint_dir).resolve()) if rank_checkpoint_dir else None,
            lora_selective_qkv_b=bool(lora_selective_qkv_b),
            idle_weight_bytes=int(idle_weight_gib*GiB),lora_stack=tuple(lora_stack),
            bundle_offline_lora=bundle_offline,
            lora_row_partial_reduce=bool(lora_row_partial_reduce),
            adaptive_lookahead=bool(adaptive_lookahead),
            video_decode_reserve_bytes=int(video_decode_reserve_gib*GiB),
            tensor_flow=TensorFlowPolicy(tensor_flow,int(residual_exponent),int(vision_residual_exponent),int(video_residual_exponent)),
            operators=H3OperatorPolicy(
                te_attention=te_attention,
                dit_attention=dit_attention,
                row_linear_backend=row_linear_backend,
                scalar_score_workspace_mib=int(scalar_score_workspace_mib),
                fc2_quantizer=fc2_quantizer,column_linear_backend=column_linear_backend))
        with _lock:
            if _service is None or _service.closed or _service.config != config:
                if _service is not None:
                    _service.close()
                _service = PhysicalTPService(config)
            return create_model(_service), create_clip(_service), _service


class MiniMaxH3TP2LoRAStack:
    DESCRIPTION='Ordered multi-LoRA stack. Default loader uses offline baking (bounded CPU block merges for TP2 bundles). Optional short/medium online mode uses RowParallel A partial reduction with FP32 hidden SUM and guarded owned B rows. Adapter order is preserved.'
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
    DESCRIPTION='Demand-loaded H3 VAEs: encoders load only for reference encoding, decoders only for output. Video supports query parallelism; Audio decodes natively on one GPU in FP32.'
    @classmethod
    def INPUT_TYPES(cls):
        import folder_paths
        return {'required':{
            'vae_name':(folder_paths.get_filename_list('vae'),),
            'session':('MINIMAX_TP2_SESSION',),
            'decode_mode':(['auto', 'native'], {'default':'auto','tooltip':'Video: auto enables query parallelism when compatible. Audio always uses Native single-GPU decoding.'})}}
    RETURN_TYPES = ('VAE',)
    FUNCTION = 'load'
    CATEGORY = 'MiniMax H3/TP2'

    def load(self, vae_name, session, decode_mode='auto'):
        import folder_paths
        # Only video shares its checkpoint with decoder ranks.
        path = folder_paths.get_full_path_or_raise('vae', vae_name)
        from patches.staged_vae import StagedVAE,checkpoint_kind
        from patches.windowed_tp import CheckpointReader
        kind=checkpoint_kind(CheckpointReader(path))
        if kind is not None:
            vae=StagedVAE(path,session)
        else:
            import nodes
            vae, = nodes.VAELoader().load_vae(vae_name)
        return (ParallelDecodeHandoff(vae, session, path, 'native' if kind=='audio' else decode_mode),)


WEB_DIRECTORY = './web'

NODE_CLASS_MAPPINGS = {'MiniMaxH3TP2Loader':MiniMaxH3TP2Loader,
    'MiniMaxH3TP2LoRAStack':MiniMaxH3TP2LoRAStack,
    'MiniMaxH3PostDecodePrefetch':MiniMaxH3PostDecodePrefetch,
    'MiniMaxH3ParallelVAELoader':MiniMaxH3ParallelVAELoader}
