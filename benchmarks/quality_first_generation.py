"""Bounded native-architecture T4 quality screening, not TP performance proof.

Reuse actual ComfyUI REF2VA entrypoints. One-device streamed loading preserves
the trained models and is explicitly excluded from deployment speed claims.
"""
import argparse
from contextlib import contextmanager
import json
import math
import os
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'benchmarks')]
import capture_real_attention as capture


class QualityProbe(capture.AttentionCapture):
    def __init__(self,config,output,mode='native',policy=None):
        super().__init__(config,output,mode,policy)
        self.selection=config['attention_policy'];self.densities=[];self.checked_masks=set()
        self.layouts={};self.trace_active=False;self.specializations=[];self.specialized=set()

    def specialize(self,query):
        if self.component!='dit' or self.layer<0:return
        setting=self.selection.get('dit',{});mode=setting.get('mode','native')
        if self.layer in setting.get('dense_layers',[]) or self.call in setting.get('dense_calls',[]):return
        key=(mode,tuple(query.shape),str(query.device),setting.get('kernel_variant'))
        if mode=='native' or key in self.specialized:return
        start=time.perf_counter();b,h,n,d=query.shape
        if mode=='sage32':
            from patches.tilelang_sage_preparation import quant_kernel
            from patches.tilelang_sparse_attention import get_kernel
            quant_kernel(b,h,n,d,32,False);quant_kernel(b,h,n,d,64,True)
            get_kernel(b,h,n,d,False,False,sm=75,causal=False)
        elif mode in ('group32_fused','group64_fused'):
            from patches.tilelang_grouped_fused import quantizer,attention_kernel
            group=int(mode.split('_')[0].removeprefix('group'))
            quantizer(b,h,n,d,group,False);quantizer(b,h,n,d,group,True)
            if group==32 and setting.get('kernel_variant')=='legacy_mma':
                from patches.tilelang_grouped_qk import grouped_kernel
                grouped_kernel(b,h,n,d,False)
            else:attention_kernel(b,h,n,d,group,False)
        elif mode=='drop':
            from patches.tilelang_sparse_attention import get_kernel
            get_kernel(b,h,n,d,False,False,sm=75,causal=False)
        else:return
        self.specialized.add(key);self.specializations.append({'mode':mode,'shape':list(query.shape),'device':str(query.device),
            'factory_wall_seconds':time.perf_counter()-start,'scope':'TileLang factory/JIT before API timing; first execution/module loading may remain in call0'})

    def phase(self,name,fn):
        if not self.config.get('profile_attention',False):return fn()
        import torch
        torch.cuda.nvtx.range_push(name)
        try:return fn()
        finally:torch.cuda.nvtx.range_pop()

    @contextmanager
    def scope(self,component,layer=-1):
        import torch
        with super().scope(component,layer):
            selected=self.config.get('nsys_denoise_step')
            if component=='dit' and self.call==selected and layer==0:
                torch.cuda.profiler.start();self.trace_active=True
            try:yield
            finally:
                if component=='dit' and self.call==selected and layer==49 and self.trace_active:
                    for device in range(torch.cuda.device_count()):torch.cuda.synchronize(device)
                    torch.cuda.profiler.stop();self.trace_active=False

    def wrap(self,original):
        def attention(q,k,v,heads,*args,**kwargs):
            import torch
            role=self.component;setting=self.selection.get(role,{'mode':'native'})
            selected=setting.get('mode','native')
            if self.layer<0 or self.layer in setting.get('dense_layers',[]) or (role=='dit' and self.call in setting.get('dense_calls',[])):
                selected='native'
            if selected=='native':return original(q,k,v,heads,*args,**kwargs)
            tensors=[x.peek() if hasattr(x,'peek') else x for x in (q,k,v)]
            if not kwargs.get('skip_reshape') or tensors[0].ndim!=4:raise ValueError('Native BHND boundary required')
            mask=kwargs.get('mask',args[0] if args else None)
            causal=role=='te';gqa=kwargs.get('enable_gqa',False)
            # Fail closed: the supported TE workload has a lower-triangular mask.
            # Its exact equality is independently qualified before candidate runs.
            if role!='te' and mask is not None:raise ValueError('Arbitrary masks unsupported')
            if causal and tensors[0].shape not in self.checked_masks:
                n=tensors[0].shape[-2]
                if mask is None or not torch.equal(mask.reshape(-1,n,n)>=0,torch.ones_like(mask.reshape(-1,n,n),dtype=torch.bool).tril()):
                    raise ValueError('TE mask is not the qualified causal mask')
                self.checked_masks.add(tensors[0].shape)
            with torch.cuda.device(tensors[0].device),torch.inference_mode():
                if selected=='fp16':
                    out=torch.nn.functional.scaled_dot_product_attention(*[x.half() for x in tensors],is_causal=causal,enable_gqa=gqa)
                elif selected=='sage32':
                    from patches.tilelang_sage_preparation import prepare_sage_fused
                    from patches.tilelang_dense_preparation import run_prevalidated_dense
                    half=self.phase('input_cast',lambda:[x.half() for x in tensors])
                    prepared,route=self.phase('sage32/QK_quantization',lambda:prepare_sage_fused(*half,enable_gqa=gqa))
                    out=self.phase('sage32/QK_softmax_PV',lambda:run_prevalidated_dense(prepared,route,causal=causal))
                elif selected in ('group32_fused','group64_fused'):
                    from patches.tilelang_grouped_fused import prepare_grouped_fused,execute_grouped_fused
                    if role!='dit':raise ValueError('New grouped generation validation is scoped to DiT')
                    group=int(selected.split('_')[0].removeprefix('group'))
                    half=self.phase('input_cast',lambda:[x.half() for x in tensors])
                    prepared=self.phase(selected+'/QK_quantization',lambda:prepare_grouped_fused(*half,group=group))
                    if group==32 and setting.get('kernel_variant')=='legacy_mma':
                        from patches.tilelang_grouped_qk import grouped_kernel
                        out=self.phase(selected+'/QK_softmax_PV',lambda:grouped_kernel(*tensors[0].shape,causal)(prepared.qi,prepared.ki,prepared.v,prepared.qs,prepared.ks)[0][:,:,:prepared.tokens])
                    else:out=self.phase(selected+'/QK_softmax_PV',lambda:execute_grouped_fused(prepared,group,causal))
                    if self.config.get('qualify_grouped',False) and self.call==0 and self.layer==0:
                        qualification_start=time.perf_counter()
                        from dataclasses import replace
                        from patches.tilelang_grouped_fused import dequantized_reference
                        from attention_root_causes import metrics
                        ids=torch.linspace(0,tensors[0].shape[1]-1,min(3,tensors[0].shape[1]),device=tensors[0].device).round().long()
                        small=replace(prepared,**{key:value[:,ids].contiguous() for key,value in prepared.__dict__.items()
                            if isinstance(value,torch.Tensor) and value.ndim>=2 and value.shape[1]==tensors[0].shape[1]})
                        error=metrics(out[:,ids],dequantized_reference(small,group))
                        if max(error['relative_l2'])>=.003:raise ValueError('Actual-input grouped kernel/reference mismatch: '+str(error))
                        (self.output/'grouped_qualification.json').write_text(json.dumps({'implementation':error,'reference_heads':ids.cpu().tolist(),
                            'wall_seconds':time.perf_counter()-qualification_start,
                            'scope':'First real layer input on cu130, not generation quality acceptance'},indent=2))
                elif selected=='group32':
                    from patches.tilelang_grouped_qk import prepare_grouped,execute_grouped
                    out=execute_grouped(prepare_grouped(*[x.half() for x in tensors],enable_gqa=gqa),causal)[0]
                elif selected in ('dense_int8','dense_pv8'):
                    from patches.tilelang_dense_preparation import dense_attention
                    out=dense_attention(*[x.half() for x in tensors],causal=causal,enable_gqa=gqa,int8_pv=selected=='dense_pv8')
                elif selected in ('drop','original_pool','scalar_pool'):
                    if role!='dit':raise ValueError('Sparse screening only for DiT')
                    from patches.tilelang_sparse_attention import prepare,make_routes,execute
                    from patches.tilelang_scalar_pool import execute_scalar_pool
                    half=self.phase('input_cast',lambda:[x.half() for x in tensors])
                    p=self.phase(selected+'/legacy_preparation',lambda:prepare(*half))
                    route=self.phase(selected+'/routing',lambda:make_routes(p,setting.get('routing','sol'),tau=setting.get('tau',0.),density=setting.get('density',.5)))
                    # Native layout may use a different temporal conversion; supplied
                    # capture-derived prefix is mandatory until layout is bound.
                    layout=kwargs.get('transformer_options',{}).get('minimax_h3_layout')
                    if layout is None:raise ValueError('Native PackedLayout required for sparse generation')
                    video=[(a,b) for a,b,kind in layout.segments if kind=='video']
                    if len(video)!=1 or video[0][1]!=layout.seq_len or layout.seq_len!=tensors[0].shape[2]:raise ValueError('Native packed video suffix is not contiguous')
                    prefix=video[0][0]
                    if 'prefix_tokens' in setting and prefix!=setting['prefix_tokens']:raise ValueError('Packed prefix changed')
                    if 'expected_tokens' in setting and tensors[0].shape[2]!=setting['expected_tokens']:raise ValueError('Packed sequence binding differs from captured workload')
                    self.layouts[str(layout.seq_len)]={'segments':layout.segments,'Q_tokens':tensors[0].shape[2],'K_tokens':tensors[1].shape[2],
                        'heads':heads,'head_dim':tensors[0].shape[-1],'mandatory_prefix_tokens':prefix,'mandatory_prefix_blocks':math.ceil(prefix/64)}
                    if not 0<prefix<tensors[0].shape[2]:raise ValueError('Packed prefix binding invalid')
                    blocks=math.ceil(prefix/64);route[...,:blocks]=1;route[...,:blocks,:]=1
                    self.densities.append((dict(layer=self.layer,call=self.call),route.float().mean().detach()))
                    out=self.phase(selected+'/public_validation_QK_softmax_PV_LSE',lambda:execute_scalar_pool(p,route)[0] if selected=='scalar_pool' else execute(p,route,pooled=selected=='original_pool')[0])
                else:raise ValueError('Unknown candidate '+selected)
                out=out.to(tensors[0].dtype)
            for tensor in (q,k,v):
                if hasattr(tensor,'peek'):tensor.take()
            return out if kwargs.get('skip_output_reshape') else out.transpose(1,2).reshape(out.shape[0],out.shape[2],-1)
        if not self.config.get('profile_attention',False):return attention
        profiled=super().wrap(attention)
        def annotated(*a,**kw):
            import torch
            layout=kw.get('transformer_options',{}).get('minimax_h3_layout')
            if self.component=='dit' and layout is not None:
                self.layouts[str(layout.seq_len)]={'segments':layout.segments,'sequence_tokens':layout.seq_len,'signature':layout.signature}
            query=a[0].peek() if hasattr(a[0],'peek') else a[0]
            # CUDA events must be recorded on the tensor's device, including
            # native attention before its internal tensor-device guards.
            with torch.cuda.device(query.device):
                self.specialize(query)
                return self.phase(f'{self.component}/layer{self.layer}/step{self.call}/Attention_API',lambda:profiled(*a,**kw))
        return annotated

    def finish(self):
        import torch
        for device in range(torch.cuda.device_count()):torch.cuda.synchronize(device)
        (self.output/'packed_layouts.json').write_text(json.dumps(self.layouts,indent=2))
        (self.output/'denoiser_timings.json').write_text(json.dumps(self.model_calls,indent=2))
        (self.output/'compilation.json').write_text(json.dumps(self.specializations,indent=2))
        if self.densities:
            # Serial layer placement produces scalars on both CUDA devices.
            # Consolidate after sampling; do not stack tensors across devices.
            values=torch.stack([value.cpu() for _,value in self.densities]).tolist()
            (self.output/'routing.json').write_text(json.dumps([meta|{'selected_density':value} for (meta,_),value in zip(self.densities,values)]))
        super().finish()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,required=True);args=parser.parse_args()
    config=json.loads(args.config.read_text());os.environ['COMFY_ROOT']=config['comfy_root']
    sys.path.insert(0,config['comfy_root'])
    import torch
    assert torch.cuda.device_count()==2 and all(torch.cuda.get_device_capability(i)==(7,5) for i in range(2)), 'Physical 2×T4 allocation required'
    assert not any(config['capture_layers'].values()),'No new QKV dumps in quality study'
    assert config.get('save_denoiser_tensors') is False and config.get('save_pixel_tensor') is False
    capture.AttentionCapture=QualityProbe
    # All modes use identical seeded noise generation and full native conditioning.
    # No candidate output is required to equal dense trajectories or pixels.
    capture.main(args.config,mode='native')
    if config.get('prepare_inputs_only',False):return
    import comfy.sd
    from patches.windowed_tp import CheckpointReader
    import av
    import numpy as np
    out=Path(config['output_directory'])
    audio_path=Path(config['model_directory'])/'minimax_h3_audio_vae_fp32_clean.safetensors'
    vae=comfy.sd.VAE(sd=CheckpointReader(audio_path).read(''),device=torch.device('cuda:0'),dtype=torch.float32)
    latent=torch.load(out/'native_audio_latent.pt',map_location='cpu',weights_only=True)
    with torch.inference_mode():waveform=vae.decode(latent).cpu()
    a=waveform[0] if waveform.ndim==3 else waveform
    if a.shape[0]>a.shape[1]:a=a.T
    with av.open(str(out/'audio.wav'),'w') as container:
        stream=container.add_stream('pcm_s16le',rate=32000);stream.layout='stereo'
        array=(a.clamp(-1,1)*32767).short().T.reshape(1,-1).numpy()
        frame=av.AudioFrame.from_ndarray(array,format='s16',layout='stereo');frame.sample_rate=32000
        for packet in stream.encode(frame):container.mux(packet)
        for packet in stream.encode():container.mux(packet)
    summary={'quality_status':'requires_metrics_and_blinded_review','performance_scope':config.get('dit_loading','streamed')+' correctness/quality run, not TP2 deployment',
        'audio_samples':a.shape[-1],'audio_seconds':a.shape[-1]/32000,'audio_absmax':a.abs().max().item(),
        'audio_rms':a.float().square().mean().sqrt().item(),'attention_policy':config['attention_policy']}
    (out/'quality_screen.json').write_text(json.dumps(summary,indent=2))


if __name__=='__main__':main()
