"""Locate audio channel-split roundoff using trained Native operators only."""
import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
for path in (ROOT,ROOT/'ComfyUI',ROOT/'.retest-deps'):
    sys.path.insert(0,str(path))
import torch
from safetensors.torch import load_file
from comfy.ldm.minimax.audio_vae import MiniMaxH3AudioVAE


def metric(a,b):
    a=a.detach().cpu().double();b=b.detach().cpu().double();d=a-b
    return dict(bit_exact=torch.equal(a,b),rel_l2=float(d.norm()/b.norm().clamp_min(1e-30)),
                max_abs=float(d.abs().max()),changed_elements=int((d!=0).sum()),elements=d.numel())


@contextmanager
def flags(**values):
    original={key:getattr(torch.backends.cudnn,key) for key in values}
    try:
        for key,value in values.items():setattr(torch.backends.cudnn,key,value)
        yield
    finally:
        for key,value in original.items():setattr(torch.backends.cudnn,key,value)


def run_capture(model,z,names):
    saved={};hooks=[]
    for name in names:
        def hook(module,args,out,name=name):
            # Clone before later residual branches mutate their inputs/outputs.
            saved[name]=(args[0].detach().clone().cpu(),out.detach().clone().cpu())
        hooks.append(model.get_submodule(name).register_forward_hook(hook))
    try:
        with torch.inference_mode():out=model.decode(z).cpu()
    finally:
        for handle in hooks:handle.remove()
    return out,saved


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,default=Path('D:/minimaxH3/minimax_h3_audio_vae_fp32_clean.safetensors'))
    p.add_argument('--output',type=Path,default=ROOT/'semantic_results/audio_channel_diagnosis.json')
    p.add_argument('--references',type=Path)
    options=p.parse_args();torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=False
    model=MiniMaxH3AudioVAE();model.load_state_dict(load_file(str(options.checkpoint)),strict=True)
    model=model.to('cuda:0',dtype=torch.float32).eval().requires_grad_(False)
    result=dict(physical_gpu_count=torch.cuda.device_count(),gpu=torch.cuda.get_device_name(),
                torch=torch.__version__,cudnn=torch.backends.cudnn.version(),
                scope='trained audio decoder, single-GPU batch/layout/rounding diagnosis, not T4 performance',
                tf32_disabled=True,checkpoint_sha256=hashlib.sha256(options.checkpoint.read_bytes()).hexdigest(),
                source_sha256={f.relative_to(ROOT).as_posix():hashlib.sha256(f.read_bytes()).hexdigest() for f in (Path(__file__),ROOT/'ComfyUI/comfy/ldm/minimax/audio_vae.py')},
                cases=[])
    def save():
        options.output.parent.mkdir(parents=True,exist_ok=True)
        options.output.write_text(json.dumps(result,indent=2))
    names=['dec_in_proj','decoder.conv_pre','decoder.ups.0.0',
           'decoder.resblocks.0.activations.0.upsample','decoder.resblocks.0.activations.0.act',
           'decoder.resblocks.0.activations.0.downsample','decoder.resblocks.0.convs1.0',
           'decoder.resblocks.0.convs2.0','decoder.resblocks.0']
    names += ['decoder.resblocks.'+str(i*3+2) for i in range(7)]
    names += ['decoder.activation_post','decoder.conv_post']
    for index,seconds in enumerate((5,10,20)):
        cpu=(torch.load(options.references/f'input_{seconds}s.pt',weights_only=True,map_location='cpu') if options.references
             else torch.randn((1,32,2,seconds*40),generator=torch.Generator().manual_seed(300+index),dtype=torch.float32))
        digest=hashlib.sha256(cpu.numpy().tobytes()).hexdigest()
        history=ROOT/'kaggle_retest_output/cu130/retest_results/round_0/audio/results.json'
        matched=digest==json.loads(history.read_text())['workloads'][str(seconds)+'s']['input_sha256'] if history.exists() else None
        z=cpu.cuda();ref,native=run_capture(model,z,names if seconds==5 else [])
        mono=[];captures=[]
        for channel in (0,1):
            out,cap=run_capture(model,z[:,:,channel:channel+1],names if seconds==5 else [])
            mono.append(out);captures.append(cap)
        split=torch.cat(mono,1);row=dict(seconds=seconds,input_sha256=digest,historical_input_hash_matches=matched,
                                         sequential_split_vs_native=metric(split,ref),stages=[])
        if options.references:
            old_ref=torch.load(options.references/f'native_{seconds}s_output.pt',weights_only=True)
            old_split=torch.load(options.references/f'channel_2gpu_{seconds}s_output.pt',weights_only=True)
            row['saved_physical_t4_split_vs_native']=metric(old_split,old_ref)
            del old_ref,old_split
        if seconds==5:
            for name in names:
                for channel in (0,1):
                    ni,no=native[name];mi,mo=captures[channel][name]
                    row['stages'].append(dict(module=name,channel=channel,input=metric(mi,ni[channel:channel+1]),
                                              output=metric(mo,no[channel:channel+1]),native_shape=list(ni.shape),mono_shape=list(mi.shape)))
            # Counterfactual one-operation replay: same input VALUES, change
            # only batch dimension/strides; replay error is not propagated input.
            row['operator_replays']=[]
            for name in ('dec_in_proj','decoder.conv_pre','decoder.ups.0.0',
                         'decoder.resblocks.0.activations.0.upsample','decoder.resblocks.0.convs1.0'):
                tensor=native[name][0].cuda();module=model.get_submodule(name)
                with torch.inference_mode():
                    both=module(tensor).cpu()
                    one=torch.cat([module(tensor[ch:ch+1].contiguous()).cpu() for ch in (0,1)],0)
                    repeat=module(tensor).cpu()
                row['operator_replays'].append(dict(module=name,same_input_batch_split=metric(one,both),
                                                     same_shape_repeat=metric(repeat,both),native_input_stride=list(tensor.stride())))
                if name=='decoder.conv_pre':
                    folder=options.output.parent/'audio_conv_pre_traces';folder.mkdir(exist_ok=True)
                    row['first_conv_kernel_plans']={}
                    for label,value in (('native_batch2',tensor),('mono_batch1',tensor[:1].contiguous())):
                        with torch.inference_mode(),torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],record_shapes=True) as prof:
                            replay=module(value);torch.cuda.synchronize()
                        path=folder/(label+'.json.gz');prof.export_chrome_trace(str(path))
                        import gzip
                        with gzip.open(path,'rt') as stream:events=json.load(stream)['traceEvents']
                        row['first_conv_kernel_plans'][label]=[dict(name=e['name'],grid=e.get('args',{}).get('grid'),block=e.get('args',{}).get('block')) for e in events if e.get('cat')=='kernel']
                        del replay
            del native,captures
        original=model.decoder.conv_pre.forward
        def native_batch_conv(x):
            doubled=torch.cat([x,torch.zeros_like(x)],0)
            return original(doubled)[:x.shape[0]]
        model.decoder.conv_pre.forward=native_batch_conv
        try:
            with torch.inference_mode():
                first_fixed=torch.cat([model.decode(z[:,:,ch:ch+1]).cpu() for ch in (0,1)],1)
        finally:model.decoder.conv_pre.forward=original
        row['only_first_conv_native_batch_vs_native']=metric(first_fixed,ref)
        del first_fixed
        # Same-shape control: preserve Native's full batch, replace the unused
        # stereo channel with zeros, then select only the owned output channel.
        # This duplicates work; it is a causal control, not an optimization.
        padded=[]
        with torch.inference_mode():
            for channel in (0,1):
                value=z.clone();value[:,:,1-channel].zero_()
                padded.append(model.decode(value)[:,channel:channel+1].cpu())
        row['native_shape_dummy_channel_control']=metric(torch.cat(padded,1),ref)
        # Determinism is tested against its OWN Native output, and separately
        # against the original Native contract; it need not preserve that contract.
        with flags(deterministic=True):
            with torch.inference_mode():
                deterministic_ref=model.decode(z).cpu()
                deterministic_split=torch.cat([model.decode(z[:,:,ch:ch+1]).cpu() for ch in (0,1)],1)
        row['deterministic_split_vs_its_native']=metric(deterministic_split,deterministic_ref)
        row['deterministic_native_vs_original_native']=metric(deterministic_ref,ref)
        result['cases'].append(row);save()
        print('AUDIO RESULT',seconds,'same-GPU sequential split',row['sequential_split_vs_native'],
              'dummy batch',row['native_shape_dummy_channel_control']['bit_exact'],
              'deterministic split',row['deterministic_split_vs_its_native']['bit_exact'],flush=True)
        del z,ref,mono,split,padded
    result['completed']=True;save()


if __name__=='__main__':main()
