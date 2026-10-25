"""Export byte-preserving trained component fixtures locally, without inference."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def sha(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def export(source, prefixes, destination):
    import torch
    from patches.windowed_tp import CheckpointReader
    from safetensors.torch import save_file
    reader=CheckpointReader(source)
    names=sorted(n for n in reader.header if n.startswith(tuple(prefixes)))
    if not names:raise ValueError('No selected weights in '+str(source))
    tensors={n:reader.read_tensor(n).contiguous() for n in names}
    hashes={n:hashlib.sha256(memoryview(t.reshape(-1).view(torch.uint8).numpy()).cast('B')).hexdigest()
            for n,t in tensors.items()}
    save_file(tensors,str(destination));del tensors;gc.collect()
    return dict(file=destination.name,sha256=sha(destination),bytes=destination.stat().st_size,
                source=str(source),source_size=reader.signature[0],source_mtime_ns=reader.signature[1],
                prefixes=prefixes,tensors_sha256=hashes,quantization='Unmodified checkpoint tensor bytes; no dequantization or weight folding')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--models',type=Path,default=Path('D:/minimaxH3'))
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--fixture',type=Path,default=Path('D:/attention_eval/fp16_residual_fix_20261020/first_block.pt'))
    parser.add_argument('--include-vae',action='store_true')
    args=parser.parse_args();sys.argv=sys.argv[:1]
    if args.output.exists():raise FileExistsError('Use a new fixture directory')
    args.output.mkdir(parents=True)
    receipt=dict(schema=1,scope='Trained component/operator fixtures only; no sampler or full H3 inference',records=[])
    specs=[('qwen3vl_32b_minimax_h3_int8_convrot.safetensors',['model.layers.0.'],'te_block0.safetensors'),
           ('MiniMax_H3_Ref2VA_pruned_int8_convrot.safetensors',
            ['blocks.0.','blocks.24.','blocks.49.'],'dit_blocks.safetensors'),
           ('MiniMax_H3_Ref2VA_pruned_int8_convrot.safetensors',
            ['token_refiner.blocks.0.attn.'],'refiner_attention.safetensors'),
           ('loras/dmad_minimax_h3_4step_full_critic_comfyui.safetensors',
            ['diffusion_model.blocks.0.','diffusion_model.blocks.24.','diffusion_model.blocks.49.'],'lora_blocks.safetensors')]
    for name,prefixes,filename in specs:
        record=export(args.models/name,prefixes,args.output/filename)
        receipt['records'].append(record)
        (args.output/'fixture_manifest.json').write_text(json.dumps(receipt,indent=2))
        print('EXPORTED',filename,record['bytes'],flush=True)
    for source,name in [(args.fixture,'first_block.pt')]+([
        (args.models/'minimax_h3_video_vae_int8_convrot.safetensors','video_vae.safetensors'),
        (args.models/'minimax_h3_audio_vae_fp32_clean.safetensors','audio_vae.safetensors')] if args.include_vae else []):
        destination=args.output/name;shutil.copyfile(source,destination)
        receipt['records'].append(dict(file=name,source=str(source),sha256=sha(destination),bytes=destination.stat().st_size))
    receipt['distribution_limits']={
        'dit':'Retained native first-block input; layers 24/49 reuse it as a stress probe, not their own trajectories',
        'te':'Seeded synthetic half-rounded inputs; original Tokyo TE QKV capture is not present in the retained local directories',
        'refiner':'Trained attention weights with seeded inputs; bounded SDPA wrapper regression',
        'vae':'If included: original full trained VAE checkpoints for bounded encode/decode only'}
    (args.output/'fixture_manifest.json').write_text(json.dumps(receipt,indent=2))


if __name__=='__main__':main()
