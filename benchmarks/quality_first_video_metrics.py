"""Bounded generation diagnostics. Embedding/flow proxies are not quality gates."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np


def video_frames(path):
    import av
    with av.open(str(path)) as container:
        stream=container.streams.video[0];fps=float(stream.average_rate)
        frames=[f.to_ndarray(format='rgb24') for f in container.decode(stream)]
    return frames,fps


def describe(values):
    a=np.asarray(values,dtype=np.float64)
    return dict(mean=float(a.mean()),median=float(np.median(a)),p05=float(np.quantile(a,.05)),p95=float(np.quantile(a,.95)),samples=a.tolist())


def score(video,reference,prompt,output,device='cpu',embeddings=False,embedding_revisions=None):
    import cv2
    from PIL import Image
    frames,fps=video_frames(video)
    assert frames and len(frames)<=300,'Bounded whole-video decoding (max300frames)'
    result={'video_sha256':hashlib.sha256(video.read_bytes()).hexdigest(),'reference_sha256':hashlib.sha256(reference.read_bytes()).hexdigest(),
        'frames':len(frames),'fps':fps,'seconds':len(frames)/fps,'quality_acceptance':'unrated; proxies are supporting diagnostics',
        'metric_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    small=[cv2.resize(f,(256,144),interpolation=cv2.INTER_AREA).astype(np.float32)/255 for f in frames]
    residual=[];motion=[]
    for previous,current in zip(small[:-1],small[1:]):
        gp=cv2.cvtColor(previous,cv2.COLOR_RGB2GRAY);gc=cv2.cvtColor(current,cv2.COLOR_RGB2GRAY)
        # Backward flow maps current coordinates into the previous frame.
        backward=cv2.calcOpticalFlowFarneback((gc*255).astype(np.uint8),(gp*255).astype(np.uint8),None,.5,3,15,3,5,1.2,0)
        yy,xx=np.mgrid[:144,:256];mapx=xx.astype(np.float32)+backward[...,0];mapy=yy.astype(np.float32)+backward[...,1]
        warped=cv2.remap(previous,mapx,mapy,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT)
        valid=(mapx>=0)&(mapx<255)&(mapy>=0)&(mapy<143)
        residual.append(float(np.abs(warped-current)[valid].mean()))
        motion.append(float(np.linalg.norm(backward,axis=-1)[valid].mean()))
    result['flow_proxy']={'method':'Farneback at256x144; not VBench/RAFT and not calibrated perceived quality',
        'warped_photometric_residual':describe(residual),'motion_pixels_per_frame':describe(motion),
        'post_two_seconds_residual':describe([v for i,v in enumerate(residual,1) if i/fps>=2]) if len(frames)/fps>2 else None}
    result['embedding_status']='not_requested'
    if embeddings:
        import torch
        from huggingface_hub import HfApi
        from transformers import AutoImageProcessor,AutoModel,CLIPModel,CLIPProcessor
        torch.set_grad_enabled(False)
        api=HfApi();repos=['facebook/dinov2-small','openai/clip-vit-base-patch32']
        revisions=embedding_revisions or {repo:api.model_info(repo).sha for repo in repos}
        if set(revisions)!=set(repos) or not all(len(v)==40 and all(c in '0123456789abcdef' for c in v) for v in revisions.values()):
            raise ValueError('Pinned DINO/CLIP revision SHA40 required')
        result['embedding_versions']=revisions
        # Resolve and freeze revisions BEFORE loading either model.
        dino=AutoModel.from_pretrained(repos[0],revision=revisions[repos[0]]).to(device).eval()
        dp=AutoImageProcessor.from_pretrained(repos[0],revision=revisions[repos[0]])
        clip=CLIPModel.from_pretrained(repos[1],revision=revisions[repos[1]]).to(device).eval()
        cp=CLIPProcessor.from_pretrained(repos[1],revision=revisions[repos[1]])
        images=[Image.open(reference).convert('RGB')]+[Image.fromarray(f) for f in frames]
        df=[];cf=[]
        for i in range(0,len(images),8):
            chunk=images[i:i+8]
            df.append(dino(**dp(images=chunk,return_tensors='pt').to(device)).last_hidden_state[:,0].float().cpu())
            feat=clip.get_image_features(**cp(images=chunk,return_tensors='pt').to(device))
            if not isinstance(feat,torch.Tensor):feat=feat.pooler_output
            cf.append(feat.float().cpu())
        normalize=lambda x:torch.nn.functional.normalize(x,dim=-1)
        df=normalize(torch.cat(df));cf=normalize(torch.cat(cf))
        text=clip.get_text_features(**cp(text=[prompt],return_tensors='pt',padding=True,truncation=True).to(device))
        if not isinstance(text,torch.Tensor):text=text.pooler_output
        text=normalize(text.float().cpu())
        dref=(df[1:]*df[:1]).sum(-1);cref=(cf[1:]*cf[:1]).sum(-1)
        adherence=(cf[1:]*text).sum(-1)
        first_drift=(df[1:]*df[1:2]).sum(-1)
        after=torch.arange(len(frames))/fps>=2
        result['embedding_status']='measured'
        result['embedding_protocol']='All decoded frames; model processor default spatial resize/crop; CLS DINOv2 and projected CLIP cosine. Whole-image appearance/semantic proxies, not identity guarantees.'
        result['reference_dino']=describe(dref.numpy());result['reference_clip']=describe(cref.numpy())
        result['prompt_clip']=describe(adherence.numpy());result['first_frame_dino']=describe(first_drift.numpy())
        result['consecutive_dino']=describe((df[2:]*df[1:-1]).sum(-1).numpy())
        if after.any():
            result['post_two_seconds']={name:describe(value[after].numpy()) for name,value in [('reference_dino',dref),('prompt_clip',adherence),('first_frame_dino',first_drift)]}
        result['prompt_embedding_limit']='CLIP truncates to its model token limit; do not claim full long-prompt adherence from this metric'
    output.write_text(json.dumps(result,indent=2));return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--video',type=Path,required=True);p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--prompt',required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--device',default='cpu');p.add_argument('--embeddings',action='store_true')
    p.add_argument('--embedding-revisions',type=Path)
    args=p.parse_args();score(args.video,args.reference,args.prompt,args.output,args.device,args.embeddings,
        json.loads(args.embedding_revisions.read_text()) if args.embedding_revisions else None)
