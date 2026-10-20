"""Build blinded paired review manifests; no ratings are fabricated."""
import argparse
import hashlib
import html
import json
from pathlib import Path
import random
import shutil


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--runs',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--candidates',default='sage32_dit,group32_dit,group64_dit,scalar_tau0_dit,drop_tau0_dit')
    parser.add_argument('--key-output',type=Path)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    rng=random.Random(7319);pairs=[];key={};excluded=[]
    for cfg in sorted(args.runs.glob('native_seed*.json')):
        baseline=json.loads(cfg.read_text());seed=baseline['seed']
        native=args.runs/f'native_seed{seed}'
        if not (native/'native_video.mp4').exists():continue
        for candidate in args.candidates.split(','):
            other=args.runs/f'{candidate}_seed{seed}'
            if not (other/'native_video.mp4').exists():continue
            if not all((directory/'audio.wav').exists() and (directory/'quality_screen.json').exists() for directory in [native,other]):
                excluded.append({'candidate':other.name,'reason':'Video exists but complete video/audio export is missing'})
                continue
            values=[native,other]
            # Balance baseline-left/right placement across consecutive pairs.
            if len(pairs)%2:values.reverse()
            pair=f'pair{len(pairs)+1:03d}'
            key[pair]={'seed':seed,'A':values[0].name,'B':values[1].name}
            for label,directory in zip(['A','B'],values):
                for source_name,extension in [('native_video.mp4','mp4'),('audio.wav','wav')]:
                    source=directory/source_name
                    if not source.exists():raise FileNotFoundError(source)
                    destination=args.output/f'{pair}_{label}.{extension}'
                    if not destination.exists():
                        try:__import__('os').link(source,destination)
                        except OSError:shutil.copy2(source,destination)
            pairs.append({'pair':pair,'prompt':baseline['prompt'],'reference':'reference.jpg','A':pair+'_A.mp4','B':pair+'_B.mp4',
                          'ratings':{'reviewer_id':'','reference_A':None,'reference_B':None,'prompt_A':None,'prompt_B':None,
                              'temporal_A':None,'temporal_B':None,'artifacts_A':None,'artifacts_B':None,'AV_A':None,'AV_B':None,
                              'overall_A':None,'overall_B':None,'severe_failure_A':None,'severe_failure_B':None,'notes':''}})
    (args.output/'pairs.json').write_text(json.dumps(pairs,indent=2))
    (args.output/'excluded_incomplete_runs.json').write_text(json.dumps(excluded,indent=2))
    # Keep the key out of the reviewer directory.
    (args.key_output or args.output.parent/'private_review_key.json').write_text(json.dumps(key,indent=2))
    ref=args.runs/'reference.jpg'
    if ref.exists():shutil.copy2(ref,args.output/'reference.jpg')
    else:raise FileNotFoundError('Place exact reference.jpg next to run configs before building the review')
    content='<!doctype html><meta charset="utf-8"><title>Blinded H3 paired review</title><style>body{font:16px sans-serif;max-width:1400px;margin:auto}video{width:46%}img{max-width:320px}</style><h1>Blinded paired review</h1><p>Score each clip 1–5 for reference, prompt, temporal/motion, artifacts, AV consistency and overall quality. Mark severe failures. Use anonymous reviewer IDs; record ratings in pairs.json. Neither pixel equality nor identical motion is required.</p><img src="reference.jpg">'
    for pair in pairs:
        content+=f'<h2>{pair["pair"]}</h2><p>{html.escape(pair["prompt"])}</p>'
        for label in ['A','B']:
            file=f'{pair["pair"]}_{label}'
            content+=f'<div style="display:inline-block;width:48%"><h3>{label}</h3><video id="{file}_video" style="width:100%" controls src="{file}.mp4"></video><audio id="{file}_audio" controls src="{file}.wav"></audio><button onclick="playPair(\'{file}\')">Play with audio</button></div>'
    content+='''<script>
function playPair(id){const v=document.getElementById(id+'_video'),a=document.getElementById(id+'_audio');a.currentTime=v.currentTime;v.play();a.play();v.onpause=()=>a.pause();v.onseeking=()=>{a.currentTime=v.currentTime};v.onended=()=>a.pause();}
</script>'''
    (args.output/'review.html').write_text(content,encoding='utf8')
    print('Blinded pairs',len(pairs),'ratings pending')


if __name__=='__main__':main()
