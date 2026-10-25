"""Read-only encoder prefetch hints from the pinned H3 API prompt graph.

Actual loading is always triggered by encode(), so unknown/dynamic producers
need no inference hooks. Decode-only VAE links never request encoder prefetch.
"""
from pathlib import Path


def _link_node(prompt,value):
    if (isinstance(value,(list,tuple)) and len(value)==2 and isinstance(value[0],(str,int))
            and isinstance(value[1],int) and not isinstance(value[1],bool)):
        return prompt.get(str(value[0]),prompt.get(value[0],{}))
    return {}


def _present(value):
    if value is None:return False
    if isinstance(value,dict):return any(_present(v) for v in value.values())
    if isinstance(value,(list,tuple)):return bool(value)
    return True


def _entries(inputs,group,prefix):
    """Support nested V3 autogrow groups and flattened API input names."""
    value=inputs.get(group)
    entries=dict(value) if isinstance(value,dict) else ({'*':value} if _present(value) else {})
    for key,value in inputs.items():
        name=key.removeprefix(group+'.')
        if name.startswith(prefix) and name[len(prefix):].isdigit():entries[name]=value
    return {key.rsplit('_',1)[-1]:value for key,value in entries.items() if _present(value)}


def _vae_name(prompt,value,seen=None):
    node=_link_node(prompt,value)
    if not node:return None
    seen=set() if seen is None else seen
    if id(node) in seen:return None
    seen.add(id(node));inputs=node.get('inputs',{});kind=node.get('class_type')
    if kind in ('MiniMaxH3ParallelVAELoader','VAELoader'):
        name=inputs.get('vae_name')
        if isinstance(name,str):return Path(name.replace('\\','/')).name
    elif kind=='MiniMaxH3PostDecodePrefetch':
        return _vae_name(prompt,inputs.get('vae'),seen)
    return None


def encoder_names(prompt):
    """Conservative reference hints; never execute graph producers or tensors."""
    names=set()
    def request(inputs,key):
        name=_vae_name(prompt,inputs.get(key))
        if name is not None:names.add(name)
    for node in prompt.values():
        inputs=node.get('inputs',{});kind=node.get('class_type')
        if kind=='MiniMaxH3ReferenceToVideo':
            images=_entries(inputs,'ref_images','ref_image_')
            videos=_entries(inputs,'ref_videos','ref_video_')
            audio=_entries(inputs,'ref_audios','ref_audio_')
            soundtracks=_entries(inputs,'ref_video_audios','ref_video_audio_')
            if images or videos:request(inputs,'vae')
            paired=(bool(set(videos)&set(soundtracks))
                or ('*' in videos and bool(soundtracks)) or ('*' in soundtracks and bool(videos)))
            # Native skips video soundtrack encoding if no video VAE is supplied.
            if audio or (paired and _present(inputs.get('vae'))):request(inputs,'audio_vae')
        elif kind=='MiniMaxH3ImageToVideo':
            if any(_present(inputs.get(k)) for k in ('first_frame','last_frame')):request(inputs,'vae')
        elif kind=='MiniMaxH3AddGuide':
            if _present(inputs.get('image')):request(inputs,'vae')
            if _present(inputs.get('audio')):request(inputs,'audio_vae')
        elif kind in ('VAEEncode','VAEEncodeTiled','VAEEncodeAudio'):
            if any(_present(inputs.get(k)) for k in ('pixels','audio')):request(inputs,'vae')
    return names
