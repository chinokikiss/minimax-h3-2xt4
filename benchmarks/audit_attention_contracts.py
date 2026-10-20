"""CPU-only inspection of existing activations; not a performance benchmark."""
import json
import math
from pathlib import Path
import torch
torch.set_num_threads(8);torch.set_grad_enabled(False)
DATA=Path('D:/attention_eval/t4_redo_20261013/dataset')
OUT=Path('D:/attention_eval/quality_constrained_20261015');OUT.mkdir(exist_ok=True)
records=[r for r in json.loads((DATA/'capture_manifest.json').read_text())['records'] if r['component']=='dit']
rows=[]
for record in records:
    data=torch.load(DATA/record['file'],map_location='cpu',weights_only=True)
    q=data['q'].half().float();k=data['k'].half().float();n=q.shape[-2];d=q.shape[-1];nb=math.ceil(n/64)
    counts=(n-torch.arange(nb)*64).clamp(0,64)
    qc=torch.nn.functional.pad(q,(0,0,0,(-n)%64)).reshape(1,q.shape[1],nb,64,d).sum(-2)/counts[None,None,:,None]
    kc=torch.nn.functional.pad(k,(0,0,0,(-n)%64)).reshape(1,k.shape[1],nb,64,d).sum(-2)/counts[None,None,:,None]
    kc=(kc-kc.mean(-2,keepdim=True)).half().float()
    scores=qc@kc.transpose(-1,-2)/math.sqrt(d)
    ids=torch.arange(nb);mandatory=((ids[:,None]-ids[None,:]).abs()<=1)|(ids[None,:]<1)
    initial=(scores>(qc*kc.mean(-2,keepdim=True)).sum(-1)[...,None]/math.sqrt(d))|mandatory
    constrained=initial.clone();constrained[...,:14]=True;constrained[...,:14,:]=True
    for h in range(q.shape[1]):
        rows.append({'file':record['file'],'capture_sha256':record['sha256'],'layer':record['layer'],'step':record['denoise_call'],
            'head':h,'q_tokens':n,'k_tokens':k.shape[-2],'head_dim':d,'leaf64_blocks':nb,'prefix_tokens':874,
            'mandatory_prefix_blocks':14,'density_before_prefix':initial[0,h].float().mean().item(),
            'density_after_prefix':constrained[0,h].float().mean().item(),
            'content_query_density':constrained[0,h,14:].float().mean().item(),
            'token_pair_weighted_density':((constrained[0,h].float()*counts[:,None]*counts[None,:]).sum()/n**2).item()})
    print('AUDITED',record['file'],'heads',q.shape[1],flush=True)
    del data,q,k,qc,kc,scores,initial,constrained
(OUT/'native_capture_head_routes.json').write_text(json.dumps({'scope':'Existing native trajectory:5layers×3steps×56heads. CPU metadata analysis only; not GPU timing.',
    'rows':rows},indent=2))
