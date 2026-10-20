"""CPU-only exhaustive maxima/RMS with explicitly sampled percentiles."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch

root=Path('D:/attention_eval/tp2_p2_20261018');data=root/'real_transport_dataset'
manifest=json.loads((data/'real_transport_manifest.json').read_text());records=[]
for fixture in manifest['records']:
    value=torch.load(data/fixture['file'],map_location='cpu',weights_only=True);x=value['x'].numpy();m,n=x.shape
    sample=np.abs(x.reshape(-1)[::max(1,x.size//1000000)])
    rowmax=np.zeros(m);rowsq=np.zeros(m);channelmax=np.zeros(n);totalmax=0.
    for start in range(0,m,128):
        block=x[start:start+128].astype(np.float64);absolute=np.abs(block)
        rowmax[start:start+block.shape[0]]=absolute.max(1);rowsq[start:start+block.shape[0]]=np.mean(block*block,1)
        channelmax=np.maximum(channelmax,absolute.max(0));totalmax=max(totalmax,float(absolute.max()))
    ratio=rowmax/np.maximum(np.sqrt(rowsq),1e-30);ws=value['weights']['weight_scale'].numpy()
    records.append(dict(name=fixture['name'],shape=[m,n],source_sha256=fixture['sha256'],max_abs_exact=totalmax,RMS_exact=float(np.sqrt(np.mean(rowsq))),
        abs_percentiles_sampled=dict(zip(('p50','p90','p99','p99_9','p99_99'),map(float,np.percentile(sample,[50,90,99,99.9,99.99])))),
        percentile_sampling=dict(method='deterministic evenly strided full flat array',sample_count=sample.size,total_elements=x.size),
        row_max_over_RMS_percentiles=list(map(float,np.percentile(ratio,[50,90,99,99.9,100]))),
        top_channels_by_max=[dict(channel=int(i),max_abs=float(channelmax[i])) for i in np.argsort(channelmax)[-8:][::-1]],
        learned_weight_scale_percentiles=list(map(float,np.percentile(ws,[0,50,90,99,100]))),
        raw_input_act='packed gate and up before native SwiGLU' if fixture['name'].startswith('fc2') else 'native full attention output',
        scope='CPU input-distribution description; not T4 latency or generation-quality evidence.'))
    del value,x
(root/'input_distribution.json').write_text(json.dumps(records,indent=2));print('REAL_INPUT_DISTRIBUTIONS',len(records))
