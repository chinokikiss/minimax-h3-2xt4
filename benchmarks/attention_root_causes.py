"""Controlled diagnostic ablations on all heads of fixed native attention captures.

Probability diagnostics sample deterministic query rows; operator error must be
reported separately over the full sequence. This is a diagnostic FP32 reference,
not a deployment kernel or performance benchmark.
"""
import math
import torch
from patches.tilelang_sparse_attention import LEAF


def metrics(actual, expected):
    a, e = actual.float(), expected.float()
    axes = (-2, -1)
    delta = a - e
    num, den = delta.square().sum(axes), e.square().sum(axes)
    return {'relative_l2': (num / den.clamp_min(1e-30)).sqrt().flatten().tolist(),
            'squared_error': num.flatten().tolist(), 'squared_reference': den.flatten().tolist(),
            'max_absolute': delta.abs().amax(axes).flatten().tolist()}


def probability_metrics(z, reference_z, mask):
    p, ref = z.softmax(-1), reference_z.softmax(-1)
    centered = torch.where(mask, z - reference_z, 0.)
    centered -= centered.sum(-1, keepdim=True) / mask.sum(-1, keepdim=True)
    centered = centered.masked_fill(~mask, 0.)
    entropy = lambda x: -(x * x.clamp_min(1e-30).log()).sum(-1)
    return {'logit_centered_rmse': (centered.square().sum(-1) / mask.sum(-1)).sqrt().mean(-1).flatten().tolist(),
        'logit_centered_absmax': centered.abs().amax((-2, -1)).flatten().tolist(),
        'kl_native_to_candidate': (ref * (ref.clamp_min(1e-30).log() - p.clamp_min(1e-30).log())).sum(-1).mean(-1).flatten().tolist(),
        'probability_total_variation': ((p-ref).abs().sum(-1)*.5).mean(-1).flatten().tolist(),
        'top1_flip_fraction': (p.argmax(-1)!=ref.argmax(-1)).float().mean(-1).flatten().tolist(),
        'entropy_delta': (entropy(p)-entropy(ref)).mean(-1).flatten().tolist()}


def diagnose(q, k, v, prepared, routes, causal=False, queries=64):
    b, h, n, d = q.shape
    assert b == 1
    ids = torch.linspace(0, n-1, min(n, queries), device=q.device).round().long().unique()
    q, k, v = q.float(), k.float(), v.float()
    mask = torch.ones((len(ids), n), device=q.device, dtype=torch.bool)
    if causal:
        mask = torch.arange(n, device=q.device)[None, :] <= ids[:, None]
    mask = mask[None, None]
    scale = d**-.5
    qs, ks = prepared.qs[:, :, ids], prepared.ks[:, :, :n]
    qhat = prepared.qi[:, :, ids].float() * qs[..., None]
    khat = prepared.ki[:, :, :n].float() * ks[..., None]
    # Match preparation's unweighted block centroid shift, including ragged leaf.
    counts = prepared.counts
    kp = torch.nn.functional.pad(k.half().float(), (0,0,0,len(counts)*LEAF-n))
    raw_kc = kp.reshape(b,h,len(counts),LEAF,d).sum(-2)/counts[None,None,:,None]
    mean = raw_kc.mean(-2, keepdim=True)
    q16, k16, v16 = q.half().float(), k.half().float(), v.half().float()
    native_z = (q[:, :, ids] @ k.transpose(-1,-2) * scale).masked_fill(~mask, -float('inf'))
    native_p = native_z.softmax(-1)
    native_o = native_p @ v
    variants = {
        'cast_only': (q16[:, :, ids], k16),
        'q8_only': (qhat, k16),
        'k8_only': (q16[:, :, ids], khat + mean),
        'q8_k8': (qhat, khat + mean),
    }
    output = {'sample_query_ids': ids.tolist(), 'query_count': len(ids), 'ablations': {},
              'scope': 'deterministic sampled query rows, every key and provided head'}
    for name, (qa, ka) in variants.items():
        z = (qa @ ka.transpose(-1,-2) * scale).masked_fill(~mask, -float('inf'))
        output['ablations'][name] = {'output': metrics(z.softmax(-1) @ v16, native_o),
                                    'probability': probability_metrics(z, native_z, mask)}
    vhat = prepared.vi[:, :, :n].float() * prepared.vs.repeat_interleave(LEAF, -2)[:, :, :n]
    output['ablations']['v8_only'] = {'output': metrics(native_p @ vhat, native_o)}
    # Round P at the same 64-key leaf scale as the TileLang INT8 PV path.
    u_p, u_pv = torch.zeros_like(native_o), torch.zeros_like(native_o)
    zero_count, positive_count = torch.zeros((b,h),device=q.device), torch.zeros((b,h),device=q.device)
    for start in range(0,n,LEAF):
        sl=slice(start,min(start+LEAF,n)); pe=native_p[...,sl]
        maximum=pe.amax(-1,keepdim=True).clamp_min(1e-30)
        pi=(pe/maximum*127+.5).floor().clamp(0,127)
        phat=pi/127*maximum
        u_p+=phat@v[...,sl,:]; u_pv+=phat@vhat[...,sl,:]
        zero_count+=((pi==0)&(pe>0)).sum((-2,-1));positive_count+=(pe>0).sum((-2,-1))
    output['ablations']['p8_only']={'output':metrics(u_p,native_o),
        'positive_probabilities_rounded_to_zero':(zero_count/positive_count.clamp_min(1)).flatten().tolist()}
    output['ablations']['p8_v8']={'output':metrics(u_pv,native_o)}
    selected=routes[:, :, ids//LEAF].repeat_interleave(LEAF,-1)[..., :n].bool()
    mass=(native_p*~selected).sum(-1)
    output['omitted_native_mass']={'mean':mass.mean(-1).flatten().tolist(),
        'p95':mass.quantile(.95,dim=-1).flatten().tolist(),'max':mass.amax(-1).flatten().tolist()}
    output['input_outliers']={name:{'absmax':x.abs().amax((-2,-1)).flatten().tolist(),
        'absmax_over_rms':(x.abs().amax((-2,-1))/x.square().mean((-2,-1)).sqrt().clamp_min(1e-30)).flatten().tolist()}
        for name,x in [('q',q),('k',k),('v',v)]}
    if not causal:
        # The exact probability mass and conditional weighted V of each omitted
        # block form an oracle; changing just one isolates the two pooling errors.
        maxz=native_z.amax(-1,keepdim=True)
        expz=torch.exp(native_z-maxz)
        exact_num=expz@v; exact_den=expz.sum(-1)
        numerators={name:exact_num.clone() for name in ['centroid_key_only','mean_value_only','both_centroids','oracle']}
        denominators={name:exact_den.clone() for name in numerators}
        for kb in range(len(counts)):
            sl=slice(kb*LEAF,min((kb+1)*LEAF,n)); omitted=~routes[:, :, ids//LEAF,kb].bool()
            pe=expz[...,sl]; weight=pe.sum(-1); weighted=pe@v[...,sl,:]
            cond=weighted/weight[...,None].clamp_min(1e-30)
            kc=k[...,sl,:].mean(-2);vc=v[...,sl,:].mean(-2)
            poolz=(q[:,:,ids]*kc[...,None,:]).sum(-1)*scale+math.log(sl.stop-sl.start)
            poolweight=(poolz-maxz.squeeze(-1)).exp()
            for name in numerators:
                new_weight=poolweight if name in ('centroid_key_only','both_centroids') else weight
                new_v=vc[...,None,:] if name in ('mean_value_only','both_centroids') else cond
                numerators[name]+=torch.where(omitted[...,None],new_weight[...,None]*new_v-weighted,0.)
                denominators[name]+=torch.where(omitted,new_weight-weight,0.)
        for name in numerators:
            out=numerators[name]/denominators[name][...,None].clamp_min(1e-30)
            output['ablations'][name]={'output':metrics(out,native_o)}
        dropped=(native_p*selected); dropped/=dropped.sum(-1,keepdim=True).clamp_min(1e-30)
        output['ablations']['drop_without_pooling']={'output':metrics(dropped@v,native_o)}
        assert max(output['ablations']['oracle']['output']['relative_l2'])<1e-4, 'Pooling oracle implementation failed'
    return output
