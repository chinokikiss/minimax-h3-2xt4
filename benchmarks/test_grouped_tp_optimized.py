"""CPU operator-equivalence gates, not sparse-versus-dense quality metrics."""
from pathlib import Path
import argparse
import hashlib
import json
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from patches import tilelang_grouped_tp as baseline
from patches import tilelang_grouped_tp_optimized as opt


def compare(q,k,v,mode,tile,prefix=1):
    old=baseline.prepare(q,k,v,sparse=mode!='dense')
    new=opt.prepare(q,k,v,mode=mode,query_tile=tile)
    for key,length in [('qi',q.shape[2]),('qs',q.shape[2]),('ki',k.shape[2]),('ks',k.shape[2])]:
        assert torch.equal(getattr(old,key)[:,:,:length],getattr(new,key)[:,:,:length]),key
    assert torch.equal(old.v,new.v) and torch.equal(old.counts,new.counts)
    if mode=='scalar':assert torch.equal(old.vc,new.vc) and torch.equal(old.kc,new.kc)
    else:assert new.vc.numel()==1
    if mode=='dense':assert new.q.numel()==new.kc.numel()==1
    r=None
    if mode!='dense':
        r=baseline.routes(old,prefix)
        assert torch.equal(r,opt.routes(new,prefix))
        assert torch.equal(r,opt.routes(new,prefix,fast_tau0=True))
    # Both CPU references consume the same contract. No comparison against a
    # different dense operator is used as a sparse quality acceptance gate.
    assert torch.equal(baseline.reference(old,r,mode,query_chunk=32),opt.reference(new,r,mode,query_chunk=32))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--real-data',type=Path);args=ap.parse_args()
    torch.manual_seed(20261016);torch.set_num_threads(4)
    for d in (64,128):
        q=torch.randn(1,4,65,d).half();k=torch.randn(1,1,129,d).half();v=torch.randn_like(k)
        for tile in (32,64,128):compare(q,k,v,'dense',tile)
        q=torch.randn(1,2,129,d).half();k=torch.randn_like(q);v=torch.randn_like(q)
        for mode in ('drop','scalar'):
            for tile in (32,64):compare(q,k,v,mode,tile)
        for tile in (32,64):compare(torch.zeros_like(q),torch.zeros_like(k),v,'drop',tile)
    q=torch.randn(1,2,773,64).half();k=torch.randn(1,2,1797,64).half();v=torch.randn_like(k)
    compare(q,k,v,'dense',128)
    try:opt.prepare(q,q,q,mode='drop',query_tile=128)
    except ValueError:pass
    else:raise AssertionError('Sparse128 must not cross two independent route rows')
    try:opt.routes(opt.prepare(q,k,v))
    except ValueError:pass
    else:raise AssertionError('Dense preparation must not be routed')
    # Finite FP16 keys can still overflow their centered FP16 centroids. The
    # default router retains the original exceptional-input behavior; only
    # the explicitly selected finite-centroid shortcut omits variance work.
    extreme=torch.full((1,1,129,64),-65504.,dtype=torch.float16);extreme[:,:,:64]=65504.
    eq=torch.randn_like(extreme);ev=torch.zeros_like(extreme)
    old=baseline.prepare(eq,extreme,ev,sparse=True);new=opt.prepare(eq,extreme,ev,mode='drop')
    assert not torch.isfinite(new.kc).all()
    assert torch.equal(baseline.routes(old,1),opt.routes(new,1))
    print('PASS: exact prepared operands/routes/references; GQA, rectangular/ragged, mode pruning, sparse tile guards')
    if args.real_data:real_captures(args.real_data)


def real_captures(directory):
    torch.set_num_threads(8);records=json.loads((directory/'capture_manifest.json').read_text())['records'];rows=[]
    for component,file in [('te','te_layer00_call-1.pt'),('dit','dit_layer25_call09.pt'),('video','video_layer00_call00.pt')]:
        record=next(r for r in records if r['file']==file);path=directory/file
        assert hashlib.file_digest(path.open('rb'),'sha256').hexdigest()==record['sha256']
        data=torch.load(path,map_location='cpu',weights_only=True)
        for rank in (0,1):
            if component=='te':
                q=data['q'][:,rank*32:(rank+1)*32].half();k=data['k'][:,rank*4:(rank+1)*4].half();v=data['v'][:,rank*4:(rank+1)*4].half()
            elif component=='dit':q,k,v=[data[key][:,rank*28:(rank+1)*28].half() for key in ('q','k','v')]
            else:
                a,b=((0,1024),(1024,1797))[rank];q=data['q'][:,:,a:b];k=data['k'];v=data['v']
            assert not q.is_cuda
            for mode in (('dense','drop','scalar') if component=='dit' else ('dense',)):
                old=baseline.prepare(q,k,v,sparse=mode!='dense')
                new=opt.prepare(q,k,v,mode=mode,query_tile=128 if component=='video' else 64)
                for key,length in [('qi',q.shape[2]),('qs',q.shape[2]),('ki',k.shape[2]),('ks',k.shape[2])]:
                    assert torch.equal(getattr(old,key)[:,:,:length],getattr(new,key)[:,:,:length]),(component,rank,mode,key)
                assert torch.equal(old.v,new.v)
                if mode=='scalar':assert torch.equal(old.vc,new.vc)
                if mode!='dense':
                    assert torch.isfinite(new.kc).all() and torch.isfinite(q).all()
                    assert torch.equal(baseline.routes(old,874),opt.routes(new,874,fast_tau0=True))
                rows.append({'component':component,'rank':rank,'mode':mode,'q_shape':list(q.shape),'k_shape':list(k.shape),
                             'operand_identity':True,'route_identity':True if mode!='dense' else None})
                print('REAL_CPU_IDENTITY',component,rank,mode,flush=True)
                del old,new
        del data,q,k,v
    out=Path('D:/attention_eval/operator_optimization_tp2_20261016/offline_optimization')
    out.mkdir(parents=True,exist_ok=True)
    (out/'real_operand_identity.json').write_text(json.dumps({'scope':'CPU exact identity to frozen sparse/dense contracts; no dense-relative quality metrics or GPU execution','cases':rows},indent=2))


if __name__=='__main__':main()
