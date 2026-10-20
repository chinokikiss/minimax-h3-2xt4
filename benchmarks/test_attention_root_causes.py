"""Check diagnostic identities without presenting synthetic checks as real error."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch
from patches.tilelang_sparse_attention import prepare, make_routes
from attention_root_causes import diagnose


def test_diagnostic_identities():
    torch.manual_seed(8021)
    q,k,v=[torch.randn(1,2,129,64).half() for _ in range(3)]
    p=prepare(q,k,v)
    dense=diagnose(q,k,v,p,make_routes(p,'dense'),queries=7)
    for name in ('centroid_key_only','mean_value_only','both_centroids','oracle','drop_without_pooling'):
        assert max(dense['ablations'][name]['output']['relative_l2'])<1e-6
    sparse=diagnose(q,k,v,p,make_routes(p,'local_global',global_blocks=0,local_radius=0),queries=7)
    assert max(sparse['ablations']['oracle']['output']['relative_l2'])<1e-5
    assert max(sparse['omitted_native_mass']['mean'])>0
    assert max(sparse['ablations']['centroid_key_only']['output']['relative_l2'])>0
    causal=diagnose(q,k,v,p,make_routes(p,'dense'),causal=True,queries=7)
    assert all(torch.isfinite(torch.tensor(x['output']['relative_l2'])).all() for x in causal['ablations'].values())


if __name__=='__main__':
    test_diagnostic_identities()
    print('Diagnostic oracle, dense identity and causal masking checks passed')
