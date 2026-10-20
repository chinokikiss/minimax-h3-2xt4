"""Recompute one full video comparison independently with NumPy FP64 reductions."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch

def verify(directory):
    latent = torch.load(directory/'input.pt',map_location='cpu',weights_only=True)
    expected = torch.randn((1,24,2,48,84),generator=torch.Generator().manual_seed(46)).half()
    assert torch.equal(latent,expected), 'Saved latent differs from the declared seed and shape'
    input_hash = hashlib.sha256(latent.numpy().tobytes()).hexdigest()
    ref_tensor = torch.load(directory/'native_output.pt',map_location='cpu',weights_only=True)
    assert tuple(ref_tensor.shape)==(1,3,5,768,1344), f'Unexpected decoded frame shape: {ref_tensor.shape}'
    ref = ref_tensor.numpy().astype(np.float64).reshape(-1)
    assert np.isfinite(ref).all() and np.any(ref), 'Invalid native reference'
    reported = json.loads((directory/'accuracy.json').read_text())
    recomputed = {}
    for mode in ['tp','sp']:
        value_tensor = torch.load(directory/f'{mode}_output.pt',map_location='cpu',weights_only=True)
        assert value_tensor.shape == ref_tensor.shape
        value = value_tensor.numpy().astype(np.float64).reshape(-1)
        assert np.isfinite(value).all() and np.any(value), f'Invalid {mode} output'
        diff = value-ref
        computed = {'rel_l2':float(np.linalg.norm(diff)/np.linalg.norm(ref)),
                    'cosine':float(np.dot(value,ref)/(np.linalg.norm(value)*np.linalg.norm(ref))),
                    'max_abs':float(np.abs(diff).max()),'mean_abs':float(np.abs(diff).mean()),
                    'is_bit_exact':bool(np.array_equal(value,ref))}
        for key in ['rel_l2','cosine','max_abs','mean_abs']:
            assert math.isclose(computed[key],reported[mode][key],rel_tol=1e-10,abs_tol=1e-12), (mode,key,computed,reported[mode])
        assert computed['is_bit_exact']==reported[mode]['is_bit_exact']
        profile = json.loads((directory/f'{mode}_rank0_profile.json').read_text())
        assert profile['input_sha256']==input_hash
        recomputed[mode]=computed
    native_profile = json.loads((directory/'native_profile.json').read_text())
    assert native_profile['input_sha256']==input_hash
    record = {'independent_numpy_verification_passed':True,'input_sha256':input_hash,
              'output_shape':list(ref_tensor.shape),'comparisons':recomputed}
    (directory/'independent_metrics.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
    print(json.dumps(record,indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('directory',type=Path)
    verify(parser.parse_args().directory)
