"""Experiment-only physical dual-T4 P0-B interception and exactness probes.

Production P0-A1/P0-C/P0-2/P0-3 stay identical in both variants.
Never import this module from the inference runtime.
"""
from contextlib import contextmanager
from unittest.mock import patch

import torch
from patches import tilelang_swiglu_quant as swiglu
from benchmarks.dit_p0b_inplace_probe import candidate
from benchmarks.dit_p0ac_t4_candidates import bytes_compare


@contextmanager
def active_p0b(enabled, record):
    """Change only the qualified DiT FC2 fused_stored preparation."""
    if not enabled:
        yield
        return
    original = swiglu.prepare

    def optimized(x, comm, mode='fused_stored', *, arch='sm_75', consume_input=False):
        eligible = (
            mode == 'fused_stored' and arch == 'sm_75'
            and consume_input and type(x) is torch.Tensor
            and not torch.is_grad_enabled() and not torch.compiler.is_compiling()
            and x.is_cuda and x.dtype == torch.float16 and x.ndim == 2
            and x.shape[0] in (128, 6814, 59134, 65166)
            and x.shape[1] == 14336 and x._base is None and x.is_contiguous()
            and torch.cuda.get_device_capability(x.device) == (7, 5)
        )
        if not eligible:
            raise RuntimeError('Physical P0-B hook reached unsupported FC2 owner/backend')
        row_count=x.shape[0]
        codes, scale = candidate(x, comm, 'sm_75')
        # Both remapped quantizer and source-view readers have completed.
        x.set_()
        record['p0b_fc2_calls'] = record.get('p0b_fc2_calls', 0) + 1
        record['last_rows'] = row_count
        return codes, scale

    with patch.object(swiglu, 'prepare', side_effect=optimized):
        yield


@torch.no_grad()
def probe_codes(rows, device, comm, rank):
    """Real SM75 kernels plus true two-rank global MAX; raw-bit equivalence."""
    g=torch.Generator(device=device).manual_seed(20261102 + rows * 3 + rank)
    source=torch.randn((rows,14336),dtype=torch.float16,device=device,generator=g)
    baseline_q, baseline_scale = swiglu.prepare(
        source.clone(),comm,'fused_stored',arch='sm_75',consume_input=True)
    trial=source.clone()
    candidate_q, candidate_scale = candidate(trial,comm,'sm_75')
    torch.cuda.synchronize(device)
    result=dict(rows=rows,rank=rank,
        q=bytes_compare(candidate_q.detach().cpu(),baseline_q.detach().cpu()),
        scale=bytes_compare(candidate_scale.detach().cpu(),baseline_scale.detach().cpu()))
    result['all_bit_exact']=result['q']['bit_exact'] and result['scale']['bit_exact']
    del source,trial,baseline_q,baseline_scale,candidate_q,candidate_scale
    return result
