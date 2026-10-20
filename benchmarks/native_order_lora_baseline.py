"""Frozen v2 add_to for same-run byte-exact baseline comparisons only."""
import torch
import torch.nn.functional as F

def add_to(self,x,out,comm=None,input_act=None):
    flat=x.reshape(-1,x.shape[-1]);target=out.reshape(-1,out.shape[-1])
    if not flat.shape[0]:return out
    if input_act=='swiglu':
        # Pointwise activation commutes with feature ownership; gathering
        # the activated features halves payload versus gathering gate/up.
        gate,up=flat.chunk(2,-1);flat=F.silu(gate).mul_(up)
    elif input_act is not None:raise ValueError('Unqualified LoRA input activation')
    if self.columns is not None:
        if comm is None or comm.get_world_size()!=2:raise ValueError('TP2 row LoRA requires its rank transport')
        parts=[torch.empty_like(flat) for _ in range(2)]
        comm.all_gather(parts,flat.contiguous());flat=torch.cat(parts,-1)
        del parts
    for term in self.terms:
        hidden=F.linear(flat,term.down.to(dtype=x.dtype))
        delta=F.linear(hidden,term.up.to(dtype=x.dtype));del hidden
        # Scale AFTER B, exactly as upstream LoRAAdapter.h. In-place
        # scaling avoids retaining two full B outputs, without reassociation.
        delta.mul_(term.scale)
        if self.rows is None:target.add_(delta)
        else:
            destination=0
            for start,end in self.rows:
                target[:,destination:destination+end-start].add_(delta[:,start:end])
                destination+=end-start
        del delta
    return out
