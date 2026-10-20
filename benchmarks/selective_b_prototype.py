"""Benchmark-only FC1 Selective B; never imported by inference/UI."""
import torch
from torch import nn
import torch.nn.functional as F
from patches.h3_lora import NativeLoRATerm


class SelectiveBFC1(nn.Module):
    def __init__(self,baseline,rank):
        super().__init__()
        expected=((rank*7168,(rank+1)*7168),(14336+rank*7168,14336+(rank+1)*7168))
        if baseline.columns is not None or tuple(baseline.rows)!=expected:
            raise ValueError('Prototype supports only validated H3 FC1 ownership')
        self.rows=expected;self.terms=nn.ModuleList()
        self.pack_bytes=0;self.pack_copies=0
        for term in baseline.terms:
            if tuple(term.up.shape)!=(28672,term.down.shape[0]) or term.down.shape[1]!=5376:
                raise ValueError('Unexpected real H3 FC1 factor geometry')
            # One packed weight allocation, outside steady-state timing.
            owned=torch.cat([term.up[a:b] for a,b in expected],0)
            self.terms.append(NativeLoRATerm(term.down,owned,term.scale))
            self.pack_bytes+=owned.numel()*owned.element_size();self.pack_copies+=1
            assert self.terms[-1].down.data_ptr()==term.down.data_ptr()
        self.output_width=sum(b-a for a,b in expected)

    def add_to(self,x,out,comm=None,input_act=None):
        if input_act is not None:raise ValueError('FC1 receives unactivated input')
        flat=x.reshape(-1,x.shape[-1]);target=out.reshape(-1,out.shape[-1])
        if target.shape[-1]!=self.output_width:raise ValueError('Incorrect packed gate/up target')
        if not flat.shape[0]:return out
        for term in self.terms:
            # A, adapter order and post-B scalar arithmetic match production.
            hidden=F.linear(flat,term.down.to(dtype=x.dtype))
            delta=F.linear(hidden,term.up.to(dtype=x.dtype));del hidden
            delta.mul_(term.scale);target.add_(delta);del delta
        return out
