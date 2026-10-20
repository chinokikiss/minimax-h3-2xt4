"""Frozen previous Row Partial and quantizer for memory/byte A/B only."""
import torch
from torch import nn
import torch.nn.functional as F
import comfy_kitchen as ck
from patches.h3_lora import NativeLoRATerm,_add_delta

class BaselineRowPartial(nn.Module):
    """Opt-in A-only approximations; B shapes, scales and ordering stay Native."""
    def __init__(self,terms,rows=None,columns=None,*,row_partial=False,concat_a=False):
        super().__init__()
        terms=[(a,b,s) for a,b,s in terms if s]
        self.rows,self.columns=rows,columns
        self.row_partial=bool(row_partial and columns is not None)
        self.concat_a=bool(concat_a and len(terms)>1)
        self.ranks=tuple(a.shape[0] for a,_,_ in terms)
        if self.concat_a:
            # Store A once: individual terms retain only B and the original scale.
            self.register_buffer('down',torch.cat([a.float() for a,_,_ in terms],0).contiguous())
            self.terms=nn.ModuleList([NativeLoRATerm(torch.empty(0),b,s) for _,b,s in terms])
        else:
            self.terms=nn.ModuleList([NativeLoRATerm(a,b,s) for a,b,s in terms])

    def add_to(self,x,out,comm=None,input_act=None):
        flat=x.reshape(-1,x.shape[-1]);target=out.reshape(-1,out.shape[-1])
        if not flat.shape[0]:return out
        if input_act=='swiglu':
            gate,up=flat.chunk(2,-1);flat=F.silu(gate).mul_(up)
        elif input_act is not None:raise ValueError('Unqualified LoRA input activation')
        if self.columns is not None:
            if comm is None or comm.get_world_size()!=2:raise ValueError('TP2 row LoRA requires its rank transport')
            if not self.row_partial:
                count,width=flat.shape
                gathered=torch.empty((2*count,width),dtype=flat.dtype,device=flat.device)
                comm.all_gather_into_tensor(gathered,flat.contiguous());del flat
                flat=gathered.view(2,count,width).transpose(0,1).reshape(count,2*width);del gathered
        combined=None
        if self.concat_a:
            combined=F.linear(flat,self.down.to(dtype=x.dtype))
            if self.row_partial:comm.all_reduce(combined,op=comm.ReduceOp.SUM)
        offset=0
        for term,width in zip(self.terms,self.ranks):
            if combined is None:
                hidden=F.linear(flat,term.down.to(dtype=x.dtype))
                if self.row_partial:comm.all_reduce(hidden,op=comm.ReduceOp.SUM)
            else:hidden=combined[:,offset:offset+width].contiguous()
            offset+=width
            delta=F.linear(hidden,term.up.to(dtype=x.dtype));del hidden
            _add_delta(target,delta,self.rows,term.scale);del delta
        return out

def global_row_quantize(x,comm,world_size,groupsize,input_act):
    backend=ck.backends.cuda
    flat=x.reshape(-1,x.shape[-1]).contiguous()
    # Retain native eager activation rounding on the native fallback route.
    flat=backend._apply_input_act(flat,input_act,None,0.0)
    local_width=flat.shape[-1]
    if local_width%groupsize:raise ValueError('ConvRot groups must remain wholly owned by a rank')
    h=backend._build_hadamard(groupsize,device=flat.device,dtype=flat.dtype)
    if flat.dtype==torch.float32:
        # FP32 rotation GEMM can change reduction order when its flattened
        # row count changes. Preserve the native global GEMM geometry; zero
        # unowned groups, then discard them. This retains its operator route
        # while still replacing the large activation gather with row maxima.
        padded=torch.zeros((flat.shape[0],local_width*world_size),device=flat.device,dtype=flat.dtype)
        start=comm.get_rank()*local_width
        padded[:,start:start+local_width].copy_(flat)
        rotated=backend._rotate_activation(padded,h,groupsize)[:,start:start+local_width].contiguous()
        # The contiguous owned slice no longer aliases the full-width input.
        # Its last CUDA read is already queued on this stream; do not retain
        # the large padding allocation across the row-max collective.
        del padded
    else:
        rotated=backend._rotate_activation(flat,h,groupsize)
    maximum=rotated.abs().amax(-1,keepdim=True).float()
    comm.all_reduce(maximum,op=comm.ReduceOp.MAX)
    # Same native row-wise CUDA quantizer, global scale, rounding and clamping.
    # The extra group is never supplied to the matrix multiplication.
    sentinel=torch.zeros((rotated.shape[0],groupsize),device=rotated.device,dtype=rotated.dtype)
    sentinel[:,0:1].copy_(maximum)
    q,scale=backend.quantize_int8_rowwise(torch.cat([rotated,sentinel],-1))
    return q[:,:local_width].contiguous(),scale
