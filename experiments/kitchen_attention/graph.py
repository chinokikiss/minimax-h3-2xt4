"""Explicit, fixed-shape CUDA Graph experiment around Kitchen 0.2.37.

No automatic dispatch/cache: a plan owns persistent input/output buffers and
must be used on its creating stream, with gradients disabled. Copies and an
independent output clone are part of every replay call.
"""
import time

import torch


def signature(q,k,v):
    tensors=(q,k,v)
    if any(x.ndim!=4 for x in tensors):raise ValueError('Expected BHND inputs')
    if any(x.device!=q.device or x.dtype!=q.dtype for x in tensors):
        raise ValueError('Inputs must have matching device and dtype')
    if k.shape!=v.shape or q.shape[0]!=k.shape[0] or q.shape[-1]!=k.shape[-1]:
        raise ValueError('Incompatible attention geometry')
    if q.shape[1]%k.shape[1] or min(*q.shape,*k.shape)<=0:
        raise ValueError('Invalid grouped-query geometry')
    if any(x.requires_grad for x in tensors) or torch.is_grad_enabled():
        raise ValueError('Graph plans require inference with gradients disabled')
    return tuple((tuple(x.shape),x.dtype,x.device) for x in tensors)


class KitchenGraphPlan:
    def __init__(self,q,k,v):
        import comfy_kitchen as ck
        self.signature=signature(q,k,v)
        if q.device.type!='cuda' or torch.cuda.get_device_capability(q.device)!=(7,5):
            raise ValueError('This experiment is qualified only for SM75 CUDA')
        self.stream=torch.cuda.current_stream(q.device).cuda_stream
        self.inputs=tuple(torch.empty_like(x,memory_format=torch.contiguous_format) for x in (q,k,v))
        for dst,src in zip(self.inputs,(q,k,v)):dst.copy_(src)
        side=torch.cuda.Stream(device=q.device)
        side.wait_stream(torch.cuda.current_stream(q.device))
        start=time.perf_counter()
        with torch.cuda.stream(side):
            for _ in range(3):
                out=ck.int8_attention(*self.inputs)
                out=out.transpose(1,2).reshape(q.shape[0],q.shape[2],-1)
                del out
        side.synchronize()
        self.graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph,stream=side):
            out=ck.int8_attention(*self.inputs)
            self.output=out.transpose(1,2).reshape(q.shape[0],q.shape[2],-1)
        side.synchronize()
        self.capture_ms=(time.perf_counter()-start)*1000

    def __call__(self,q,k,v):
        if signature(q,k,v)!=self.signature:raise ValueError('Graph input signature changed')
        if torch.cuda.current_stream(q.device).cuda_stream!=self.stream:
            raise ValueError('Graph plan must run on its creating stream')
        for dst,src in zip(self.inputs,(q,k,v)):dst.copy_(src)
        self.graph.replay()
        return self.output.clone()
