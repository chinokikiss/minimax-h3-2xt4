"""Native-ordered H3 multi-LoRA with component-aware TP output ownership.

Only ordinary linear A/B LoRA is accepted. No ignored keys or dense runtime
weight delta. Native arithmetic is the default; row A partial reduction is opt-in. A-rank
concatenation is deprecated and rejected. Offline merging uses Native math.
"""
import hashlib
import json
import math
from pathlib import Path
import re
import torch
from torch import nn
import torch.nn.functional as F
from patches.windowed_tp import CheckpointReader

TARGET=re.compile(r'(?:blocks\.([0-9]+)|token_refiner\.blocks\.([0-9]+))\.(attn\.(?:qkv_proj|out_proj)|mlp\.(?:fc1|fc2)|adaln_proj\.linear)$')


def _add_delta(target,delta,rows,scale):
    # Scale AFTER B, exactly as upstream LoRAAdapter.h. In-place
    # scaling avoids retaining two full B outputs, without reassociation.
    if rows is None:
        delta.mul_(scale);target.add_(delta)
    else:
        # Packed QKV / gate-up own the same half of each component.
        # Keep full B geometry, but scale/add only the used output view
        # in one kernel each. Every retained element sees the same
        # multiply then add; discarded B rows need no scaling.
        components=len(rows);component=delta.shape[-1]//components
        start,end=rows[0];owned=end-start
        regular=all((a,b)==(c*component+start,c*component+end)
                    for c,(a,b) in enumerate(rows))
        if regular and component*components==delta.shape[-1]:
            used=delta.view(-1,components,component)[:,:,start:end]
            used.mul_(scale)
            target.view(-1,components,owned).add_(used)
            # A view owns its full B storage. Drop it before the next
            # adapter allocates another full B output.
            del used
        else:
            delta.mul_(scale);destination=0
            for start,end in rows:
                target[:,destination:destination+end-start].add_(delta[:,start:end])
                destination+=end-start


class NativeLoRATerm(nn.Module):
    def __init__(self,down,up,scale):
        super().__init__()
        self.register_buffer('down',down.float().contiguous())
        self.register_buffer('up',up.float().contiguous())
        self.scale=float(scale)


class NativeAlignedLoRA(nn.Module):
    """Preserve Native per-adapter order, full GEMM geometry and post-B scale.

    TP row inputs are gathered before A instead of reducing FP32 partials.
    Column projections evaluate full B before selecting component-aware rows.
    Gathering changes storage only: A/B GEMM shapes and adapter order stay Native.
    """
    def __init__(self,terms,rows=None,columns=None):
        super().__init__()
        self.terms=nn.ModuleList([NativeLoRATerm(a,b,s) for a,b,s in terms if s])
        if not self.terms:raise ValueError('No active LoRA terms')
        self.rows,self.columns=rows,columns

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
            # A single rank-major receive buffer avoids the tensor-list gather's
            # output staging/copies. Release the local activation before layout
            # conversion; the collective establishes the current-stream wait.
            count,width=flat.shape
            gathered=torch.empty((2*count,width),dtype=flat.dtype,device=flat.device)
            comm.all_gather_into_tensor(gathered,flat.contiguous())
            del flat
            flat=gathered.view(2,count,width).transpose(0,1).reshape(count,2*width)
            del gathered
        for term in self.terms:
            hidden=F.linear(flat,term.down.to(dtype=x.dtype))
            delta=F.linear(hidden,term.up.to(dtype=x.dtype));del hidden
            _add_delta(target,delta,self.rows,term.scale)
            del delta
        return out


class RowPartialLoRA(nn.Module):
    """Opt-in row A partial SUM, with bounded short-lived hidden staging."""
    HIDDEN_STAGE_BYTES=32*2**20

    def __init__(self,terms,columns):
        super().__init__()
        if columns is None:raise ValueError('Row Partial requires TP input ownership')
        self.columns=columns
        self.terms=nn.ModuleList([NativeLoRATerm(a,b,s) for a,b,s in terms if s])
        if not self.terms:raise ValueError('No active LoRA terms')

    def add_to(self,x,out,comm=None,input_act=None):
        if comm is None or comm.get_world_size()!=2:raise ValueError('TP2 row LoRA requires its rank transport')
        flat=x.reshape(-1,x.shape[-1]);target=out.reshape(-1,out.shape[-1])
        if not flat.shape[0]:return out
        if input_act=='swiglu':
            gate,up=flat.chunk(2,-1);flat=F.silu(gate).mul_(up)
        elif input_act is not None:raise ValueError('Unqualified LoRA input activation')
        # A GEMM shapes and per-adapter SUM order are unchanged. Stage only
        # when tiny hidden tensors can replace a larger owned activation.
        hidden_bytes=flat.shape[0]*sum(t.down.shape[0] for t in self.terms)*x.element_size()
        stage=(input_act=='swiglu' and hidden_bytes<=self.HIDDEN_STAGE_BYTES
               and hidden_bytes<flat.numel()*flat.element_size())
        pending=None
        if stage:
            pending=[]
            for term in self.terms:
                hidden=F.linear(flat,term.down.to(dtype=x.dtype))
                comm.all_reduce(hidden,op=comm.ReduceOp.SUM);pending.append(hidden)
            del flat,hidden
        for index,term in enumerate(self.terms):
            if pending is None:
                hidden=F.linear(flat,term.down.to(dtype=x.dtype))
                comm.all_reduce(hidden,op=comm.ReduceOp.SUM)
                if index+1==len(self.terms):del flat
            else:
                hidden=pending[index];pending[index]=None
            delta=F.linear(hidden,term.up.to(dtype=x.dtype));del hidden
            _add_delta(target,delta,None,term.scale);del delta
        return out


class NativeLoraLinear(nn.Module):
    def __init__(self,base,adapter):super().__init__();self.base=base;self.lora_adapter=adapter
    def forward(self,x):return self.lora_adapter.add_to(x,self.base(x))


class NativeLoraMLP(nn.Module):
    """Native fused fc2 skips module.forward, so apply its bypass explicitly."""
    def __init__(self,base,fc1,fc2):
        super().__init__();self.fc1=base.fc1;self.fc2=base.fc2;self.fc1_adapter=fc1;self.fc2_adapter=fc2
    def forward(self,x):
        import comfy.ops
        packed=self.fc1(x)
        if self.fc1_adapter is not None:packed=self.fc1_adapter.add_to(x,packed)
        out=comfy.ops.linear_input_act(self.fc2,packed,'swiglu')
        if self.fc2_adapter is not None:out=self.fc2_adapter.add_to(packed,out,input_act='swiglu')
        return out


class H3LoRAStore:
    def __init__(self,sources,base_reader,*,row_partial=False,concat_a=False,selective_b=False):
        if concat_a:raise ValueError('Concat A is deprecated; disable concat_a and use optional Row Partial')
        self.row_partial=bool(row_partial)
        self.selective_b=bool(selective_b)
        self.sources=[];self.targets=set()
        for path,strength in sources:
            if not math.isfinite(strength):raise ValueError('Finite LoRA strength required')
            reader=CheckpointReader(path);known=set();targets={}
            for key in reader.header:
                if key=='__metadata__' or not key.endswith('.lora_A.weight'):continue
                raw=key.removesuffix('.lora_A.weight');target=raw.removeprefix('diffusion_model.')
                match=TARGET.fullmatch(target)
                if match is None or (match[1] is not None and int(match[1])>=50) or (match[2] is not None and int(match[2])>=2):
                    raise ValueError('Unsupported H3 LoRA target: '+raw)
                a=raw+'.lora_A.weight';b=raw+'.lora_B.weight';alpha=raw+'.alpha'
                if b not in reader.header:raise ValueError('LoRA B factor missing: '+raw)
                shape=base_reader.header.get(target+'.weight',{}).get('shape')
                ashape=reader.header[a]['shape'];bshape=reader.header[b]['shape']
                if len(ashape)!=2 or len(bshape)!=2 or any(d<=0 for d in ashape+bshape) or ashape[0]!=bshape[1] or shape!=[bshape[0],ashape[1]]:
                    raise ValueError('LoRA/base shape mismatch: '+target)
                if target in targets:raise ValueError('Duplicate canonical LoRA target: '+target)
                if reader.header[a]['dtype'] not in ('F16','BF16','F32') or reader.header[b]['dtype'] not in ('F16','BF16','F32'):
                    raise ValueError('Unsupported LoRA factor dtype')
                known.update((a,b));scale=float(strength)
                if alpha in reader.header:
                    value=reader.read_tensor(alpha)
                    if value.numel()!=1 or not torch.isfinite(value).all():raise ValueError('Invalid LoRA alpha')
                    scale=(float(value)/ashape[0])*float(strength);known.add(alpha)
                targets[target]=(a,b,scale)
            unknown=set(reader.header)-known-{'__metadata__'}
            if unknown or not targets:raise ValueError('Unconsumed LoRA keys: '+str(sorted(unknown)[:8]))
            self.sources.append((reader,float(strength),targets));self.targets.update(targets)
        identity=['native-default-row-partial-selective-b-v6',self.row_partial,self.selective_b,[(r.identity,s) for r,s,_ in self.sources]]
        self.identity=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()

    def check(self):
        for reader,_,_ in self.sources:
            stat=reader.path.stat()
            if (stat.st_size,stat.st_mtime_ns)!=reader.signature:raise RuntimeError('LoRA changed; reload the TP2 session')

    def terms(self,target):
        self.check();terms=[]
        for reader,_,targets in self.sources:
            if target not in targets:continue
            a,b,scale=targets[target]
            if scale:terms.append((reader.read_tensor(a),reader.read_tensor(b),scale))
        return terms

    def selective_rows(self,target,rank):
        """Only the physically qualified TP2 H3 FC1 / rank128 B geometry.

        Other targets, refiner and independently sized LoRAs keep full B.
        Selected rows are read/packed once on CPU before residency admission.
        """
        if not self.selective_b or rank not in (0,1) or not re.fullmatch(r'blocks\.\d+\.mlp\.fc1',target):return None
        for reader,_,targets in self.sources:
            if target not in targets:continue
            a,b,scale=targets[target]
            if scale and (reader.header[a]['shape']!=[128,5376] or reader.header[b]['shape']!=[28672,128]):return None
        return [(rank*7168,(rank+1)*7168),(14336+rank*7168,14336+(rank+1)*7168)]

    def adapter(self,target,rank=None):
        self.check();terms=[];rows=columns=None
        if rank not in (None,0,1):raise ValueError('H3 supports TP2 ranks 0/1')
        selected=self.selective_rows(target,rank)
        for reader,_,targets in self.sources:
            if target not in targets:continue
            a,b,scale=targets[target]
            if not scale:continue
            if rank is not None:
                name=target.rsplit('.',1)[-1]
                if name in ('qkv_proj','fc1'):
                    components=3 if name=='qkv_proj' else 2
                    size=reader.header[b]['shape'][0]//components
                    if size%2:raise ValueError('Packed LoRA component does not divide into TP2')
                    rows=[(c*size+rank*size//2,c*size+(rank+1)*size//2) for c in range(components)]
                elif name in ('out_proj','fc2'):
                    width=reader.header[a]['shape'][1]
                    if width%2:raise ValueError('LoRA input does not divide into TP2')
                    columns=(rank*width//2,(rank+1)*width//2)
            # Only the explicitly selected row-partial mode slices A. B keeps
            # its full Native geometry and strength is never folded into B.
            owned=columns if self.row_partial else None
            terms.append((reader.read_tensor(a,columns=owned),reader.read_tensor(b,rows=selected),scale))
        if not terms:return None
        if self.row_partial and columns is not None:
            return RowPartialLoRA(terms,columns)
        adapter=NativeAlignedLoRA(terms,None if selected is not None else rows,columns)
        adapter.selective_b=selected is not None
        adapter.selected_output_rows=selected
        return adapter

    def prefix_bytes(self,prefix,rank=None):
        total=0
        for reader,_,targets in self.sources:
            for target,(a,b,scale) in targets.items():
                if not scale or not target.startswith(prefix):continue
                ashape=reader.header[a]['shape'];bshape=reader.header[b]['shape']
                a_count=math.prod(ashape)
                if rank is not None and self.row_partial and target.rsplit('.',1)[-1] in ('out_proj','fc2'):
                    a_count//=2
                b_count=math.prod(bshape)//(2 if self.selective_rows(target,rank) is not None else 1)
                total+=4*(a_count+b_count)
        return total

    def block_bytes(self,kind,index,rank):
        return self.prefix_bytes(f'blocks.{index}.',rank) if kind=='dit' else 0

    def attach_tp(self,module,index,rank):
        for source,target in (('attn.qkv_proj','qkv_proj'),('attn.out_proj','out_proj'),('mlp.fc1','fc1'),('mlp.fc2','fc2')):
            getattr(module,target).lora_adapter=self.adapter(f'blocks.{index}.'+source,rank)
        adapter=self.adapter(f'blocks.{index}.adaln_proj.linear')
        if adapter is not None:module.adaln_proj=NativeLoraLinear(module.adaln_proj,adapter)
        return module

    def attach_refiner(self,model):
        for index,block in enumerate(model.token_refiner.blocks):
            prefix=f'token_refiner.blocks.{index}.'
            for name in ('qkv_proj','out_proj'):
                adapter=self.adapter(prefix+'attn.'+name)
                if adapter is not None:setattr(block.attn,name,NativeLoraLinear(getattr(block.attn,name),adapter))
            a=self.adapter(prefix+'mlp.fc1');b=self.adapter(prefix+'mlp.fc2')
            if a is not None or b is not None:block.mlp=NativeLoraMLP(block.mlp,a,b)
        return model
