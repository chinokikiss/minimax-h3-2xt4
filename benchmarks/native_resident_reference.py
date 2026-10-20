"""Native DiT operators with the previously validated two-device placement.

Load each block once, refine text before resident blocks occupy VRAM, and
return Native outputs on the sampler's device. This is a reference loading
policy, not tensor or pipeline parallel execution.
"""
import gc
import time
import torch
from torch import nn
import retest
from full_models import PlacedBlock, PlacedFinal, PLACEMENT
from patches.comfy_tp_runtime import native_dit
from patches.weight_residency import copy_module


class ResidentNativeDiT(nn.Module):
    dtype=torch.float32
    patch_size=(1,2,2)
    hidden_size=5376

    def __init__(self,reader,layers,devices=('cuda:0','cuda:1'),placed_factory=None,profile_loading=False):
        super().__init__()
        self.reader,self.layers,self.devices=reader,layers,devices
        self.inner=None
        self.canonical={}
        self.loaded=False
        self.placed_factory=placed_factory or PlacedBlock
        self.profile_loading=profile_loading;self.header_load_seconds=None;self.block_load_seconds=None

    def load(self):
        if self.inner is None:
            start=time.perf_counter()
            import comfy.model_management
            comfy.model_management.unload_all_models()
            self.inner=native_dit(self.reader)
            names=('condition_proj','token_refiner')
            self.canonical={name:getattr(self.inner,name) for name in names}
            for name,child in list(self.inner.named_children()):
                if name not in names:
                    setattr(self.inner,name,copy_module(child,lambda t:t.to(self.devices[0])))
            for name,value in list(self.inner._parameters.items()):
                if value is not None:
                    self.inner._parameters[name]=nn.Parameter(value.to(self.devices[0]),requires_grad=False)
            for name,value in list(self.inner._buffers.items()):
                if value is not None:self.inner._buffers[name]=value.to(self.devices[0])
            if self.profile_loading:
                torch.cuda.synchronize(self.devices[0]);self.header_load_seconds=time.perf_counter()-start
        return self.inner

    def preprocess_text_embeds(self,context):
        model=self.load()
        if context.shape[-1]==model.hidden_size:return context
        for name,canonical in self.canonical.items():
            setattr(model,name,copy_module(canonical,lambda t:t.to(self.devices[0])))
        try:
            return model.preprocess_text_embeds(context)
        finally:
            torch.cuda.synchronize(self.devices[0])
            for name,canonical in self.canonical.items():setattr(model,name,canonical)
            # Inactive refiner allocations must not compete with either
            # device's resident Native blocks and full attention scratch.
            torch.cuda.empty_cache()

    def forward(self,*args,**kwargs):
        model=self.load()
        if not self.loaded:
            start=time.perf_counter()
            split=(self.layers+1)//2
            blocks=[]
            for i in range(self.layers):
                device=self.devices[0 if i<split else 1]
                block=retest.load_native_block('dit',self.reader.path,i,device)
                blocks.append(self.placed_factory(block,'dit',device,i))
                if i%10==9:print('AUDIT NATIVE resident_blocks',i+1,flush=True)
            model.blocks=nn.ModuleList(blocks)
            model.final_layer=PlacedFinal(model.final_layer,self.devices[0])
            self.loaded=True
            if self.profile_loading:
                for device in self.devices:torch.cuda.synchronize(device)
                self.block_load_seconds=time.perf_counter()-start
        previous=getattr(PLACEMENT,'cache',None)
        PLACEMENT.cache={}
        try:return model(*args,**kwargs)
        finally:PLACEMENT.cache=previous

    def release(self):
        for device in set(self.devices):torch.cuda.synchronize(device)
        # Drop GPU blocks instead of copying the entire 20 GB model back
        # into a 32 GB host at the VAE handoff.
        self.inner=None
        self.canonical.clear()
        self.loaded=False
        gc.collect()
        for device in set(self.devices):
            with torch.cuda.device(device):torch.cuda.empty_cache()
