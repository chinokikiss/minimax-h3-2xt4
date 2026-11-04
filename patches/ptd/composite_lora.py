"""Offline-only PTD + ordinary H3 LoRA composition.

The Native merge/ConvRot path consumes these terms once at full matrix width.
There is no runtime remerge or second INT8 quantization.
"""
import hashlib
import json

from patches.h3_lora import H3LoRAStore
from .checkpoint import HopStore


class CompositeLoRAStore:
    def __init__(self, checkpoint, hop, base_reader, ordinary_loras=()):
        self.ptd = HopStore(checkpoint, hop)
        self.ordinary = (H3LoRAStore(tuple(ordinary_loras), base_reader)
                         if ordinary_loras else None)
        self.targets = set(self.ptd.targets)
        if self.ordinary is not None:
            self.targets.update(self.ordinary.targets)
        fingerprint = dict(ptd_hop=hop, ordinary=self.ordinary.identity
                           if self.ordinary is not None else None)
        self.identity = hashlib.sha256(json.dumps(fingerprint, sort_keys=True).encode()).hexdigest()

    def terms(self, target):
        result = list(self.ptd.terms(target))
        if self.ordinary is not None:
            result.extend(self.ordinary.terms(target))
        return result

    def check(self):
        self.ptd.check()
        if self.ordinary is not None:
            self.ordinary.check()
