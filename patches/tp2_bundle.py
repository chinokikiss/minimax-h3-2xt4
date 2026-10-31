"""Hash-bound base TP2 blocks plus a compact Native auxiliary checkpoint.

LoRA is prepared on CPU inside the existing byte-bounded host weight cache.
No complete merged model or persistent per-LoRA checkpoint is produced.
The legacy checkpoint/GPU offline converter remains unchanged.
"""
import hashlib
import json
import math
from pathlib import Path
import re
import threading
from types import SimpleNamespace

import torch

from patches.windowed_tp import CheckpointReader
from patches.rank_local_int8 import CONFIG, FORMAT, shell, state_hashes, check_state_schema
from patches.file_metadata import FileChecks

DESCRIPTOR = 'h3-tp2-bundle-reader-v1'
BLOCK = re.compile(r'^(model\.layers|blocks)\.(\d+)\.(.+)$')
DIT_LINEAR = {'attn.qkv_proj':'qkv_proj', 'attn.out_proj':'out_proj',
              'mlp.fc1':'fc1', 'mlp.fc2':'fc2', 'adaln_proj.linear':'adaln_proj'}


def sha(path):
    with Path(path).open('rb') as stream: return hashlib.file_digest(stream,'sha256').hexdigest()


def signature(path):
    s=Path(path).stat(); return s.st_size,s.st_mtime_ns


def open_checkpoint(path,*,verify_dit_weights=True,verify_weights=None):
    """Tooling verifies by default; inference explicitly selects metadata checks."""
    verify = verify_dit_weights if verify_weights is None else verify_weights
    if Path(path).suffix.lower()=='.json':
        checks=FileChecks()
        value=checks.read_json(path)
        if value.get('format')=='h3-tp2-disk-baked-v1':
            from patches.tp2_disk_bake import DiskBakedReader
            reader=DiskBakedReader(path,value,verify_payloads=verify)
        else:
            if value.get('format')!=DESCRIPTOR: raise ValueError('Unknown checkpoint descriptor')
            reader=BundleReader(path,value,verify_payloads=verify if verify_weights is not None or value.get('kind')=='dit' else True)
        checks.check()
        return reader
    reader = CheckpointReader(path)
    reader.verify_payloads = verify
    return reader


def split_matrix(value, name, rank, *, scale=False):
    """Preserve component-packed output ownership and full-width row scales."""
    if name in ('qkv_proj','fc1'):
        parts=3 if name=='qkv_proj' else 2
        if value.shape[0]%(parts*2): raise ValueError('Invalid packed TP2 rows')
        rows=value.shape[0]//parts; half=rows//2
        return torch.cat([value[c*rows+rank*half:c*rows+(rank+1)*half] for c in range(parts)]).contiguous()
    if name in ('out_proj','fc2') and not scale:
        if value.shape[1]%2: raise ValueError('Invalid TP2 input width')
        return value[:,rank*value.shape[1]//2:(rank+1)*value.shape[1]//2].contiguous()
    return value.clone().contiguous()


def join_matrix(first, second, name, *, scale=False):
    if name in ('qkv_proj','fc1'):
        parts=3 if name=='qkv_proj' else 2
        if first.shape[0]%parts or first.shape!=second.shape: raise ValueError('Invalid packed shard pair')
        return torch.cat([piece for pair in zip(first.chunk(parts),second.chunk(parts)) for piece in pair]).contiguous()
    if name in ('out_proj','fc2') and not scale: return torch.cat((first,second),dim=1).contiguous()
    if not torch.equal(first,second): raise ValueError('Replicated TP2 tensors/scales differ')
    return first


class BundleReader:
    def __init__(self,path,descriptor=None,*,verify_payloads=True):
        self.path=Path(path).resolve(); self.signature=signature(self.path)
        self.file_checks=FileChecks()
        value=descriptor or self.file_checks.read_json(self.path)
        self.file_checks.admit(self.path)
        if value.get('format')!=DESCRIPTOR or value.get('kind') not in ('te','dit'): raise ValueError('Invalid bundle descriptor')
        self.kind=value['kind']; self.root=(self.path.parent/value['bundle_root']).resolve()
        if type(verify_payloads) is not bool:
            raise ValueError('Explicit payload verification boolean required')
        self.verify_payloads=verify_payloads
        self.bundle_file=self.root/'bundle.json'
        if verify_payloads and sha(self.bundle_file)!=value['bundle_sha256']: raise ValueError('Bundle index checksum mismatch')
        self.bundle_signature=signature(self.bundle_file)
        bundle=self.file_checks.read_json(self.bundle_file); self.plan=bundle['plan']
        self.bundle_digest=value['bundle_sha256']  # Offline provenance label; not a runtime content proof.
        if not bundle['complete'] or self.plan['format']!=FORMAT or self.plan['configuration']!=CONFIG:
            raise ValueError('Unsupported/incomplete Native TP2 bundle')
        expected={f'{k}.block{i:02d}.rank{r}' for k in ('te','dit') for i in range(50) for r in (0,1)}
        if set(self.plan['blocks'])!=expected or set(bundle['blocks'])!=expected: raise ValueError('Incomplete TP2 layer coverage')
        if value['source']!=self.plan['sources'][self.kind]: raise ValueError('Bundle source identity mismatch')
        self.header=value['native_header']
        if verify_payloads and hashlib.sha256(json.dumps(self.header,sort_keys=True).encode()).hexdigest()!=value['native_header_sha256']:
            raise ValueError('Native header checksum mismatch')
        self.aux=CheckpointReader((self.path.parent/value['auxiliary_file']).resolve())
        if verify_payloads and sha(self.aux.path)!=value['auxiliary_sha256']: raise ValueError('Auxiliary checkpoint checksum mismatch')
        prefix='model.layers.' if self.kind=='te' else 'blocks.'
        names={n for n in self.header if n!='__metadata__' and not n.startswith(prefix)}
        if names!={n for n in self.aux.header if n!='__metadata__'}: raise ValueError('Auxiliary tensor coverage mismatch')
        for name in names:
            if any(self.header[name][k]!=self.aux.header[name][k] for k in ('dtype','shape')):
                raise ValueError('Auxiliary tensor schema mismatch')
        self.identity=('tp2-bundle-v1',str(self.path),*self.signature,value['bundle_sha256'],value['source']['sha256'],value['auxiliary_sha256'])
        if not verify_payloads:self.identity=(*self.identity,'metadata-checked-weights-v1')
        self.read_lock=threading.RLock(); self.prepare_lock=threading.Lock()
        self.logical_read_bytes=self.range_read_calls=0
        self.verified={}; self.state_verified={}; self.shard_readers={}
        self.offline_store=None; self.offline_targets=set()
        self.stats=dict(prepared_blocks=0,merged_matrices=0,persistent_baked_bytes=0,merge_device='cpu')

    def _check_roots(self):
        self.file_checks.admit(self.path)
        self.file_checks.admit(self.bundle_file)
        self.aux.check()
        if signature(self.path)!=self.signature or signature(self.bundle_file)!=self.bundle_signature:
            raise RuntimeError('Bundle descriptor/index changed; reload the session')
        if signature(self.aux.path)!=self.aux.signature: raise RuntimeError('Auxiliary weights changed')
        if self.offline_store is not None: self.offline_store.check()

    def check(self):
        """Validate the full admitted set at an explicit preparation boundary."""
        self._check_roots()
        self.file_checks.check()

    def set_offline_lora(self,stack):
        from patches.h3_lora import H3LoRAStore
        if self.kind!='dit' or self.offline_store is not None: raise ValueError('Offline bundle LoRA must be attached once to DiT')
        self.offline_store=H3LoRAStore(tuple(stack),self)
        self.offline_targets={t for _,_,mapping in self.offline_store.sources for t,(_,_,s) in mapping.items() if s}
        self.identity=(*self.identity,'cpu-native-offline-v1',self.offline_store.identity)

    def _shard(self,index,rank):
        self._check_roots(); stem=f'{self.kind}.block{index:02d}.rank{rank}'
        expected=self.plan['blocks'][stem]; manifest_path=self.root/(stem+'.json'); path=self.root/(stem+'.safetensors')
        before=(signature(path),signature(manifest_path))
        self.file_checks.admit(path)
        self.file_checks.admit(manifest_path)
        for artifact,current in zip((path,manifest_path),before):
            old=self.verified.get(str(artifact))
            if old is not None and current!=old:raise RuntimeError('Verified bundle shard/manifest changed')
        if str(manifest_path) not in self.verified:
            m=self.file_checks.read_json(manifest_path)
            self.file_checks.admit(path, size=m.get('bytes'))
            if m['format']!=FORMAT or m['configuration']!=CONFIG or m['native_tensor_bit_exact'] is not True:
                raise ValueError('Unqualified bundle shard')
            if any(m[k]!=expected[k] for k in ('kind','index','rank','state_sha256','quant_metadata')):
                raise ValueError('Shard differs from Native tensor proof')
            if self.verify_payloads and sha(path)!=m['file_sha256']: raise ValueError('Shard file checksum mismatch')
            if before!=(signature(path),signature(manifest_path)):
                raise RuntimeError('Bundle shard changed during validation')
            self.verified[str(path)],self.verified[str(manifest_path)]=before
        if stem not in self.shard_readers:self.shard_readers[stem]=CheckpointReader(path)
        return self.shard_readers[stem]

    def _target(self,name):
        match=BLOCK.fullmatch(name)
        if not match: return None
        if (match[1]=='blocks')!=(self.kind=='dit'): raise ValueError('Wrong bundle kind')
        index=int(match[2]); local=match[3]
        if self.kind=='dit':
            local=local.replace('attn.q_norm.','q_norm.').replace('attn.k_norm.','k_norm.')
            for native,target in DIT_LINEAR.items():
                if local.startswith(native+'.'): return index,target,local[len(native)+1:]
            return index,local.rsplit('.',1)[0],local.rsplit('.',1)[1]
        raise ValueError('TE full-matrix reconstruction is unnecessary; use rank blocks')

    def _raw_tensor(self,name):
        self._check_roots(); target=self._target(name)
        if target is None: value=self.aux.read_tensor(name)
        else:
            index,local,suffix=target
            if suffix=='comfy_quant':
                value=torch.tensor(list(json.dumps({'format':'int8_tensorwise','convrot':True,'convrot_groupsize':256}).encode()),dtype=torch.uint8)
            else:
                first=self._shard(index,0).read_tensor(local+'.'+suffix)
                second=self._shard(index,1).read_tensor(local+'.'+suffix)
                value=join_matrix(first,second,local,scale=suffix=='weight_scale')
                value=value.to(CheckpointReader.DTYPES[self.header[name]['dtype']])
        with self.read_lock:
            self.logical_read_bytes+=value.numel()*value.element_size(); self.range_read_calls+=1
        return value

    def read_tensor(self,name,*,rows=None,columns=None):
        if rows or columns: raise ValueError('Bundle reader exposes full Native tensors; use load_rank for owned weights')
        if (name.endswith('.weight') and name.removesuffix('.weight') in self.offline_targets
                and not name.startswith('blocks.')):
            from patches.offline_lora import merged_weight
            with self.prepare_lock:
                base=SimpleNamespace(read_tensor=self._raw_tensor,header=self.header)
                value,scale=merged_weight(base,self.offline_store,name.removesuffix('.weight'),'cpu')
                if scale is not None: raise ValueError('Unexpected INT8 global LoRA target')
                self.stats['merged_matrices']+=1
                return value
        return self._raw_tensor(name)

    def read(self,prefix,exclude=()):
        names=[n for n in self.header if n!='__metadata__' and n.startswith(prefix) and not n.startswith(tuple(exclude))]
        if not names: raise ValueError('No matching bundle tensors')
        return {n[len(prefix):]:self.read_tensor(n) for n in names}

    def bytes(self,prefix):
        return sum(v['data_offsets'][1]-v['data_offsets'][0] for n,v in self.header.items() if n!='__metadata__' and n.startswith(prefix))

    def workspace_bytes(self,index):
        if not self.offline_targets: return 0
        tensors=[v for n,v in self.header.items() if n.startswith(f'blocks.{index}.') and n.endswith('.weight')
                 and n.removesuffix('.weight') in self.offline_targets]
        # Quant/dequant, Native LoRA delta and dense workspace; one matrix at a time.
        factors=0
        for source,_,mapping in self.offline_store.sources:
            for target,(a,b,strength) in mapping.items():
                if strength and target.startswith(f'blocks.{index}.'):
                    factors+=8*(math.prod(source.header[a]['shape'])+math.prod(source.header[b]['shape']))
        return max((math.prod(v['shape'])*32 for v in tensors),default=0)+factors+16*2**20

    def load_rank(self,index,rank):
        from patches.offline_lora import merged_weight
        with self.prepare_lock:
            self.check()
            source=self._shard(index,rank); module=shell(self.kind,rank)
            state={n:source.read_tensor(n) for n in module.state_dict()}
            expected=self.plan['blocks'][f'{self.kind}.block{index:02d}.rank{rank}']
            check_state_schema(state,expected['state_sha256'])
            # Runtime checks schemas/metadata; offline tools additionally prove
            # payload hashes once per immutable file identity, before merging.
            module.load_state_dict(state,strict=True,assign=True)
            stem=f'{self.kind}.block{index:02d}.rank{rank}'
            proof=(self.verified[str(source.path)],self.verified[str(self.root/(stem+'.json'))])
            if self.verify_payloads and self.state_verified.get(stem)!=proof:
                if state_hashes(module)!=expected['state_sha256']: raise ValueError('Rank tensors differ from Native proof')
                self._shard(index,rank)  # Reject mutation during the payload proof.
                self.state_verified[stem]=proof
            del module
            if self.kind=='dit' and self.offline_targets:
                base=SimpleNamespace(read_tensor=self._raw_tensor,header=self.header)
                for native,local in DIT_LINEAR.items():
                    target=f'blocks.{index}.{native}'
                    if target not in self.offline_targets: continue
                    q,scale=merged_weight(base,self.offline_store,target,'cpu')
                    state[local+'.weight']=split_matrix(q,local,rank)
                    if scale is not None: state[local+'.weight_scale']=split_matrix(scale,local,rank,scale=True)
                    del q,scale
                    self.stats['merged_matrices']+=1
            result=shell(self.kind,rank); result.load_state_dict(state,strict=True,assign=True)
            result._rank_checkpoint_metadata=expected['quant_metadata']
            # Cached proofs still require unchanged identities after the read,
            # including mutation during the last tensor/LoRA preparation.
            self.check()
            self.stats['prepared_blocks']+=1
            return result.eval().requires_grad_(False)
