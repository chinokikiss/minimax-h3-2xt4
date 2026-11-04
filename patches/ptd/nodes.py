"""Optional PTD graph nodes; disabled adapters return the exact input MODEL."""
from pathlib import Path


class MiniMaxH3TP2PTDAdapter:
    CATEGORY = 'MiniMax H3/TP2/PTD'
    FUNCTION = 'apply'
    RETURN_TYPES = ('MODEL','H3_PTD_CONTROLLER')

    @classmethod
    def INPUT_TYPES(cls):
        return {'required':{
            'model':('MODEL',),'session':('MINIMAX_TP2_SESSION',),'bank':('H3_PTD_BANK',),
            'enabled':('BOOLEAN',{'default':False}),
            'allow_pruned_adaln':('BOOLEAN',{'default':False}),
            'use_hop_refiner':('BOOLEAN',{'default':True}),
            'use_pdd_heads':('BOOLEAN',{'default':True}),
            'audio_chord_correction':('BOOLEAN',{'default':True}),
            'hop_handoff':(['queue','rebuild'],{'default':'queue',
                'tooltip':'Queue keeps one shell/window and retires weights by CUDA event. Rebuild is the explicit comparison path.'}),
            'stream_weights':('BOOLEAN',{'default':True,
                'tooltip':'Stream one active expert. Disabling requests resident active-expert weights and requires VRAM admission.'})},
            'optional':{
            'future_cache':('BOOLEAN',{'default':True,
                'tooltip':'Experimental known-sequence CPU reclamation; preserves CUDA lifetime fences.'}),
            'prefetch_refiner':('BOOLEAN',{'default':True,
                'tooltip':'Budgeted next-hop CPU refiner lane; disable for the no-prefetch comparison. Requires future_cache.'})}}

    def apply(self,model,session,bank,enabled=False,allow_pruned_adaln=False,
              use_hop_refiner=True,use_pdd_heads=True,audio_chord_correction=True,stream_weights=True,hop_handoff='queue',
              future_cache=True,prefetch_refiner=True,expected_bank_sha256=''):
        from patches.ptd.runtime import PTDOptions,adapt_model
        options=PTDOptions(enabled=enabled,allow_pruned_adaln=allow_pruned_adaln,use_hop_refiner=use_hop_refiner,
            use_pdd_heads=use_pdd_heads,audio_chord_correction=audio_chord_correction,
            stream_weights=stream_weights,hop_handoff=hop_handoff,
            future_cache=future_cache,prefetch_refiner=prefetch_refiner)
        return adapt_model(model,session,bank,options,expected_bank_sha256=expected_bank_sha256 or None)


class MiniMaxH3TP2PTDBankLoader:
    """Explicit Base-free Bank v2 entry. The ordinary Loader is unchanged."""
    CATEGORY = 'MiniMax H3/TP2/PTD'
    FUNCTION = 'load'
    RETURN_TYPES = ('MODEL', 'CLIP', 'MINIMAX_TP2_SESSION', 'H3_PTD_BANK')

    @classmethod
    def INPUT_TYPES(cls):
        from custom_nodes.minimax_h3_tp2 import MiniMaxH3TP2Loader
        original = MiniMaxH3TP2Loader.INPUT_TYPES()
        required = dict(original['required'])
        required.pop('diffusion_model')
        required['bank_manifest'] = ('STRING', {'default': '/kaggle/input/h3-ptd-bank-v2/bank_v2.json'})
        optional = dict(original['optional'])
        optional.pop('lora_stack', None)
        optional.pop('lora_mode', None)
        optional['part_roots_json'] = ('STRING', {'default': '{}',
            'tooltip': 'Explicit immutable part-id to mounted directory map. Relative roots in the manifest need no override.'})
        optional['ptd_io_workers'] = ('INT', {'default': 1, 'min': 1, 'max': 8, 'step': 1,
            'tooltip': 'Experimental v2-only CPU reader threads per rank. Four/rank means eight total readers.'})
        optional['ptd_retire_page_cache'] = ('BOOLEAN', {'default': False,
            'tooltip': 'Experimental Linux posix_fadvise DONTNEED after complete v2 rank reads; no inference data change.'})
        return {'required': required, 'optional': optional}

    def load(self, text_encoder, bank_manifest, part_roots_json='{}',
             ptd_io_workers=1, ptd_retire_page_cache=False, **options):
        import json
        from custom_nodes.minimax_h3_tp2 import MiniMaxH3TP2Loader
        from patches.ptd.bank_v2 import PTDBankV2
        if options.pop('lora_stack', ()):
            raise ValueError('Bank v2 already includes all PTD and ordinary LoRA merges')
        if options.pop('lora_mode', 'offline') != 'offline':
            raise ValueError('Bank v2 forbids online LoRA; prepare another offline Bank instead')
        mounts = json.loads(part_roots_json)
        if (not isinstance(mounts, dict)
                or any(not isinstance(k, str) or not isinstance(v, str) or not v
                       for k, v in mounts.items())):
            raise ValueError('Bank v2 part roots must be a JSON object of ID/path strings')
        path = Path(bank_manifest).resolve()
        PTDBankV2(path, part_roots=tuple(sorted(mounts.items())), verify_payloads=False).configure_io(
            ptd_io_workers, ptd_retire_page_cache)
        # 'online' here only suppresses the legacy Loader's JSON-based offline
        # LoRA installation. No runtime adapters exist in this v2 session.
        model, clip, service = MiniMaxH3TP2Loader().load(
            text_encoder, str(path), lora_mode='online',
            ptd_part_roots=tuple(sorted(mounts.items())),
            ptd_io_workers=ptd_io_workers, ptd_retire_page_cache=ptd_retire_page_cache,
            **options)
        return model, clip, service, str(path)


class MiniMaxH3TP2PTDSampler:
    CATEGORY = 'MiniMax H3/TP2/PTD'
    FUNCTION = 'sampler'
    RETURN_TYPES = ('SAMPLER','SIGMAS')

    @classmethod
    def INPUT_TYPES(cls):
        return {'required':{'controller':('H3_PTD_CONTROLLER',),'enabled':('BOOLEAN',{'default':True})},
                'optional':{'native_sampler':('SAMPLER',),'native_sigmas':('SIGMAS',)}}

    def sampler(self,controller,enabled=True,native_sampler=None,native_sigmas=None):
        if not enabled or controller is None:
            if native_sampler is None or native_sigmas is None:
                raise ValueError('Connect the original Native sampler and sigmas when PTD is disabled')
            return native_sampler,native_sigmas
        from patches.ptd.sampler import make_sampler
        return make_sampler(controller),controller.sigmas.float()


NODE_CLASS_MAPPINGS={cls.__name__:cls for cls in
    (MiniMaxH3TP2PTDBankLoader,MiniMaxH3TP2PTDAdapter,MiniMaxH3TP2PTDSampler)}
