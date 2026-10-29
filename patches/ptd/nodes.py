"""Optional PTD graph nodes; disabled adapters return the exact input MODEL."""
from pathlib import Path


class MiniMaxH3TP2PTDBake:
    CATEGORY = 'MiniMax H3/TP2/PTD'
    FUNCTION = 'prepare'
    RETURN_TYPES = ('H3_PTD_BANK',)

    @classmethod
    def INPUT_TYPES(cls):
        return {'required':{
            'enabled':('BOOLEAN',{'default':False}),
            'base_descriptor':('STRING',{'default':'/tmp/h3_base/dit_reader.json'}),
            'checkpoint_path':('STRING',{'default':'download_on_t4'}),
            'output_directory':('STRING',{'default':'/tmp/h3_ptd_refasymopd'}),
            'allow_pruned_adaln':('BOOLEAN',{'default':False,
                'tooltip':'Explicitly allow the current pruned AdaLN bundle as an unqualified PTD approximation.'})},
            'optional':{'session':('MINIMAX_TP2_SESSION',),
                'runtime_layout':('BOOLEAN',{'default':False,
                    'tooltip':'Experimental FP32 refiner sidecars for exact Native FP16/e6 loading/folding; increases disk usage.'})}}

    def prepare(self,enabled,base_descriptor,checkpoint_path,output_directory,allow_pruned_adaln=False,session=None,
                runtime_layout=False):
        if not enabled:
            return (None,)
        from patches.ptd.bank import PTDBank,bake,disk_plan
        from patches.ptd.checkpoint import download_on_t4
        from patches.tp2_bundle import BundleReader
        base = BundleReader(base_descriptor)
        if 'adaln_t_table' in base.header and not allow_pruned_adaln:
            raise ValueError('Enable the explicit pruned-AdaLN approximation switch for this bundle')
        cached = Path(output_directory) / 'bank.json'
        if cached.exists():
            bank=PTDBank(cached)
            if bank.base != base.path:
                raise ValueError('Cached PTD bank uses another base')
            from patches.ptd.checkpoint import RELEASE
            from patches.tp2_bundle import sha
            expected = RELEASE['sha256'] if checkpoint_path == 'download_on_t4' else sha(Path(checkpoint_path))
            if expected != bank.recipe['checkpoint_sha256']:
                raise ValueError('Cached PTD bank uses another checkpoint')
            if runtime_layout and 'runtime_refiner_layout' not in bank.recipe:
                raise ValueError('Cached bank has no runtime layout; choose a new bake directory')
            return (str(cached.resolve()),)
        plan=disk_plan(base.path,output_directory,include_download=checkpoint_path=='download_on_t4',
                       runtime_layout=runtime_layout)
        if plan['free_bytes'] < plan['required_free_bytes']:
            raise OSError(f"Eight PTD experts need {plan['required_free_bytes']/2**30:.1f} GiB free; only {plan['free_bytes']/2**30:.1f} GiB available")
        def run():
            checkpoint=(download_on_t4(Path(output_directory).parent/'ptd_download')
                        if checkpoint_path=='download_on_t4' else Path(checkpoint_path))
            return bake(base.path,checkpoint,output_directory,allow_pruned_adaln=allow_pruned_adaln,
                         runtime_layout=runtime_layout)
        if session is None:
            return (str(run()),)
        with session.lock:
            session.release_staged_encoders()
            session.call('end_sample')  # release the existing TE/DiT GPU window
            session.call('vae_release')
            return (str(run()),)


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
            'future_cache':('BOOLEAN',{'default':False,
                'tooltip':'Experimental known-sequence CPU reclamation; preserves CUDA lifetime fences.'}),
            'prefetch_refiner':('BOOLEAN',{'default':False,
                'tooltip':'Experimental budgeted next-hop CPU refiner lane; requires future_cache.'}),
            'baked_refiner':('BOOLEAN',{'default':False,
                'tooltip':'Use independently verified runtime-ready refiner weights when baked for the exact tensor flow.'})}}

    def apply(self,model,session,bank,enabled=False,allow_pruned_adaln=False,
              use_hop_refiner=True,use_pdd_heads=True,audio_chord_correction=True,stream_weights=True,hop_handoff='queue',
              future_cache=False,prefetch_refiner=False,baked_refiner=False):
        from patches.ptd.runtime import PTDOptions,adapt_model
        options=PTDOptions(enabled=enabled,allow_pruned_adaln=allow_pruned_adaln,use_hop_refiner=use_hop_refiner,
            use_pdd_heads=use_pdd_heads,audio_chord_correction=audio_chord_correction,
            stream_weights=stream_weights,hop_handoff=hop_handoff,
            future_cache=future_cache,prefetch_refiner=prefetch_refiner,baked_refiner=baked_refiner)
        return adapt_model(model,session,bank,options)


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
    (MiniMaxH3TP2PTDBake,MiniMaxH3TP2PTDAdapter,MiniMaxH3TP2PTDSampler)}
