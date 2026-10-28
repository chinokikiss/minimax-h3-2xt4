"""Bounded PTD qualification, not a complete H3 generation or quality test."""
import argparse
from contextlib import closing
import json
import os
from pathlib import Path
import sys
import time
import traceback


def smoke(base, bank, output):
    import torch
    from safetensors.torch import save_file
    from comfy.nested_tensor import NestedTensor
    from comfy.ldm.minimax.model import PackedLayout
    from patches.comfy_tp_runtime import RuntimeConfig, PhysicalTPService
    from patches.ptd.runtime import Controller, PTDOptions
    from patches.ptd.sampler import sample_ptd
    # The TE is deliberately never executed. An empty, valid file satisfies
    # the service's metadata contract without downloading another model.
    te = output/'unused_te.safetensors'; save_file({}, str(te))
    config = RuntimeConfig(str(te), str(base), layers=2, weight_bytes=2*2**30,
        host_bytes=12*2**30, cache_bytes=2*2**30, pinned_bytes=2*2**30,
        reserve_bytes=4*2**30, cross_request_te=False, cross_model_cpu=False, timeout_seconds=420)
    generator = torch.Generator().manual_seed(20261028)
    noise = NestedTensor([torch.randn(1,24,1,4,4,generator=generator),
                          torch.randn(1,32,2,8,generator=generator)])
    text = torch.randn(1,4,5120,generator=generator)*10
    layout = PackedLayout(4,1,4,4,8)
    timings = []
    with closing(PhysicalTPService(config)) as service:
        controller = Controller(service,bank,PTDOptions(enabled=True,allow_fixture=True,allow_pruned_adaln=True))
        # Compilation is explicit and separate from expert execution. The
        # real H3 rotation is 96/128, not the historical 128/128 rehearsal.
        start=time.monotonic(); aot=service.call('prepare_dit',layout.seq_len)
        preparation=time.monotonic()-start
        controller.begin()
        try:
            service.call('begin_sample')
            try:
                class Denoiser:
                    latent_image = None
                    def __init__(self): self.noise=noise
                    def __call__(self,state,sigma,**kwargs):
                        begin=time.monotonic()
                        values=service.call('denoise',list(state.unbind()),sigma.float()*1000,
                            context=torch.zeros(1,4,5120,dtype=torch.float16),
                            minimax_payload={'layout':layout,'ptd_text_states':text,'seed':20261028},
                            transformer_options={})
                        if len(values)!=2 or any(not torch.isfinite(v).all() for v in values):
                            raise ValueError('Nonfinite PTD denoiser output')
                        timings.append(time.monotonic()-begin)
                        # Native FLOW_AV x0 convention, retained around the
                        # actual TP2 Native body rather than replacing it.
                        return NestedTensor([x-sigma.float().reshape((-1,)+(1,)*(x.ndim-1))*v
                                             for x,v in zip(state.unbind(),values)])
                final=sample_ptd(Denoiser(),noise,controller.sigmas.float(),controller=controller)
                if any(not torch.isfinite(v).all() for v in final.unbind()):
                    raise ValueError('Nonfinite PTD FP32 state carry')
                selected=service.call('ptd_state')
                assert selected['calls']==list(range(8))
                stats=service.call('stats')
            finally:
                if not service.closed: service.call('end_sample')
        finally:
            if not service.closed: controller.end()
        # Restoration must let the ordinary reader be used after the sample.
        restored=service.call('stats')
    return dict(status='complete',hops=selected['calls'],hop_seconds=timings,
        aot_wall_seconds=preparation,aot_rank0=aot,rank_stats=stats,restored_stats=restored,
        final_shapes=[list(v.shape) for v in final.unbind()],finite=True,
        real_trained_layers=2,seeded_operator_inputs=True,raw_te_fp32_payload=True,
        pruned_adaln_approximation=True,full_pipeline=False,quality_qualified=False)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--cpu-preflight',action='store_true')
    args=parser.parse_args(); args.output.mkdir(parents=True,exist_ok=True)
    source=args.source.resolve(); sys.path[:0]=[str(source),str(source/'ComfyUI'),str(source/'benchmarks')]
    sys.argv=['ptd-tp2','--cpu'] if args.cpu_preflight else ['ptd-tp2']
    import comfy.options
    comfy.options.enable_args_parsing()
    import comfy.cli_args
    from patches.legacy_runtime import validate_packages,validate_source
    validate_packages();validate_source(source/'ComfyUI')
    manifest=json.loads((source.parent/'h3_light_distribution/delivery_manifest.json').read_text())
    state=dict(status='starting',capsule_sha256=manifest['capsule_sha256'],full_pipeline=False,
        no_local_lora_download=True,scope='2 trained blocks, both refiners, full head banks, all 8 hops')
    def save(): (args.output/'result.json').write_text(json.dumps(state,indent=2),encoding='utf-8')
    try:
        import unittest
        if args.cpu_preflight:
            import inspect
            from patches.ptd.runtime import PTDOptions,adapt_model
            from patches.ptd.bank import disk_plan
            from patches.comfy_sampler import create_model
            import comfy.model_base
            native_forward=comfy.model_base.MiniMaxH3._apply_model
            assert PTDOptions().enabled is False and PTDOptions().allow_pruned_adaln is False
            assert adapt_model(native_forward,None,None,PTDOptions())==(native_forward,None)
            assert 'adapter' in inspect.signature(create_model).parameters
            assert 'include_download' in inspect.signature(disk_plan).parameters
            contracts=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromNames([
                'benchmarks.test_ptd_tp2.AdapterContracts.test_node_registration_under_dynamic_loader_names',
                'benchmarks.test_ptd_tp2.AdapterContracts.test_disabled_returns_native_model_sampler_and_sigmas_by_identity']))
            assert contracts.wasSuccessful()
            # Setup/API admission only. No checkpoint download or GPU quota.
            state.update(status='ptd_cpu_ready',core_api_imports=True,no_weights_fetched=True,
                         registration_contract_tests=contracts.testsRun)
            return
        import torch
        import importlib.metadata as md
        devices=[dict(name=torch.cuda.get_device_name(i),capability=list(torch.cuda.get_device_capability(i)),
            uuid=str(torch.cuda.get_device_properties(i).uuid)) for i in range(torch.cuda.device_count())]
        assert len(devices)==2 and len({d['uuid'] for d in devices})==2
        assert all('T4' in d['name'] and d['capability']==[7,5] for d in devices)
        state['hardware']=dict(devices=devices,torch=str(torch.__version__),cuda=torch.version.cuda,
            kitchen=md.version('comfy-kitchen'),tilelang=md.version('tilelang'))
        assert torch.version.cuda=='12.8'
        os.environ.update(HF_HUB_DISABLE_XET='1',HF_HUB_DISABLE_PROGRESS_BARS='1',
            OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',MALLOC_ARENA_MAX='2')
        torch.set_num_threads(2)
        result=unittest.TextTestRunner(verbosity=2).run(
            unittest.defaultTestLoader.loadTestsFromName('benchmarks.test_ptd_tp2'))
        assert result.wasSuccessful(); state['contract_tests']=result.testsRun
        state['status']='staging_public_subset';save()
        from tp2_public_dit_assets import stage
        from ptd_t4_assets import fetch_subset
        scratch=Path('/tmp/h3_ptd_qualification_20261028');scratch.mkdir(exist_ok=False)
        base=stage(source,scratch/'base',layers=(0,1))
        checkpoint=fetch_subset(scratch/'adapter')
        state['subset_provenance']=json.loads((scratch/'adapter/provenance.json').read_text())
        from patches.ptd.bank import bake,disk_plan,PTDBank
        state['disk_plan']=disk_plan(base,scratch/'bank',layers=2)
        state['status']='baking_eight_two_layer_experts';save()
        begin=time.monotonic();bank=bake(base,checkpoint,scratch/'bank',layers=2,allow_pruned_adaln=True)
        state['bake_seconds']=time.monotonic()-begin
        admitted=PTDBank(bank,allow_fixture=True)
        state['verified_targets_per_hop']=[len(row['native_verified_targets']) for _,row in admitted.hops]
        state['baked_bytes']=sum(sum(r['bytes'] for r in row['shards'].values())+row['auxiliary']['bytes']
                                 for _,row in admitted.hops)
        try: PTDBank(bank)
        except ValueError: state['deployment_fixture_rejected']=True
        else: raise AssertionError('Fixture bank was incorrectly admitted for deployment')
        torch.cuda.empty_cache()
        state['status']='eight_hop_native_tp2_smoke';save()
        state['smoke']=smoke(base,bank,args.output)
        state['status']='complete'
    except BaseException as exc:
        state.update(status='failed',error=repr(exc))
        (args.output/'exception.txt').write_text(traceback.format_exc(),encoding='utf-8')
        raise
    finally: save()


if __name__=='__main__': main()
