"""Bounded PTD qualification, not a complete H3 generation or quality test."""
import argparse
from contextlib import closing, contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

SCHEDULING_CASES = (
    ('queue_baseline', {}),
    ('future_cache', dict(future_cache=True)),
    ('future_refiner', dict(future_cache=True,prefetch_refiner=True)),
    ('future_refiner_repeat', dict(future_cache=True,prefetch_refiner=True)),
    ('queue_baseline_repeat', {}))



def resource_plan(base=None, *, header=None):
    from types import SimpleNamespace
    from patches.tp2_bundle import BundleReader
    from patches.rank_local_int8 import rank_block_bytes
    reader=BundleReader(base) if header is None else SimpleNamespace(header=header)
    refiner=sum(math.prod(info['shape'])*4 for name,info in reader.header.items()
        if name.startswith(('condition_proj.','token_refiner.')))+2**20
    block=max(rank_block_bytes(reader,'dit',i,r)+2**20 for i in (0,1) for r in (0,1))
    return dict(refiner_bound=refiner,block_bound=block,weight_bytes=max(6*2**30,refiner+2*block),
        cache_bytes=max(4*2**30,refiner+block),pinned_bytes=max(4*2**30,2*block),
        host_bytes=20*2**30,reserve_bytes=4*2**30)


def smoke(base, bank, output, handoff, resources, cpu_options=None):
    # Controls must remain explicit after promoting P1 as the runtime default.
    cpu_options=dict(future_cache=False,prefetch_refiner=False)|dict(cpu_options or {})
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
    config = RuntimeConfig(str(te), str(base), layers=2,
        **{key:value for key,value in resources.items() if key.endswith('_bytes')},
        cross_request_te=False,cross_model_cpu=False,timeout_seconds=420)
    generator = torch.Generator().manual_seed(20261028)
    noise = NestedTensor([torch.randn(1,24,1,4,4,generator=generator),
                          torch.randn(1,32,2,8,generator=generator)])
    text = torch.randn(1,4,5120,generator=generator)*10
    layout = PackedLayout(4,1,4,4,8)
    timings, handoffs, queue_states, hashes = [], [], [], []
    class TimedController(Controller):
        @contextmanager
        def activate(self,hop,sigma):
            begin=time.perf_counter()
            with super().activate(hop,sigma):
                handoffs.append(time.perf_counter()-begin)
                yield
    from unittest.mock import patch
    import patches.comfy_tp_runtime as runtime
    from ptd_worker_probe import worker
    os.environ['H3_PTD_PROBE_DIR']=str(output)
    with patch.object(runtime,'_worker',worker), closing(PhysicalTPService(config)) as service:
        controller = TimedController(service,bank,PTDOptions(enabled=True,allow_fixture=True,
            allow_pruned_adaln=True,hop_handoff=handoff,**(cpu_options or {})))
        # Compilation is explicit and separate from expert execution. The
        # real H3 rotation is 96/128, not the historical 128/128 rehearsal.
        start=time.monotonic(); aot=service.call('prepare_dit',layout.seq_len)
        preparation=time.monotonic()-start
        (output/'preparation.json').write_text(json.dumps(dict(aot=aot,seconds=preparation,
            resources=resources,hop_handoff=handoff),indent=2),encoding='utf-8')
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
                        hashes.append([hashlib.sha256(value.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
                                       for value in values])
                        queue_states.append(service.call('ptd_state')['queue'])
                        # Native FLOW_AV x0 convention, retained around the
                        # actual TP2 Native body rather than replacing it.
                        return NestedTensor([x-sigma.float().reshape((-1,)+(1,)*(x.ndim-1))*v
                                             for x,v in zip(state.unbind(),values)])
                final=sample_ptd(Denoiser(),noise,controller.sigmas.float(),controller=controller)
                if any(not torch.isfinite(v).all() for v in final.unbind()):
                    raise ValueError('Nonfinite PTD FP32 state carry')
                selected=service.call('ptd_state')
                assert selected['calls']==list(range(8))
                if handoff=='queue':
                    assert len({state['window_id'] for state in queue_states})==1
                    assert len({state['shell_id'] for state in queue_states})==1
                stats=service.call('stats')
            finally:
                if not service.closed: service.call('end_sample')
        finally:
            if not service.closed: controller.end()
        # Restoration must let the ordinary reader be used after the sample.
        restored=service.call('stats')
        assert all(len(row['ptd_probe']['spans'])==16 and all(span['default_stream_interval_ms'] is not None
            for span in row['ptd_probe']['spans']) for row in restored), 'Incomplete worker metric capture'
    return dict(status='complete',hop_handoff=handoff,hops=selected['calls'],hop_seconds=timings,
        cpu_options=cpu_options or {},
        selection_seconds=handoffs,hop_output_sha256=hashes,queue_states=queue_states,
        final_sha256=[hashlib.sha256(v.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest() for v in final.unbind()],
        aot_wall_seconds=preparation,aot_rank0=aot,rank_stats=stats,restored_stats=restored,
        final_shapes=[list(v.shape) for v in final.unbind()],finite=True,
        real_trained_layers=2,seeded_operator_inputs=True,raw_te_fp32_payload=True,
        pruned_adaln_approximation=True,full_pipeline=False,quality_qualified=False)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--cpu-preflight',action='store_true')
    parser.add_argument('--cpu-scheduling',action='store_true')
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
            names=[
                'benchmarks.test_ptd_tp2.AdapterContracts.test_node_registration_under_dynamic_loader_names',
                'benchmarks.test_ptd_tp2.AdapterContracts.test_disabled_returns_native_model_sampler_and_sigmas_by_identity',
                'benchmarks.test_ptd_queue']
            if args.cpu_scheduling:names+=['benchmarks.test_ptd_cpu_schedule']
            contracts=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromNames(names))
            assert contracts.wasSuccessful()
            assets=json.loads((source/'bundle_assets.json').read_text())
            state['resources']=resource_plan(header=assets['auxiliary']['dit']['native_header'])
            # Setup/API admission only. No checkpoint download or GPU quota.
            state.update(status='ptd_cpu_ready',core_api_imports=True,no_weights_fetched=True,
                         registration_contract_tests=2,queue_contract_tests=7,
                         scheduling_contract_tests=11 if args.cpu_scheduling else 0,
                         cpu_scheduling=args.cpu_scheduling)
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
            OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',MALLOC_ARENA_MAX='2',MINIMAX_H3_PROFILE_DETAILS='1')
        torch.set_num_threads(2)
        names=['benchmarks.test_ptd_tp2','benchmarks.test_ptd_queue']
        if args.cpu_scheduling:names+=['benchmarks.test_ptd_cpu_schedule']
        result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromNames(names))
        assert result.wasSuccessful(); state['contract_tests']=result.testsRun
        state['status']='staging_public_subset';save()
        from tp2_public_dit_assets import stage
        from ptd_t4_assets import fetch_subset
        scratch=Path('/tmp/h3_ptd_qualification_20261028');scratch.mkdir(exist_ok=False)
        base=stage(source,scratch/'base',layers=(0,1))
        state['resources']=resource_plan(base)
        for device in (0,1):
            free,_=torch.cuda.mem_get_info(device)
            assert free>=state['resources']['weight_bytes']+state['resources']['reserve_bytes'], 'Required group cannot fit on T4'
        save()  # Resource admission precedes PTD download and baking.
        from patches.ptd.bank import bake,disk_plan,PTDBank
        state['disk_plan']=disk_plan(base,scratch/'bank',layers=2)
        assert state['disk_plan']['free_bytes']>=state['disk_plan']['required_free_bytes'], 'Insufficient disk for comparison banks'
        checkpoint=fetch_subset(scratch/'adapter')
        state['subset_provenance']=json.loads((scratch/'adapter/provenance.json').read_text())
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
        state['smoke_cases']=[]
        cases=(SCHEDULING_CASES if args.cpu_scheduling else
               tuple((handoff,{}) for handoff in ('rebuild','queue','queue','rebuild')))
        for index,(label,options) in enumerate(cases):
            output=args.output/f'{index}_{label}';output.mkdir()
            handoff='queue' if args.cpu_scheduling else label
            value=smoke(base,bank,output,handoff,state['resources'],options)
            value['case']=label
            state['smoke_cases'].append(value);save()
            first=state['smoke_cases'][0]
            assert value['hop_output_sha256']==first['hop_output_sha256'], 'Handoff changed a denoiser output'
            assert value['final_sha256']==first['final_sha256'], 'Handoff changed the FP32 trajectory'
        state['handoff_outputs_byte_exact']=True
        state['status']='complete'
    except BaseException as exc:
        state.update(status='failed',error=repr(exc))
        (args.output/'exception.txt').write_text(traceback.format_exc(),encoding='utf-8')
        raise
    finally: save()


if __name__=='__main__': main()
