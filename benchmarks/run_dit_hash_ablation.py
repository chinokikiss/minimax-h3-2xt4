"""Real baked PTD hash A/B on two T4s; no TE/VAE or full H3 generation."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time
import traceback
import unittest

CASES=(('verified_0',True),('trusted_0',False),('trusted_1',False),
       ('verified_1',True),('verified_2',True),('trusted_2',False))


def execution_seconds(case):
    return sum(case['selection_seconds'])+sum(case['hop_seconds'])


def summary(cases):
    values={flag:[execution_seconds(c) for c in cases if c['verify_dit_weights']==flag] for flag in (True,False)}
    if any(len(rows)!=3 for rows in values.values()):
        return dict(status='insufficient_repeats',completed_cases=len(cases))
    a,b=(statistics.median(values[flag]) for flag in (True,False))
    return dict(status='complete',verified_seconds=values[True],trusted_seconds=values[False],
        verified_median_seconds=a,trusted_median_seconds=b,speedup=a/b,saved_percent=100*(a-b)/a,
        verified_range_seconds=[min(values[True]),max(values[True])],
        trusted_range_seconds=[min(values[False]),max(values[False])],
        other_phase_medians={key:{str(flag):statistics.median(c[key] for c in cases
            if c['verify_dit_weights']==flag) for flag in (True,False)}
            for key in ('startup_seconds','controller_admission_seconds','aot_wall_seconds')},
        semantics='Parent selection + denoiser RPCs for two real blocks and eight hops; not full H3 latency',
        statistical_limit='Three fresh-service trials per arm; report dispersion, no significance claim from a point estimate')


def load_probe(bank_path,out):
    """Isolate first content checks from reloads using the same reader identity."""
    from patches.ptd.bank import PTDBank
    from patches.rank_local_int8 import state_hashes
    from dit_hash_observer import HashObserver
    rows=[];reference={}
    for verify in (True,False,False,True):
        observer=HashObserver()
        try:
            begin=time.perf_counter()
            bank=PTDBank(bank_path,allow_fixture=True,verify_payloads=verify)
            reader=bank.reader(0)
            admission=time.perf_counter()-begin
            for rank in (0,1):
                for repeat in range(3):
                    start=len(observer.rows);begin=time.perf_counter();cpu=time.thread_time()
                    module=reader.load_rank(0,rank)
                    row=dict(verify=verify,rank=rank,repeat=repeat,wall_seconds=time.perf_counter()-begin,
                        thread_cpu_seconds=time.thread_time()-cpu,admission_seconds=admission,
                        hash_calls=observer.rows[start:].copy())
                    # This comparison is deliberately outside the measured load.
                    proof=state_hashes(module);del module
                    if rank not in reference:reference[rank]=proof
                    assert reference[rank]==proof,'Reader A/B changed a weight tensor'
                    rows.append(row)
        finally:observer.close()
    (out/'reader_probe.json').write_text(json.dumps(dict(rows=rows,
        semantics='Sequential CPU owned-shard loads on the T4 host; first-use/reload, not GPU or physically cold I/O'),indent=2))
    return dict(trials=len(rows),weights_byte_exact=True,artifact='reader_probe.json')


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--cpu-preflight',action='store_true')
    args=p.parse_args();source=args.source.resolve();out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    sys.path[:0]=[str(source),str(source/'ComfyUI'),str(source/'benchmarks')];sys.argv=sys.argv[:1]
    import comfy.cli_args
    if args.cpu_preflight:comfy.cli_args.args.cpu=True
    from patches.legacy_runtime import validate_packages,validate_source
    validate_packages();validate_source(source/'ComfyUI')
    manifest=json.loads((source.parent/'h3_light_distribution/delivery_manifest.json').read_text())
    state=dict(status='starting',capsule_sha256=manifest['capsule_sha256'],full_pipeline=False,
        fixtures='Real PTD subset: two blocks, both refiners, heads and eight hops',cases=[],
        immutable_input_verification=True,bake_verification=True,default_verification_unchanged=True)
    def save():(out/'result.json').write_text(json.dumps(state,indent=2))
    try:
        if args.cpu_preflight:
            tests=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromNames([
                'benchmarks.test_dit_hash_ablation','benchmarks.test_ptd_overlay',
                'benchmarks.test_h3_dit_fusion','benchmarks.test_ptd_tp2','benchmarks.test_ptd_queue']))
            assert tests.wasSuccessful()
            import comfy_kitchen as ck
            assert callable(ck.prequantize_int8_attention) and callable(ck.int8_attention_from_prequantized)
            state.update(status='hash_ablation_cpu_ready',tests=tests.testsRun,no_weights_fetched=True,gpu_used=False)
            return
        import torch
        import importlib.metadata as md
        devices=[dict(name=torch.cuda.get_device_name(i),capability=list(torch.cuda.get_device_capability(i)),
            uuid=str(torch.cuda.get_device_properties(i).uuid)) for i in range(torch.cuda.device_count())]
        if len(devices)!=2 or len({d['uuid'] for d in devices})!=2 or any(d['capability']!=[7,5] or 'T4' not in d['name'] for d in devices):
            raise RuntimeError('Two actual distinct T4s required')
        state['hardware']=dict(devices=devices,torch=str(torch.__version__),cuda=torch.version.cuda,
            kitchen=md.version('comfy-kitchen'),tilelang=md.version('tilelang'))
        if torch.version.cuda!='12.8':raise RuntimeError('Keep the qualified native cu128 environment')
        os.environ.update(HF_HUB_DISABLE_XET='1',HF_HUB_DISABLE_PROGRESS_BARS='1',OMP_NUM_THREADS='2',
            MKL_NUM_THREADS='2',MALLOC_ARENA_MAX='2',H3_PTD_HASH_ABLATION='1')
        torch.set_num_threads(2);torch.set_grad_enabled(False)
        from tp2_public_dit_assets import stage
        from ptd_t4_assets import fetch_subset
        from patches.ptd.bank import bake,disk_plan
        from run_ptd_tp2_t4 import smoke,resource_plan
        scratch=Path('/tmp/h3_dit_hash_ablation_20261029');scratch.mkdir(exist_ok=False)
        state.update(status='staging_two_layer_base');save()
        base=stage(source,scratch/'base',layers=(0,1))
        resources=resource_plan(base)
        for device in (0,1):
            free,_=torch.cuda.mem_get_info(device)
            assert free>=resources['weight_bytes']+resources['reserve_bytes'],'T4 resource admission failed'
        state['disk_plan']=disk_plan(base,scratch/'bank',layers=2)
        assert state['disk_plan']['free_bytes']>=state['disk_plan']['required_free_bytes'],'Insufficient bank storage'
        state.update(status='staging_ptd_subset');save()
        checkpoint=fetch_subset(scratch/'adapter')
        state['subset_provenance']=json.loads((scratch/'adapter/provenance.json').read_text())
        state.update(status='baking_once');save();begin=time.monotonic()
        bank=bake(base,checkpoint,scratch/'bank',layers=2,allow_pruned_adaln=True)
        state['bake_seconds']=time.monotonic()-begin
        state['bank_manifest_sha256']=hashlib.sha256(bank.read_bytes()).hexdigest()
        state['reader_probe']=load_probe(bank,out);save()
        cpu_options=dict(future_cache=True,prefetch_refiner=True)
        # One unreported latency rehearsal fills the binary cache and exercises
        # the exact layout. Every measured trial then has fresh worker/readers.
        state.update(status='warmup');save();folder=out/'warmup';folder.mkdir()
        warmup=smoke(base,bank,folder,'queue',resources,cpu_options)
        reference=warmup['hop_output_sha256'],warmup['final_sha256']
        (folder/'case.json').write_text(json.dumps(warmup,indent=2))
        for name,verify in CASES:
            state.update(status=name);save();folder=out/name;folder.mkdir()
            begin=time.monotonic();case=smoke(base,bank,folder,'queue',resources,cpu_options,verify_dit_weights=verify)
            case.update(name=name,whole_case_seconds=time.monotonic()-begin)
            assert (case['hop_output_sha256'],case['final_sha256'])==reference,'Hash mode changed denoiser/trajectory bytes'
            for rank in case['restored_stats']:
                calls=rank['ptd_probe']['hash_verification']['calls']
                payload=[r for r in calls if r['kind'] in ('payload','tensor_state')]
                if verify:assert payload,'Verified arm failed to measure any payload hash work'
                else:assert not payload,'Trusted arm still hashed a DiT payload/state'
            (folder/'case.json').write_text(json.dumps(case,indent=2))
            state['cases'].append(case);save()
        state.update(status='complete',comparison=summary(state['cases']),outputs_byte_exact=True,
            interpretation='Two-layer operator/streaming fixture; no full-model/full-pipeline or statistical-significance claim')
    except BaseException as error:
        state.update(status='failed',error=repr(error));(out/'exception.txt').write_text(traceback.format_exc());raise
    finally:save()


if __name__=='__main__':main()
