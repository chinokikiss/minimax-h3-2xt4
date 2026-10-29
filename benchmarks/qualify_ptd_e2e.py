"""CPU-only legacy API, Native sampler integration and reference-file gate."""
import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
import traceback
import unittest


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--storage-admission',action='store_true'); args=parser.parse_args()
    source=args.source.resolve(); args.output.mkdir(parents=True,exist_ok=True)
    sys.path[:0]=[str(source),str(source/'ComfyUI'),str(source/'benchmarks')]
    os.environ['CUDA_VISIBLE_DEVICES']='-1'; sys.argv=sys.argv[:1]
    import comfy.cli_args
    comfy.cli_args.args.cpu=True
    from patches.legacy_runtime import validate_packages,validate_source
    validate_packages(); validate_source(source/'ComfyUI')
    manifest=json.loads((source.parent/'h3_light_distribution/delivery_manifest.json').read_text())
    state=dict(status='starting',capsule_sha256=manifest['capsule_sha256'],no_weights_fetched=True,
        full_pipeline_authorized=True,gpu_used=False,auxiliary_layout='monolithic',credentials=False)
    try:
        from benchmarks.ptd_e2e_contract import validate,resource_plan
        from benchmarks.qualify_anime_inputs import download_image
        plan=json.loads((source/'ptd_e2e_plan.json').read_text())
        state['workload']=validate(plan)
        from patches.ptd.checkpoint import RELEASE
        if RELEASE!=plan['ptd_release']: raise ValueError('Plan differs from pinned PTD release')
        assets=json.loads((source/'bundle_assets.json').read_text())
        assert assets['publication']['repo']=='cnmds/minimax-h3-tp2-int8-561a6211'
        assert assets['publication']['commit']=='d3e6a4b131056d2c626a01acf6ba48123a885783'
        vaes=json.loads((source/'l1_model_identity.json').read_text())
        state['resource_estimate']=resource_plan(assets['auxiliary']['dit']['native_header'],assets,vaes,
                                                free_bytes=2**40,host_limit=30*2**30)
        state['references']=[download_image(row,source/'bench_assets/l1_current') for row in plan['reference_images']]
        result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromNames([
            'benchmarks.test_ptd_tp2','benchmarks.test_ptd_queue','benchmarks.test_ptd_cpu_schedule',
            'benchmarks.test_ptd_e2e','benchmarks.test_e2e_attention_validation',
            'benchmarks.test_ptd_overlay','benchmarks.test_ptd_storage']))
        state['contract_tests']=result.testsRun
        if not result.wasSuccessful(): raise RuntimeError('PTD E2E CPU contracts failed')
        if args.storage_admission:
            from benchmarks.ptd_storage import capacity_probe
            with tempfile.TemporaryDirectory(prefix='h3_ptd_storage_',dir='/tmp') as scratch:
                state['storage_admission']=capacity_probe(scratch,state['resource_estimate']['required_free_bytes'],
                                                          args.output/'storage_admission.json')
        state.update(status='ptd_e2e_cpu_ready',native_sampler_integration=True)
    except BaseException as error:
        state.update(status='failed',error=repr(error)); (args.output/'exception.txt').write_text(traceback.format_exc()); raise
    finally:
        (args.output/'result.json').write_text(json.dumps(state,indent=2)); print(json.dumps(state),flush=True)


if __name__=='__main__': main()
