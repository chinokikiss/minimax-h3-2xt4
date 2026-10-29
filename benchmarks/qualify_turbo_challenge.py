"""CPU source/core, actual headers and exact reference gate; no model downloads."""
import argparse
import json
import os
from pathlib import Path
import sys
import traceback
import unittest


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    source=args.source.resolve();out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    sys.path[:0]=[str(source),str(source/'ComfyUI'),str(source/'benchmarks')]
    os.environ['CUDA_VISIBLE_DEVICES']='-1';sys.argv=sys.argv[:1]
    import comfy.cli_args
    comfy.cli_args.args.cpu=True
    from patches.legacy_runtime import validate_packages,validate_source
    validate_packages();validate_source(source/'ComfyUI')
    manifest=json.loads((source.parent/'h3_light_distribution/delivery_manifest.json').read_text())
    state=dict(status='starting',capsule_sha256=manifest['capsule_sha256'],no_weights_fetched=True,
        gpu_used=False,ptd=False,online_lora=False,full_pipeline_authorized=True)
    try:
        from benchmarks.turbo_challenge_contract import validate
        from benchmarks.turbo_e2e_contract import validate_header
        from benchmarks.qualify_anime_inputs import download_image
        plan=json.loads((source/'turbo_challenge_plan.json').read_text())
        state['workload']=validate(plan)
        assets=json.loads((source/'bundle_assets.json').read_text())
        assert assets['publication']['repo']=='cnmds/minimax-h3-tp2-int8-561a6211'
        assert assets['publication']['commit']=='d3e6a4b131056d2c626a01acf6ba48123a885783'
        header=json.loads((source/'turbo_lora_header.json').read_text())
        assert len(validate_header(header,assets['auxiliary']['dit']['native_header']))==208
        state['references']=[download_image(r,source/'bench_assets/l1_current') for r in plan['reference_images']]
        tests=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromNames([
            'benchmarks.test_turbo_challenge','benchmarks.test_anime_e2e','benchmarks.test_e2e_attention_validation']))
        assert tests.wasSuccessful()
        state.update(status='turbo_challenge_cpu_ready',contract_tests=tests.testsRun,lora_targets=208,
            native_frame_grid=True,trusted_bake_default=True)
    except BaseException as e:
        state.update(status='failed',error=repr(e));(out/'exception.txt').write_text(traceback.format_exc());raise
    finally:
        (out/'result.json').write_text(json.dumps(state,indent=2));print(json.dumps(state),flush=True)


if __name__=='__main__':main()
