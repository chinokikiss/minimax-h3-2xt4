"""CPU qualification for the reviewed runtime changes, before a T4 submission."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'ComfyUI'),str(ROOT/'benchmarks')]
os.environ['CUDA_VISIBLE_DEVICES']='-1'
import comfy.cli_args
comfy.cli_args.args.cpu=True

NAMES=['test_pipeline_review','test_dit_temporaries','test_tp2_bundle',
    'test_video_decode_memory','test_native_audio_decode','test_h3_default_fused_row',
    'test_h3_row_partial_combined','test_h3_reference_loading','test_h3_tensor_flow',
    'test_h3_operator_enhancements','test_residency_policy','test_legacy_runtime',
    'test_frozen_kaggle_setup','test_direct_pip_delivery','test_h3_loading_stages',
    'test_idle_lifecycle','test_idle_identity','test_review_optimizations','test_review_t4',
    'test_h3_configuration_defaults','test_te_loading','test_tp2_disk_bake','test_h3_l1_turbo_e2e',
    'test_temporal_vae','test_temporal_delivery','test_e2e_attention_validation',
    'test_h3_dit_fusion','test_h3_optimization_batch.Operators']


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();sys.argv=[sys.argv[0]]
    # Runtime-lock unit tests exercise the original Windows manifest as test
    # data. Actual native package imports/signatures are separately qualified
    # by prepare_kaggle_direct_pip before this subprocess is launched.
    os.environ['H3_RUNTIME_LOCK']=str(ROOT/'runtime/local_conda.json')
    start=time.monotonic()
    result=unittest.TextTestRunner(verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromNames(['benchmarks.'+n for n in NAMES]))
    receipt=dict(passed=result.wasSuccessful(),tests=result.testsRun,
        failures=len(result.failures),errors=len(result.errors),skipped=len(result.skipped),
        seconds=time.monotonic()-start,scope='CPU contracts, not physical T4 qualification')
    args.output.write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt),flush=True)
    raise SystemExit(not result.wasSuccessful())


if __name__=='__main__':main()
