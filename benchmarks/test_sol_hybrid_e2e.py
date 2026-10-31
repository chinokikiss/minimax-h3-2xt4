"""CPU contract for the Sol-hybrid 0.85 MP long-sequence E2E policy."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import unittest

from benchmarks.run_h3_l1_turbo_e2e import runtime_config
from patches.attention_schedule import select_attention_policy
from patches.h3_operator_policy import protected_prefix

BASELINE=Path(r'D:/attention_eval/h3_turbo_085mp_10s_20261029/retry2/kaggle/gpu_final/turbo_085mp_e2e/retry_request.json')
BASELINE_RESULT=Path(r'D:/attention_eval/h3_turbo_085mp_10s_20261029/retry2/kaggle/gpu_final/turbo_085mp_e2e/generation/result.json')


class SolHybridE2E(unittest.TestCase):
    def test_exact_baseline_workload_with_attention_only_override(self):
        if not BASELINE.is_file():
            self.skipTest('Local completed baseline request is unavailable')
        old=json.loads(BASELINE.read_text(encoding='utf-8'))
        plan=copy.deepcopy(old)
        plan['paths']={'te':'te','dit':'dit'}
        plan['operators']=dict(
            te_attention='kitchen_causal',
            dit_attention='kitchen',
            dit_attention_schedule='2-7:sol@1.0',
            sol_tau=1.0,
            video_attention='kitchen',
        )
        runtime=runtime_config(plan)
        self.assertEqual(runtime.operators.te_attention,'kitchen_causal')
        self.assertEqual(runtime.operators.dit_attention,'kitchen')
        self.assertEqual(runtime.video_attention,'kitchen')
        self.assertEqual(runtime.sol_tau,1.0)
        selected=[select_attention_policy(runtime.dit_attention_schedule,n,'kitchen',runtime.sol_tau)
                  for n in range(1,9)]
        self.assertEqual(selected,[
            ('kitchen',None),
            ('sol',1.0),('sol',1.0),('sol',1.0),('sol',1.0),('sol',1.0),('sol',1.0),
            ('kitchen',None),
        ])
        for key in ('width','height','frames','fps','steps','sampler','scheduler','cfg',
                    'video_shift','audio_shift','seed','prompt_sha256','turbo_lora','reference_images'):
            self.assertEqual(plan[key],old[key],key)

    def test_real_long_sequence_layout_protects_only_conditioning(self):
        if not BASELINE_RESULT.is_file():
            self.skipTest('Completed long-sequence baseline result is unavailable')
        result=json.loads(BASELINE_RESULT.read_text(encoding='utf-8'))
        layout=result['ranks'][0]['benchmark']['layouts'][0]
        self.assertEqual(layout['sequence_tokens'],65166)
        self.assertEqual(layout['segments'],[
            [0,2528,'text'],[2528,3344,'ref_img'],[3344,4164,'ref_img'],
            [4164,4974,'audio'],[4974,65166,'video']])
        native=SimpleNamespace(seq_len=layout['sequence_tokens'],segments=[tuple(x) for x in layout['segments']])
        self.assertEqual(protected_prefix({'minimax_h3_layout':native},native.seq_len),4164)


if __name__=='__main__':
    unittest.main(verbosity=2)
