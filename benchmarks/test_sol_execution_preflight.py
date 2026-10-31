"""CPU contracts for the physical Sol execution-preflight definition."""
from pathlib import Path
import unittest

from patches.attention_schedule import parse_attention_schedule,select_attention_policy
from benchmarks.dit_pair_contract import TOKENS,SEGMENTS

ROOT=Path(__file__).resolve().parents[1]


class SolExecutionPreflightContracts(unittest.TestCase):
    def test_two_call_schedule_is_kitchen_then_sol(self):
        schedule=parse_attention_schedule('2:sol@1.0')
        self.assertEqual([select_attention_policy(schedule,n,'kitchen',1.0) for n in (1,2)],
            [('kitchen',None),('sol',1.0)])

    def test_fixture_protects_only_conditioning_before_generated_audio(self):
        self.assertGreater(TOKENS,0)
        self.assertEqual(SEGMENTS[-2][2:]+SEGMENTS[-1][2:],('audio','video'))
        self.assertEqual(SEGMENTS[-2][0],3892)

    def test_runner_is_fail_closed_on_real_execution_seams(self):
        source=(ROOT/'benchmarks/sol_execution_preflight.py').read_text(encoding='utf-8')
        required=[
            "service.call('prepare_dit',TOKENS)",
            "kitchen['total']==50",
            "sol['total']==50",
            "not t['contiguous']",
            "srow['protected_prefix']==expected_prefix",
            "decoder.last_strategy=='native_temporal_pairs'",
            "x['q']['shape'][-1]==64",
            "aot.get('dit',{}).get('sol_precompiled')",
            "aot.get('video',{}).get('sol_precompiled')",
        ]
        for value in required:self.assertIn(value,source)


if __name__=='__main__':unittest.main(verbosity=2)
