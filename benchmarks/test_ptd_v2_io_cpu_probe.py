"""Bounded CPU PTD v2 I/O qualification harness regression tests."""
import tempfile
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import torch

import benchmarks.test_ptd_bank_v2 as fixture_module
import benchmarks.test_tp2_bundle as bundle_fixture
from benchmarks.run_ptd_v2_io_cpu_probe import (FORMAT, probe_bank,
                                                STANDARD_PROFILES, _summary, main)
from patches.ptd.bank_v2 import PTDBankV2


class MountedIOProbe(unittest.TestCase):
    def setUp(self):
        fixture = fixture_module.SelfContainedV2('runTest')
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        self.fixture = fixture
        self.bank = PTDBankV2(fixture.v2_manifest(), allow_fixture=True)
        self.template = lambda kind, rank: bundle_fixture.block({
            name: torch.zeros_like(value)
            for name, value in fixture.source.local[rank].items()})

    def test_bounded_io_profiles_are_exact_and_measured(self):
        profiles = ((1, False), (4, False), (4, True))
        checkpoints = []
        with patch('patches.ptd.bank_v2.shell', side_effect=self.template):
            result = probe_bank(self.bank, hop=2, layers=(0,), ranks=(0, 1),
                                profiles=profiles, repeats=1, max_seconds=60.,
                                on_progress=lambda value: checkpoints.append(
                                    (value['status'], len(value['cases']))))
        self.assertEqual(result['status'], 'passed')
        self.assertEqual(result['format'], FORMAT)
        self.assertTrue(result['cpu_only'])
        self.assertFalse(result['physical_t4_qualified'])
        self.assertFalse(result['ranks_simultaneous'])
        self.assertEqual(len(result['manifest_sha256']), 64)
        self.assertEqual(len(result['cases']), 6)
        self.assertEqual(checkpoints[0], ('running', 0))
        self.assertEqual(checkpoints[-1], ('passed', 6))
        self.assertEqual(len(result['summary']), 3)
        for case in result['cases']:
            self.assertEqual(case['status'], 'bit_exact')
            self.assertEqual(len(case['state_digest_sha256']), 64)
            self.assertGreater(case['read_ms'], 0)
            self.assertGreaterEqual(case['load_ms'], case['read_ms'])
            self.assertGreater(case['source_read_bytes'], 0)
            self.assertGreater(case['read_calls'], 0)
            self.assertGreater(case['memory']['rss_peak_bytes'], 0)
        self.assertEqual(result['summary']['4-on']['runs'], 2)
        self.assertIn('median_io_speedup_vs_1_off', result['summary']['4-on'])

    def test_mismatched_tensor_state_fails_and_retains_partial_report(self):
        states = []
        with patch('patches.ptd.bank_v2.shell', side_effect=self.template), \
                patch('benchmarks.run_ptd_v2_io_cpu_probe.state_hashes',
                      return_value={'bad': {'sha256': 'wrong'}}):
            with self.assertRaisesRegex(ValueError, 'bitwise state mismatch'):
                probe_bank(self.bank, hop=0, layers=(0,), ranks=(0,),
                           profiles=((1, False),), max_seconds=30.,
                           on_progress=lambda value: states.append(
                               (value['status'], value.get('error_type'))))
        self.assertEqual(states[0][0], 'running')
        self.assertEqual(states[-1], ('failed', 'ValueError'))

    def test_invalid_sweep_is_rejected_before_allocation(self):
        invalid = (
            dict(layers=(0, 0)),
            dict(layers=(0, 1)),
            dict(ranks=(0, 0)),
            dict(profiles=((1, False), (1, False))),
            dict(repeats=4),
            dict(max_seconds=float('inf')),
        )
        for option in invalid:
            kwargs = dict(hop=0, layers=(0,), ranks=(0,),
                          profiles=((1, False),), max_seconds=20.)
            kwargs.update(option)
            with self.subTest(option=option), self.assertRaises(ValueError):
                probe_bank(self.bank, **kwargs)

    def test_cli_accepts_explicit_mount_map_and_preserves_previous_output(self):
        with tempfile.TemporaryDirectory() as scratch:
            folder = Path(scratch)
            mounts = folder / 'mounts.json'
            mounts.write_text(json.dumps({'shared': 'C:/dataset/shared'}), encoding='utf-8')
            output = folder / 'probe.json'

            def fake_probe(bank, **kwargs):
                self.assertEqual(kwargs['layers'], (0,))
                self.assertTrue(kwargs['on_progress'])
                kwargs['on_progress'](dict(status='passed', summary={}))
                return dict(status='passed', summary={})

            args = ['--manifest', str(self.bank.path), '--part-roots-file', str(mounts),
                    '--hop', '1', '--layers', '0', '--fixture', '--output', str(output)]
            with patch('benchmarks.run_ptd_v2_io_cpu_probe.PTDBankV2') as constructor, \
                    patch('benchmarks.run_ptd_v2_io_cpu_probe.probe_bank',
                          side_effect=fake_probe):
                self.assertEqual(main(args), 0)
                self.assertEqual(constructor.call_args.kwargs['part_roots'],
                                 (('shared', 'C:/dataset/shared'),))
                self.assertEqual(json.loads(output.read_text())['status'], 'passed')
                with self.assertRaises(FileExistsError):
                    main(args)

    def test_summary_handles_no_baseline_without_fabricating_speedup(self):
        sample = dict(profile='4-on', read_ms=10., load_ms=12.,
                      source_read_bytes=2**20, memory=dict(rss_peak_bytes=5),
                      page_retire_hints=1, page_retire_errors=0,
                      page_retire_unavailable=0)
        row = _summary([sample])['4-on']
        self.assertNotIn('median_io_speedup_vs_1_off', row)
        self.assertAlmostEqual(row['aggregate_read_mib_per_sec'], 100.)


if __name__ == '__main__':
    unittest.main(verbosity=2)
