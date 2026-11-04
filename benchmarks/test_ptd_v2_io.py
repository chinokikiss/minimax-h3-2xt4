"""CPU-only PTD v2 I/O concurrency and advisory cache-retirement contracts."""
from pathlib import Path
import threading
import time
import unittest
from unittest.mock import patch

import torch

import benchmarks.test_ptd_bank_v2 as fixture_module
import benchmarks.test_tp2_bundle as bundle_fixture
from patches.ptd.bank_v2 import PTDBankV2
from patches.comfy_tp_runtime import RuntimeConfig
from patches.windowed_tp import CheckpointReader


class PTDV2ReaderIO(unittest.TestCase):
    def setUp(self):
        fixture = fixture_module.SelfContainedV2('runTest')
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        self.fixture = fixture
        self.path = fixture.v2_manifest()
        self.bank = PTDBankV2(self.path, allow_fixture=True)
        self.template = lambda kind, rank: bundle_fixture.block({
            name: torch.zeros_like(value)
            for name, value in fixture.source.local[rank].items()})

    def _load(self, reader, rank=0):
        with patch('patches.ptd.bank_v2.shell', side_effect=self.template):
            return reader.load_rank(0, rank).state_dict()

    def test_one_pool_is_reused_across_hops_and_reconfigured_safely(self):
        from concurrent.futures import ThreadPoolExecutor
        self.bank.configure_io(4,False)
        first,second=self.bank.reader(2),self.bank.reader(3)
        created=[]
        def make_pool(*args,**kwargs):
            pool=ThreadPoolExecutor(*args,**kwargs)
            created.append(pool)
            return pool
        with patch('patches.ptd.bank_v2.ThreadPoolExecutor',side_effect=make_pool):
            self._load(first)
            self._load(second)
            self.assertEqual(len(created),1)
            self.bank.configure_io(2,False)
            self._load(first)
            self.assertEqual(len(created),2)
        self.bank.close_io()
        self.assertIsNone(self.bank._io_pool)
        self.assertTrue(all(pool._shutdown for pool in created))

    def test_overlay_cache_retires_once_after_all_auxiliary_reads(self):
        self.bank.configure_io(2,True)
        reader=self.bank.reader(3)
        names=[name for name in reader.overlay.header if name!='__metadata__']
        self.assertTrue(names)
        with patch('patches.ptd.bank_v2._retire_page_cache',return_value='hinted') as retire:
            for name in names[:-1]:reader.read_tensor(name)
            retire.assert_not_called()
            reader.read_tensor(names[-1])
            retire.assert_called_once_with(reader.overlay.path)
            reader.read_tensor(names[-1])
            self.assertEqual(retire.call_count,1)
        self.assertEqual(reader.stats['overlay_page_retire_hints'],1)

    def test_parallel_reads_are_tensor_identical_and_admitted(self):
        reader = self.bank.reader(3)
        reference = self._load(reader)
        self.assertEqual(self.bank.io_workers, 1)
        self.assertEqual(reader.workspace_bytes(0), 0)
        self.bank.configure_io(4, False)
        concurrent = self.bank.reader(3)
        self.assertEqual(concurrent.workspace_bytes(0), 3 * 8 * 2**20)
        visited = set()
        lock = threading.Lock()
        original = CheckpointReader.read_tensor

        def tracked(instance, name, *args, **kwargs):
            with lock:
                visited.add(threading.get_ident())
            time.sleep(0.002)
            return original(instance, name, *args, **kwargs)

        with patch.object(CheckpointReader, 'read_tensor', tracked):
            actual = self._load(concurrent)
        self.assertGreaterEqual(len(visited), 2)
        self.assertEqual(concurrent.stats['parallel_rank_loads'], 1)
        self.assertEqual(concurrent.stats['io_workers'], 4)
        for name, value in reference.items():
            self.assertTrue(torch.equal(actual[name], value), name)
        self.assertEqual(reader.identity, concurrent.identity)

    def test_parallel_reader_transient_bytes_are_in_weight_spec_admission(self):
        from patches.windowed_tp import tp_specs
        # This synthetic one-layer Native header omits norm1; isolate the
        # existing admission arithmetic from the full production shape plan.
        with patch('patches.rank_local_int8.rank_block_bytes', return_value=8192):
            serial = tp_specs(self.bank.reader(3), 'dit', 0, layers=1)[0]
            self.bank.configure_io(4, False)
            parallel = tp_specs(self.bank.reader(3), 'dit', 0, layers=1)[0]
        self.assertEqual(parallel.prepare_peak_bytes - serial.prepare_peak_bytes,
                         3 * 8 * 2**20)
        self.assertEqual(parallel.max_bytes, serial.max_bytes)

    def test_page_retirement_happens_only_after_rank_reads(self):
        self.bank.configure_io(2, True)
        reader = self.bank.reader(2)
        with patch('patches.ptd.bank_v2._retire_page_cache',
                   return_value='hinted') as retire:
            self._load(reader, 1)
            retire.assert_called_once()
            self.assertEqual(Path(retire.call_args.args[0]),
                             self.bank.shard_paths[2]['dit.block00.rank1'][0])
        self.assertEqual(reader.stats['page_retire_hints'], 1)
        self.assertEqual(reader.stats['parallel_rank_loads'], 1)
        self.assertEqual(reader.stats['page_retire_unavailable'], 0)

    def test_unavailable_page_hint_is_nonfatal(self):
        self.bank.configure_io(1, True)
        reader = self.bank.reader(0)
        with patch('patches.ptd.bank_v2._retire_page_cache', return_value='unavailable'):
            self._load(reader)
        self.assertEqual(reader.stats['page_retire_unavailable'], 1)
        self.assertEqual(reader.stats['page_retire_hints'], 0)

    def test_options_are_v2_only_and_strictly_validated(self):
        for workers in (0, -1, 9, 2.5, True):
            with self.subTest(workers=workers), self.assertRaisesRegex(ValueError, '1..8'):
                self.bank.configure_io(workers)
        with self.assertRaisesRegex(ValueError, 'boolean'):
            self.bank.configure_io(4, 1)
        normal = RuntimeConfig('te', 'dit')
        self.assertEqual(normal.ptd_io_workers, 1)
        self.assertFalse(normal.ptd_retire_page_cache)
        with self.assertRaisesRegex(ValueError, '1..8'):
            RuntimeConfig('te', 'dit', ptd_io_workers=9)
        with self.assertRaisesRegex(ValueError, 'boolean'):
            RuntimeConfig('te', 'dit', ptd_retire_page_cache=1)


if __name__ == '__main__':
    unittest.main(verbosity=2)
