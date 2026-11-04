"""CPU-only IOAB regression: opt-in GPU admission, bounded prefetch, proof reuse."""
from concurrent.futures import Future, ThreadPoolExecutor
import ast
import base64
import json
import re
import os
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from benchmarks.test_ptd_v2_bake import BoundedParts, EmptySyntheticPTD
from patches.ptd.bake_v2 import _devices_ok, bake_hop_part
from patches.ptd.publish_v2 import verify_part, admit_baked_part, prepare_private_kaggle_part
from patches.tp2_bundle import sha


class IndependentHopIoAB(unittest.TestCase):
    def setUp(self):
        fixture = BoundedParts('runTest')
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        self.fixture = fixture
        self.root = fixture.root

    def test_single_t4_opt_in_is_not_production_default(self):
        with self.assertRaisesRegex(ValueError, 'two separate T4'):
            _devices_ok(('cuda:0',), False)
        with patch('torch.cuda.get_device_capability', return_value=(7, 5)), \
             patch('torch.cuda.get_device_name', return_value='Tesla T4'):
            _devices_ok(('cuda:0',), False, independent_hop_gpu=True)
        with patch('torch.cuda.get_device_capability', return_value=(8, 6)), \
             patch('torch.cuda.get_device_name', return_value='RTX'):
            with self.assertRaisesRegex(RuntimeError, 'T4'):
                _devices_ok(('cuda:0',), False, independent_hop_gpu=True)

    def test_async_prefetch_preserves_complete_rank_and_file_hashes(self):
        source = self.fixture.fixture.source.descriptor
        for hop, prefetch in ((1, False), (2, True)):
            dest = self.root / f'pipeline_hop_{hop}'
            with patch('patches.ptd.bake_v2.PTDCheckpoint', EmptySyntheticPTD):
                receipt = bake_hop_part(
                    source, self.fixture.heads, hop, dest,
                    devices=('cpu',), layers=1, allow_fixture=True,
                    prefetch_layers=prefetch)
            data = json.loads(receipt.read_text(encoding='utf-8'))
            self.assertEqual(set(data['shards']), {
                'dit.block00.rank0', 'dit.block00.rank1'})
            proof = verify_part(dest, f'hop_{hop:02d}',
                                expected_receipt_sha256=sha(receipt),
                                capture_source_snapshot=True)
            self.assertEqual(proof['payload_files'], 6)
            self.assertEqual(set(proof['source_snapshot']),
                             {p.name for p in dest.iterdir()})

    def test_verified_staging_does_not_hash_the_part_again(self):
        with patch('patches.ptd.bake_v2.PTDCheckpoint', EmptySyntheticPTD):
            receipt = bake_hop_part(
                self.fixture.fixture.source.descriptor, self.fixture.heads,
                1, self.root / 'hop_01', devices=('cpu',),
                layers=1, allow_fixture=True, prefetch_layers=True)
        part = receipt.parent
        proof = verify_part(part, 'hop_01', capture_source_snapshot=True)
        admission = admit_baked_part(part, 'hop_01')
        self.assertEqual(admission['payload_tree_sha256'], proof['payload_tree_sha256'])
        self.assertEqual(admission['payload_bytes'], proof['payload_bytes'])
        with patch('patches.ptd.publish_v2.verify_part',
                   side_effect=AssertionError('duplicate full SHA256')):
            result = prepare_private_kaggle_part(
                part, 'hop_01', self.root / 'stage_01',
                dataset_id='pamcks/test-hop-01-ioab', license_name='other',
                verified_proof=proof)
        self.assertEqual(result['payload_bytes'], proof['payload_bytes'])
        self.assertEqual(set(p.name for p in (self.root/'stage_01'/'payload').iterdir()),
                         set(proof['source_snapshot']))
        # A forged/restored mtime is insufficient; inode ctime still changes.
        shard = part / 'dit.block00.rank0.safetensors'
        before = shard.stat()
        with shard.open('r+b') as stream:
            stream.seek(-1, os.SEEK_END)
            b = stream.read(1)
            stream.seek(-1, os.SEEK_END)
            stream.write(bytes([b[0] ^ 1]))
        os.utime(shard, ns=(before.st_atime_ns, before.st_mtime_ns))
        with self.assertRaisesRegex(ValueError, 'changed'):
            prepare_private_kaggle_part(
                part, 'hop_01', self.root / 'stage_02',
                dataset_id='pamcks/test-hop-02-ioab', license_name='other',
                verified_proof=proof)


    def test_baker_receipt_admission_avoids_rehashing_large_payloads(self):
        with patch('patches.ptd.bake_v2.PTDCheckpoint', EmptySyntheticPTD):
            receipt = bake_hop_part(
                self.fixture.fixture.source.descriptor, self.fixture.heads,
                1, self.root / 'fast_hop_01', devices=('cpu',),
                layers=1, allow_fixture=True, prefetch_layers=True)
        from patches import ptd
        from patches.ptd import publish_v2
        original_sha = publish_v2.sha
        hashed = []

        def allow_small_files(path):
            if str(path).endswith('.safetensors'):
                raise AssertionError('Baker payload must not be SHA256-rehashed')
            hashed.append(str(path))
            return original_sha(path)

        with patch.object(publish_v2, 'sha', side_effect=allow_small_files), \
             patch('patches.ptd.publish_v2.verify_part',
                   side_effect=AssertionError('full local verifier called')):
            admitted = admit_baked_part(receipt.parent, 'hop_01')
            self.assertEqual(admitted['status'], 'bake_receipt_admitted')
            self.assertFalse(admitted['independent_local_payload_sha256'])
            staged = prepare_private_kaggle_part(
                receipt.parent, 'hop_01', self.root / 'fast_staging',
                dataset_id='pamcks/test-fast-receipt', license_name='other',
                verified_proof=admitted)
        self.assertEqual(staged['payload_bytes'], admitted['payload_bytes'])
        self.assertTrue(hashed)
        self.assertEqual(len(list((self.root/'fast_staging'/'payload').iterdir())),
                         admitted['payload_files'] + 1)
        shard = receipt.parent / 'dit.block00.rank0.safetensors'
        with shard.open('ab') as stream:
            stream.write(b'corrupt')
        with self.assertRaisesRegex(ValueError, 'changed'):
            prepare_private_kaggle_part(
                receipt.parent, 'hop_01', self.root / 'fast_staging_tamper',
                dataset_id='pamcks/test-fast-tamper', license_name='other',
                verified_proof=admitted)

    def test_serial_dataset_upload_ack_gate_preserves_parallel_staging(self):
        from benchmarks.run_ptd_v2_stage_c_dual_hop_t4 import _publish_one, UploadSkipped

        def exercise(first_succeeds, replace_existing=False):
            root = self.root / ('serial_replace' if replace_existing else
                                ('serial_ok' if first_succeeds else 'serial_fail'))
            parts = root / 'parts'
            scratch = root / 'scratch'
            output = root / 'output'
            for name in ('hop_01', 'hop_02'):
                (parts / name).mkdir(parents=True)
                (parts / name / 'receipt.json').write_text('{}', encoding='utf-8')
            scratch.mkdir()
            output.mkdir()
            license_path = root / 'LICENSE'
            license_path.write_text('fixture license', encoding='utf-8')
            started_first = threading.Event()
            staged_second = threading.Event()
            release_first = threading.Event()
            requests = []
            modes = []

            def prepare(part_root, part, staging, **kwargs):
                staging.mkdir(parents=True)
                if part == 'hop_02':
                    staged_second.set()
                return {'source_receipt_sha256': sha(part_root / 'receipt.json'),
                        'payload_bytes': 1}

            class FakeApi:
                def authenticate(self):
                    pass

                def dataset_create_version(self, staging, **kwargs):
                    self_inner = self
                    assert kwargs['delete_old_versions'] is True
                    modes.append('replace')
                    return self_inner.dataset_create_new(staging, **kwargs)

                def dataset_create_new(self, staging, **kwargs):
                    name = Path(staging).name
                    requests.append(name)
                    if name == 'hop_01':
                        started_first.set()
                        if not release_first.wait(5):
                            raise TimeoutError('fixture hung upload')
                        if not first_succeeds:
                            raise ConnectionError('fixture provider error')
                    return SimpleNamespace(status='ok', error=None)

            def verify(part_root, part, **kwargs):
                return {'source_snapshot': {}, 'payload_bytes': 1,
                        'receipt_sha256': sha(part_root / 'receipt.json')}

            def stage(name, fn):
                return fn()

            with patch('kaggle.api.kaggle_api_extended.KaggleApi', FakeApi), \
                 patch('patches.ptd.publish_v2.admit_baked_part', side_effect=verify), \
                 patch('patches.ptd.publish_v2.prepare_private_kaggle_part', side_effect=prepare), \
                 ThreadPoolExecutor(max_workers=2) as pool:
                gate = Future()
                first = pool.submit(_publish_one, 1, parts, scratch, license_path,
                                    output, stage, replace_existing=replace_existing)
                def open_gate(done):
                    try:
                        done.result()
                    except BaseException:
                        gate.set_result(False)
                    else:
                        gate.set_result(True)
                first.add_done_callback(open_gate)
                second = pool.submit(_publish_one, 2, parts, scratch, license_path,
                                     output, stage, previous_upload=gate,
                                     replace_existing=replace_existing)
                self.assertTrue(started_first.wait(5))
                self.assertTrue(staged_second.wait(5))
                self.assertEqual(requests, ['hop_01'])
                release_first.set()
                if first_succeeds:
                    self.assertEqual(first.result(timeout=5)['dataset_id'].split('/')[-1],
                                     'h3-ptd-v2-hop-01-ioab-20261103')
                    self.assertEqual(second.result(timeout=5)['dataset_id'].split('/')[-1],
                                     'h3-ptd-v2-hop-02-ioab-20261103')
                    self.assertEqual(requests, ['hop_01', 'hop_02'])
                    self.assertEqual(modes, ['replace', 'replace'] if replace_existing else [])
                else:
                    with self.assertRaisesRegex(RuntimeError, 'ConnectionError'):
                        first.result(timeout=5)
                    with self.assertRaises(UploadSkipped):
                        second.result(timeout=5)
                    self.assertEqual(requests, ['hop_01'])
        exercise(True)
        exercise(False)
        exercise(True, replace_existing=True)

    def test_builder_has_isolated_private_dataset_names(self):
        from benchmarks.run_ptd_v2_stage_c_dual_hop_t4 import dataset_id
        self.assertEqual(dataset_id(1), 'pamcks/h3-ptd-v2-hop-01-ioab-20261103')
        self.assertEqual(dataset_id(2), 'pamcks/h3-ptd-v2-hop-02-ioab-20261103')
        with self.assertRaises(ValueError):
            dataset_id(0)


    def test_dual_hop_builder_embeds_explicit_version_replacement(self):
        from benchmarks.build_ptd_v2_stage_c_dual_hop_job import build, DEFAULT_BASE
        output = self.root / 'dual_hop_replace_builder'
        binding = build(DEFAULT_BASE, output, replace_existing=True)
        self.assertEqual(binding['publish_mode'], 'replace_delete_old_versions')
        meta = json.loads((output / 'kernel-metadata.json').read_text(encoding='utf-8'))
        self.assertEqual(meta['dataset_sources'], [])
        self.assertTrue(meta['is_private'])
        code = ''.join(json.loads((output / 'experiment.ipynb').read_text())['cells'][0]['source'])
        ast.parse(code)
        worker_matches = re.findall(r"worker.write_bytes\(base64.b64decode\('([^']+)'\)\)", code)
        self.assertTrue(worker_matches)
        worker = base64.b64decode(worker_matches[-1]).decode('utf-8')
        self.assertIn("argv.append('--replace-existing')", worker)

    def test_download_verified_checkpoint_is_not_read_twice(self):
        from benchmarks import preflight_ptd_v2_bake as preflight
        from benchmarks.test_ptd_v2_preflight import V2BakerPreflight
        case = V2BakerPreflight('runTest')
        case.setUp()
        self.addCleanup(case.doCleanups)
        st = case.checkpoint.stat()
        seal = dict(sha256=case.fake_release['sha256'],
                    verifier='download_on_t4',
                    fingerprint=(st.st_dev, st.st_ino, st.st_size,
                                 st.st_mtime_ns, st.st_ctime_ns))
        original = preflight._file_hash

        def hash_except_checkpoint(path):
            if Path(path).resolve() == case.checkpoint.resolve():
                raise AssertionError('checkpoint must not be rehashed after downloader SHA256')
            return original(path)

        with patch.object(preflight, '_file_hash', side_effect=hash_except_checkpoint):
            result = case._inspect(verify_ptd_sha256=True,
                                   preverified_checkpoint=seal)
        self.assertTrue(result['release_sha256_checked'])
        self.assertEqual(result['checkpoint_digest_source'], 'download_on_t4_same_inode')
        self.assertEqual(result['issues'], [])
        seal['fingerprint'] = (0, 0, 0, 0, 0)
        blocked = case._inspect(verify_ptd_sha256=True, preverified_checkpoint=seal)
        self.assertFalse(blocked['release_sha256_checked'])
        self.assertIn('Preverified PTD release source identity is invalid',
                      blocked['issues'])

    def test_comparator_requires_both_completed_hops(self):
        from benchmarks.compare_ptd_v2_stage_c_ioab import compare
        baseline = {
            'status': 'upload_requests_accepted_remote_readback_pending',
            'stages': {name: {'status': 'complete', 'seconds': 100}
                       for name in ('license_source', 'private_create_auth',
                                    'stage_full_base', 'download_pinned_ptd',
                                    'preflight_full_50_layer', 'bake_hop_00',
                                    'verify_local_hop_00', 'upload_hop_00')}}
        trial = {
            'status': 'upload_requests_accepted_remote_readback_pending',
            'hops': [1, 2],
            'started_utc': '2026-11-03T00:00:00+00:00',
            'finished_utc': '2026-11-03T00:20:00+00:00',
            'stages': {f'{step}_hop_{hop:02d}': {'status': 'complete',
                                                'seconds': 200}
                       for hop in (1, 2)
                       for step in ('bake', 'verify_local', 'upload')},
            'published': {f'hop_{hop:02d}': {
                'status': 'create_request_accepted_processing_unverified'}
                          for hop in (1, 2)}}
        baseline_path = self.root / 'baseline_trial.json'
        trial_path = self.root / 'ioab_trial.json'
        baseline_path.write_text(json.dumps(baseline), encoding='utf-8')
        trial_path.write_text(json.dumps(trial), encoding='utf-8')
        summary = compare(baseline_path, trial_path)
        self.assertEqual(summary['extrapolated_speedup_vs_hop00_sequential'], 4 / 3)
        self.assertFalse(summary['remote_provider_version_and_sha256_verified'])
        trial['published'].pop('hop_02')
        trial_path.write_text(json.dumps(trial), encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'both'):
            compare(baseline_path, trial_path)


if __name__ == '__main__':
    unittest.main()
