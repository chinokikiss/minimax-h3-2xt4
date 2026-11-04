"""CPU-only Stage C Kaggle delivery and safety contracts."""
import ast
import base64
import hashlib
import json
import re
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from benchmarks import build_ptd_v2_stage_c_job as gpu
from benchmarks import build_ptd_v2_c0_tar_probe as c0
from benchmarks import build_ptd_v2_c0_readback as readback
from benchmarks import run_ptd_v2_stage_c_publish_t4 as runner
from benchmarks.verify_ptd_v2_delivery import qualify

ROOT=Path(__file__).resolve().parents[1]


class StageCDelivery(unittest.TestCase):
    def test_part_identities_and_private_one_create_contract(self):
        self.assertEqual(runner.dataset_id('shared'),
                         'pamcks/h3-ptd-v2-shared-20261103')
        self.assertEqual(runner.dataset_id('hop_07'),
                         'pamcks/h3-ptd-v2-hop-07-20261103')
        with self.assertRaises(ValueError):
            runner.dataset_id('extra')
        license_path=ROOT/runner.LICENSE_RELATIVE
        self.assertEqual(hashlib.sha256(license_path.read_bytes()).hexdigest(),
                         runner.LICENSE_SHA256)
        source=(ROOT/'benchmarks/run_ptd_v2_stage_c_publish_t4.py').read_text(encoding='utf-8')
        self.assertIn('public=False',source)
        self.assertIn("dir_mode='tar'",source)
        self.assertIn("status='upload_requests_accepted_remote_readback_pending'",source)
        self.assertIn('dataset_create_version(',source)
        self.assertIn('delete_old_versions=True',source)
        self.assertNotIn('dataset_list(',source)
        self.assertNotIn('kernel_status(',source)

    def test_explicit_version_replacement_is_opt_in(self):
        with tempfile.TemporaryDirectory() as folder:
            parent = Path(folder)
            root, scratch, output = parent / 'part', parent / 'scratch', parent / 'out'
            for directory in (root, scratch, output):
                directory.mkdir()
            (root / 'receipt.json').write_text('{}', encoding='utf-8')
            license_path = parent / 'license.txt'
            license_path.write_text('fixture', encoding='utf-8')
            calls = []

            def prepare(part_root, part, staging, **kwargs):
                staging.mkdir(parents=True, exist_ok=True)
                return dict(source_receipt_sha256=hashlib.sha256(
                    (part_root / 'receipt.json').read_bytes()).hexdigest(), payload_bytes=1)

            class FakeApi:
                def authenticate(self):
                    return None
                def dataset_create_new(self, path, **kwargs):
                    calls.append(('create', kwargs))
                    return SimpleNamespace(status='ok', error=None)
                def dataset_create_version(self, path, **kwargs):
                    calls.append(('replace', kwargs))
                    return SimpleNamespace(status='ok', error=None)

            with patch('kaggle.api.kaggle_api_extended.KaggleApi', FakeApi), \
                 patch('patches.ptd.publish_v2.prepare_private_kaggle_part', side_effect=prepare):
                for replace in (False, True):
                    state = dict(published={})
                    result = runner._upload_one('hop_01', root, scratch,
                                                license_path, state, output,
                                                replace_existing=replace)
                    expected = 'replace' if replace else 'create'
                    self.assertEqual(calls[-1][0], expected)
                    self.assertEqual(result['operation'],
                                     'replace_delete_old_versions' if replace else 'create_new')
                    self.assertEqual(result['remote_readback'], 'pending')
            self.assertTrue(calls[0][1]['public'] is False)
            self.assertNotIn('delete_old_versions', calls[0][1])
            self.assertTrue(calls[1][1]['delete_old_versions'])
            self.assertNotIn('public', calls[1][1])
            self.assertEqual(calls[1][1]['dir_mode'], 'tar')

    def test_small_probe_and_readback_are_private_and_synthetic_only(self):
        with tempfile.TemporaryDirectory() as temp:
            left=Path(temp)/'c0'
            right=Path(temp)/'readback'
            c0.build(left)
            readback.build(right)
            for folder in (left,right):
                meta=json.loads((folder/'kernel-metadata.json').read_text())
                self.assertTrue(meta['is_private'])
                self.assertFalse(meta['enable_gpu'])
                script=''.join(json.loads((folder/'experiment.ipynb').read_text())['cells'][0]['source'])
                ast.parse(script)
            self.assertEqual(json.loads((right/'kernel-metadata.json').read_text())['dataset_sources'],
                             [readback.DATASET])
            self.assertIn("status='passed_direct_payload_mount'",readback.WORKER)
            self.assertIn("nested_root/'pamcks'/'h3-ptd-v2-c0-tar-probe-20261103'",readback.WORKER)
            self.assertIn("state['nested_owner_datasets']",readback.WORKER)

    def test_full_hop0_delivery_closure_and_no_unqualified_pass(self):
        with tempfile.TemporaryDirectory() as temp:
            output=Path(temp)/'stage-c'
            binding=gpu.build(gpu.DEFAULT_BASE,output,hop=0)
            second=gpu.build(gpu.DEFAULT_BASE,Path(temp)/'stage-c-repeat',hop=0)
            replaced_dir=Path(temp)/'stage-c-replace'
            replacement=gpu.build(gpu.DEFAULT_BASE,replaced_dir,hop=0,replace_existing=True)
            self.assertEqual(replacement['publish_mode'], 'replace_delete_old_versions')
            replacement_source=''.join(json.loads((replaced_dir/'experiment.ipynb').read_text())['cells'][0]['source'])
            ast.parse(replacement_source)
            worker_matches = re.findall(r"worker.write_bytes\(base64.b64decode\('([^']+)'\)\)", replacement_source)
            self.assertTrue(worker_matches)
            worker = base64.b64decode(worker_matches[-1]).decode('utf-8')
            self.assertIn("argv.append('--replace-existing')", worker)
            self.assertEqual(binding['overlay_sha256'],second['overlay_sha256'])
            self.assertEqual(binding['notebook_sha256'],second['notebook_sha256'])
            meta=json.loads((output/'kernel-metadata.json').read_text())
            self.assertTrue(meta['is_private'] and meta['enable_gpu'])
            self.assertEqual(meta['dataset_sources'],[])
            self.assertEqual(binding['dataset_ids'],[
                'pamcks/h3-ptd-v2-shared-20261103',
                'pamcks/h3-ptd-v2-hop-00-20261103'])
            self.assertTrue(binding['publication_ack_is_not_remote_proof'])
            self.assertIn(runner.LICENSE_RELATIVE,binding['overlay_files'])
            self.assertEqual(qualify(output)['status'],'passed')
            self.assertEqual(hashlib.sha256((output/'experiment.ipynb').read_bytes()).hexdigest(),
                             binding['notebook_sha256'])


if __name__=='__main__':
    unittest.main(verbosity=2)
