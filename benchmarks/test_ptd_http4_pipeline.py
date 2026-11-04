"""CPU-only behavioral tests for the opt-in HTTP4 prefetch/Bank handoff."""
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import threading
import time
import types
import unittest
from unittest.mock import patch

from benchmarks import ptd_http4_staging as stage


class Http4Pipeline(unittest.TestCase):
    def fixture(self, root):
        mounts, proofs = {}, {}
        for hop in (0, 1):
            part = f'hop_{hop:02d}'
            folder = root / 'mounted' / part
            folder.mkdir(parents=True)
            shards = {}
            for block in range(50):
                for rank in (0, 1):
                    name = f'dit.block{block:02d}.rank{rank}'
                    (folder / (name + '.safetensors')).write_bytes(b'1234')
                    (folder / (name + '.json')).write_text('{"ok":true}')
                    shards[name] = {'bytes':4}
            (folder / 'aux_overlay.safetensors').write_bytes(b'1234')
            (folder / 'receipt.json').write_text(json.dumps({
                'hop':hop, 'shards':shards,
                'auxiliary':{'file':'aux_overlay.safetensors','bytes':4}}))
            mounts[part] = folder
            proofs[part] = {'dataset':'pamcks/fake-' + part}
        return mounts, proofs

    def test_http4_size_gated_publish_and_bank_wait(self):
        from patches.ptd.bank_v2 import _http4_stage_file
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            mounts, proofs = self.fixture(root)
            scratch = root / 'http_stage'
            metrics = root / 'metrics.json'

            fake_kagglehub = types.ModuleType('kagglehub')
            fake_config = types.ModuleType('kagglehub.config')
            fake_config.is_kaggle_cache_disabled = lambda: True
            def download(handle, path, output_dir=None):
                time.sleep(0.001)
                src = mounts[handle.split('fake-')[1]] / path.split('/')[-1]
                dest = Path(output_dir) / src.name
                shutil.copyfile(src, dest)
                return str(dest)
            fake_kagglehub.dataset_download = download

            with patch.dict(sys.modules, {'kagglehub':fake_kagglehub,
                                          'kagglehub.config':fake_config}), \
                 patch.object(stage.importlib.metadata, 'version',return_value='1.0.3'), \
                 patch.object(stage, 'RESERVE_BYTES',0), \
                 patch.dict(os.environ, {
                    'H3_PTD_HTTP4_PIPELINE_ROOT':str(scratch),
                    'H3_PTD_HTTP4_PIPELINE_METRICS':str(metrics)}):
                errors = []
                def producer():
                    try:
                        stage.stage_two_hops(mounts,proofs,metrics,scratch=scratch)
                    except BaseException as exc:
                        errors.append(exc)
                background = threading.Thread(target=producer,daemon=True)
                background.start()
                shard = _http4_stage_file(0,'dit.block05.rank0.safetensors',4)
                self.assertEqual(shard.read_bytes(),b'1234')
                self.assertTrue(shard.with_name(shard.name+'.ready').is_file())
                background.join(timeout=15)
                self.assertFalse(background.is_alive())
                self.assertFalse(errors)
                outcome = json.loads(metrics.read_text())
                self.assertEqual(outcome['completed_files'],202)
                self.assertEqual(outcome['completed_bytes'],808)
                self.assertEqual(outcome['status'],'complete')
                self.assertIsNone(_http4_stage_file(2,'missing',4))

    def test_readiness_checks_expected_byte_size(self):
        from patches.ptd.bank_v2 import _http4_stage_file
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)/'parts'/'hop_00'
            folder.mkdir(parents=True)
            file = folder/'bad.safetensors'
            file.write_bytes(b'12')
            (folder/'bad.safetensors.ready').write_text('complete')
            with patch.dict(os.environ,{'H3_PTD_HTTP4_PIPELINE_ROOT':str(Path(temp))}):
                with self.assertRaisesRegex(RuntimeError,'Incomplete HTTP-staged'):
                    _http4_stage_file(0,'bad.safetensors',4)


if __name__ == '__main__':
    unittest.main()
