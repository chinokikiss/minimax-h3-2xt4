"""Regression gates for the Stage A and full-hop PTD Kaggle source delivery.

Build temporary private notebook packages without submitting them, then cold
import exactly those embedded overlays against the pinned frozen git source.
"""
import json
import re
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from benchmarks.verify_ptd_v2_delivery import qualify

ROOT = Path(__file__).resolve().parents[1]
STAGE_BUILDER = ROOT / 'benchmarks/build_ptd_v2_stage_a_job.py'
BAKE_BUILDER = ROOT / 'benchmarks/build_ptd_v2_bake_job.py'
OLD_STAGE = ROOT / 'docs/data/h3_ptd_v2_stage_a_t4_20261102/gpu_job_v2/source_binding.json'


class PTDDeliveryClosure(unittest.TestCase):
    def test_prior_failed_job_omitted_changed_rank_api(self):
        old = json.loads(OLD_STAGE.read_text(encoding='utf-8'))
        self.assertNotIn('patches/rank_local_int8.py', old['source_overlays'])

    def test_stage_a_and_bake_notebook_cold_import(self):
        with tempfile.TemporaryDirectory(prefix='ptd-v2-job-rehearsal-') as tmp:
            for builder, kind in ((STAGE_BUILDER, 'stage'), (BAKE_BUILDER, 'bake')):
                with self.subTest(kind=kind):
                    output = Path(tmp) / kind
                    subprocess.run([sys.executable, str(builder), '--output', str(output)],
                                   cwd=ROOT, check=True, capture_output=True, text=True,
                                   timeout=90)
                    binding = json.loads((output / 'source_binding.json').read_text(encoding='utf-8'))
                    proof_key = 'source_overlays' if kind == 'stage' else 'reviewed_overlay_files'
                    proofs = binding[proof_key]
                    self.assertIn('patches/rank_local_int8.py', proofs)
                    self.assertTrue(set(binding['source_dependency_closure']) <= set(proofs))
                    self.assertTrue(binding['no_kaggle_status_polling'] if kind == 'stage'
                                    else binding['no_status_polling'])
                    metadata = json.loads((output / 'kernel-metadata.json').read_text())
                    self.assertTrue(metadata['is_private'])
                    slug = re.sub(r'[^a-z0-9]+', '-', metadata['title'].lower()).strip('-')
                    self.assertEqual(slug, metadata['id'].split('/')[-1])
                    qualification = qualify(output)
                    self.assertEqual(qualification['status'], 'passed')
                    self.assertIn('PTD_V2_FROZEN_IMPORT_PASS', qualification['stdout'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
