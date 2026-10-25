"""Offline ownership gates for the one-shot artifact collector; no API calls."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from benchmarks.collect_kaggle_final_once import collect


class FinalCollection(unittest.TestCase):
    def fixture(self,root):
        path=root/'submission.json'
        path.write_text(json.dumps(dict(returncode=0,kernel='owner/kernel',final_collection_count=0,
            source_binding=dict(publication=dict(capsule_sha256='expected')),
            expected_prior_inputs=dict(noise_0=dict(sha256='noise')))))
        return path

    def test_only_one_call_and_summary_is_separate_from_quality(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);path=self.fixture(root);calls=[];out=root/'output'
            def run(argv,**kw):
                calls.append(argv)
                binding=out/'light_setup/delivery_binding.json';binding.parent.mkdir()
                binding.write_text(json.dumps(dict(capsule_sha256='expected')))
                result=out/'light_execution/e2e/L1/result.json';result.parent.mkdir(parents=True)
                result.write_text(json.dumps(dict(status='failed',sampler_steps=[1],denoiser_outputs=[1],
                    inputs=dict(noise_0=dict(sha256='noise')))))
                return SimpleNamespace(returncode=0,stdout='',stderr='')
            first=collect(path,out,run=run);second=collect(path,out,run=run)
            self.assertEqual(len(calls),1)
            self.assertEqual(first['experiment']['status'],'failed')
            self.assertEqual(first['experiment']['completed_sampler_steps'],1)
            self.assertTrue(first['prior_input_hash_matches']['noise_0'])
            self.assertEqual(second['status'],'already_claimed')
            self.assertEqual(json.loads(path.read_text())['final_collection_count'],1)

    def test_wrong_source_or_no_files_is_not_a_pass(self):
        for wrong_source in (False,True):
            with tempfile.TemporaryDirectory() as temporary:
                root=Path(temporary);path=self.fixture(root);out=root/'output'
                def run(*args,**kw):
                    if wrong_source:
                        p=out/'light_setup/delivery_binding.json';p.parent.mkdir()
                        p.write_text(json.dumps(dict(capsule_sha256='older-version')))
                    return SimpleNamespace(returncode=0,stdout='',stderr='')
                status=collect(path,out,run=run)['status']
                self.assertEqual(status,'source_binding_mismatch' if wrong_source else 'final_outputs_unavailable')

    def test_operator_collection_retains_coverage_blocker_without_generation_claim(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);path=self.fixture(root);out=root/'output'
            state=json.loads(path.read_text());state.update(experiment_kind='review_operators',
                result_relative='review_operators/result.json');path.write_text(json.dumps(state))
            def run(*args,**kw):
                binding=out/'light_setup/delivery_binding.json';binding.parent.mkdir()
                binding.write_text(json.dumps(dict(capsule_sha256='expected')))
                result=out/'review_operators/result.json';result.parent.mkdir()
                result.write_text(json.dumps(dict(status='operators_complete_profiler_blocked',
                    all_outputs_byte_exact=True,ranks=[dict(cases=[1,2]),dict(cases=[1,2])],
                    profiling=dict(status='blocked',blocker='Incomplete worker CUDA coverage'))))
                return SimpleNamespace(returncode=0,stdout='',stderr='')
            result=collect(path,out,run=run)['experiment']
            self.assertEqual(result['cases_by_rank'],[2,2]);self.assertTrue(result['all_outputs_byte_exact'])
            self.assertEqual(result['profiling']['status'],'blocked')
            self.assertNotIn('completed_sampler_steps',result)


if __name__=='__main__':unittest.main(verbosity=2)
