"""Synthetic evidence guards only; these fixtures are never hardware results."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from benchmarks.summarize_h3_row_partial_t4 import summarize


class SummaryGuards(unittest.TestCase):
    def fixture(self,root,missing=False,duplicate_device=False):
        gpu=root/'gpu';folder=gpu/'t4_validation';folder.mkdir(parents=True)
        cpu=root/'cpu';profile=cpu/'native_profile';profile.mkdir(parents=True)
        submitted=root/'submitted';submitted.mkdir()
        def write(path,value):path.write_text(json.dumps(value))
        write(profile/'preflight_receipt.json',dict(status='cpu_ready'))
        write(folder/'source_hashes.json',{})
        write(submitted/'source_binding.json',dict(files_sha256={},cpu_receipt_sha256=hashlib.sha256((profile/'preflight_receipt.json').read_bytes()).hexdigest()))
        devices=[dict(uuid=str(i),capability=[7,5],name='Tesla T4') for i in (0,1)]
        if duplicate_device:devices[1]['uuid']=devices[0]['uuid']
        write(folder/'batch_results.json',dict(hardware=dict(devices=devices,kitchen='0.2.37',tilelang='0.1.13')))
        for rank in (0,1):
            records=[]
            for kind,projection in (('row','out_proj'),('row','fc2'),('block','whole_block')):
                for layer in (0,49):
                    for stack in (1,2):
                        for mode in ('gather','partial'):
                            records.append(dict(kind=kind,projection=projection,layer=layer,strengths=[1.]*stack,mode=mode,tokens=513,
                                timing=dict(samples_ms=[10. if mode=='gather' else 5.]*10,peak_increment_bytes=[4]*10,resident_before_bytes=[100]*10),
                                vs_gather=dict(nonfinite=0),finite=True,independent_lora_exact=True,output=dict(nonfinite=0),provenance={}))
            if missing and rank==1:records.pop()
            write(folder/f'combined_rows_rank{rank}.json',dict(status='completed_not_generation_acceptance',rank=rank,
                hardware=dict(uuid=str(rank)),records=records))
        write(folder/'reference_demand_vae.json',dict(status='passed'))
        for base in (cpu,gpu):
            files=[dict(file=p.relative_to(base).as_posix(),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in base.rglob('*.json')]
            write(base/'download_manifest.json',dict(files=files))
        return gpu,submitted,cpu

    def test_complete_aligned_ranks_and_missing_case(self):
        for missing in (False,True):
            with tempfile.TemporaryDirectory() as folder:
                result=summarize(*self.fixture(Path(folder),missing=missing))
                self.assertEqual(result['status'],'incomplete_or_failed' if missing else 'passed_component_scope')
                if not missing:
                    partial=[r for r in result['rows'] if r['mode']=='partial']
                    self.assertTrue(all(r['median_paired_speedup']==2. for r in partial))

    def test_duplicate_gpu_identity_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(AssertionError):summarize(*self.fixture(Path(folder),duplicate_device=True))

    def test_partial_without_gather_control_is_incomplete(self):
        with tempfile.TemporaryDirectory() as folder:
            gpu,submitted,cpu=self.fixture(Path(folder))
            for rank in (0,1):
                path=gpu/f't4_validation/combined_rows_rank{rank}.json'
                value=json.loads(path.read_text());value['records'].pop(0)
                path.write_text(json.dumps(value))
            manifest=gpu/'download_manifest.json';value=json.loads(manifest.read_text())
            for item in value['files']:
                item['sha256']=hashlib.sha256((gpu/item['file']).read_bytes()).hexdigest()
            manifest.write_text(json.dumps(value))
            result=summarize(gpu,submitted,cpu)
            self.assertEqual(result['status'],'incomplete_or_failed')
            self.assertTrue(any(r.get('comparison_status')=='missing_gather_control' for r in result['rows']))

    def test_empty_provider_output_is_unavailable_not_a_hardware_pass(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);gpu,submitted,cpu=self.fixture(root)
            empty=root/'empty';empty.mkdir()
            (empty/'download_manifest.json').write_text(json.dumps(dict(files=[])))
            result=summarize(empty,submitted,cpu)
            self.assertEqual(result['status'],'gpu_outputs_unavailable')
            self.assertEqual(result['rows'],[])


if __name__=='__main__':unittest.main()
