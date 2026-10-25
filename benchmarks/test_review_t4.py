"""Keep the follow-up experiment bounded and distinct from full-H3 experiments."""
import ast
import base64
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import weakref
from benchmarks.run_review_t4 import fixture_plan,expected_labels,ROWS,REPETITIONS,reference_on_host,recovery_skipped_draws

ROOT=Path(__file__).resolve().parents[1]


class ReviewT4(unittest.TestCase):
    def test_validation_reference_does_not_retain_source_storage(self):
        import torch
        handles=[]
        def produce():
            value=torch.tensor([-0.,1.,-2.],dtype=torch.float16)
            handles.append(weakref.ref(value))
            return (value,[value.view(1,3)])
        reference=reference_on_host(produce)
        self.assertIsNone(handles[0]())
        self.assertEqual(reference[0].device.type,'cpu')
        self.assertEqual(reference[1][0].device.type,'cpu')
        self.assertTrue(torch.equal(reference[0].view(torch.uint8),
            torch.tensor([-0.,1.,-2.],dtype=torch.float16).view(torch.uint8)))

    def test_recovery_preserves_rng_draw_order_without_repeating_operators(self):
        import torch
        rows=(1,2,3);seed=20261024
        torch.manual_seed(seed)
        for projection in ('qkv_proj','fc1'):
            for count in rows:torch.randn(count,5376,dtype=torch.float16)
        torch.randn(1,2,5120,dtype=torch.float16)
        expected=torch.get_rng_state().clone()
        torch.manual_seed(seed)
        for shape in recovery_skipped_draws(rows=rows,te_rows=2):torch.randn(*shape,dtype=torch.float16)
        self.assertTrue(torch.equal(expected,torch.get_rng_state()))

    def plan(self):
        return dict(repository='public/base',revision='a'*40,token_required=False,
            objects={f'{kind}.block00.rank{rank}.safetensors':dict(bytes=200000000,sha256='b'*64)
                     for kind in ('dit','te') for rank in (0,1)})

    def test_fixture_plan_refuses_full_bundle_mutable_revision_and_auth(self):
        self.assertEqual(len(fixture_plan(self.plan())['objects']),4)
        for field,value in [('revision','main'),('token_required',True)]:
            plan=self.plan();plan[field]=value
            with self.assertRaises(ValueError):fixture_plan(plan)
        plan=self.plan();plan['objects']['dit.block01.rank0.safetensors']={}
        with self.assertRaises(ValueError):fixture_plan(plan)
        plan=self.plan();next(iter(plan['objects'].values()))['bytes']=2**30
        with self.assertRaises(ValueError):fixture_plan(plan)

    def test_both_worker_trace_labels_cover_each_selected_route(self):
        labels=expected_labels()
        self.assertEqual(len(labels),14);self.assertEqual(len(set(labels)),14)
        self.assertIn('column/fc1/cuda_int8_tilelang',labels)
        self.assertIn('te/qkv/shared',labels);self.assertIn('qkv_lora/owned',labels)
        self.assertEqual(ROWS,(261,4096,51220));self.assertEqual(REPETITIONS,5)

    def test_cpu_and_gpu_jobs_have_no_mounts_and_bind_matching_review_qualification(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);pub=root/'publication.json';qual=root/'qualification.json'
            pub.write_text(json.dumps(dict(published=True,capsule_sha256='a'*64)))
            qual.write_text(json.dumps(dict(status='cpu_ready',capsule_sha256='a'*64,review_suite=dict(passed=True))))
            for mode in ('cpu','review-operators'):
                out=root/mode
                argv=[sys.executable,str(ROOT/'benchmarks/build_kaggle_light_job.py'),'--mode',mode,
                    '--publication',str(pub),'--output',str(out),'--setup-budget','480','--execution-budget','1200']
                argv+=['--review-suite'] if mode=='cpu' else ['--qualification',str(qual),'--review-recovery']
                result=subprocess.run(argv,capture_output=True,text=True,timeout=20)
                self.assertEqual(result.returncode,0,result.stderr)
                meta=json.loads((out/'kernel-metadata.json').read_text())
                code=''.join(json.loads((out/'experiment.ipynb').read_text())['cells'][0]['source'])
                embedded=[base64.b64decode(ast.literal_eval(n.args[0])).decode() for n in ast.walk(ast.parse(code))
                    if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='b64decode']
                code+='\n'+'\n'.join(embedded)
                self.assertEqual(meta['dataset_sources'],[]);self.assertEqual(meta['kernel_sources'],[])
                self.assertNotIn('private_input_locator',code)
                if mode=='cpu':self.assertIn('--review-suite',code);self.assertFalse(meta['enable_gpu'])
                else:
                    self.assertEqual(meta['machine_shape'],'NvidiaTeslaT4')
                    self.assertIn('run_review_t4.py',code)
                    self.assertIn('--retry-lora-only',code)
                    self.assertTrue(json.loads((out/'source_binding.json').read_text())['review_recovery'])
                    self.assertNotIn('run_h3_l1_current',code)
            broken=json.loads(qual.read_text());broken['review_suite']['passed']=False;qual.write_text(json.dumps(broken))
            result=subprocess.run([sys.executable,str(ROOT/'benchmarks/build_kaggle_light_job.py'),
                '--mode','review-operators','--publication',str(pub),'--qualification',str(qual),
                '--output',str(root/'invalid')],capture_output=True,text=True,timeout=20)
            self.assertNotEqual(result.returncode,0)


if __name__=='__main__':unittest.main(verbosity=2)
