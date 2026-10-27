"""Prevent ungated GPU launches or accidental full-H3/private-input publication."""
import ast
import base64
from contextlib import redirect_stderr
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'benchmarks')]
from benchmarks.build_kaggle_light_job import main, execution_supervisor


def expanded(source):
    pending, result = [source], []
    while pending:
        text = pending.pop();result.append(text)
        for node in ast.walk(ast.parse(text)):
            if (isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute)
                    and node.func.attr=='b64decode' and node.args and isinstance(node.args[0],ast.Constant)):
                pending.append(base64.b64decode(node.args[0].value).decode())
    return '\n'.join(result)


class DecoderDelivery(unittest.TestCase):
    def fixture(self, root):
        values = dict(publication=dict(published=True,capsule_sha256='a'*64),
            private=dict(private=True,sha256='b'*64),
            qualification=dict(status='cpu_ready',capsule_sha256='a'*64,review_suite=dict(passed=True)),
            gate=dict(status='cpu_input_ready',source_capsule_sha256='a'*64,input=dict(file_sha256='b'*64)))
        for name, value in values.items():
            (root/(name+'.json')).write_text(json.dumps(value))

    def args(self, root, mode):
        return ['builder','--mode',mode,'--publication',str(root/'publication.json'),
            '--private-input',str(root/'private.json'),'--output',str(root/'job')]

    def test_cpu_preflight_has_no_gpu_and_uses_hash_auth_probe(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);self.fixture(root)
            with patch.object(sys,'argv',self.args(root,'cpu')+['--vae-compare-preflight','--review-suite']):
                main()
            meta=json.loads((root/'job/kernel-metadata.json').read_text())
            self.assertFalse(meta['enable_gpu'])
            self.assertEqual(meta['dataset_sources'],[])
            code=expanded(''.join(json.loads((root/'job/experiment.ipynb').read_text())['cells'][0]['source']))
            self.assertIn('--cpu-preflight',code)
            self.assertIn('run_vae_temporal_batch.py',code)

    def test_gpu_requires_matching_cpu_input_then_runs_only_decoder_batch(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);self.fixture(root)
            args=self.args(root,'vae-parallel')+['--qualification',str(root/'qualification.json')]
            with patch.object(sys,'argv',args),redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):
                main()
            args+=['--vae-input-qualification',str(root/'gate.json')]
            with patch.object(sys,'argv',args):
                main()
            code=expanded(''.join(json.loads((root/'job/experiment.ipynb').read_text())['cells'][0]['source']))
            self.assertIn('run_vae_temporal_batch.py',code)
            self.assertNotIn('launch_kaggle_light.py',code)
            self.assertNotIn('run_h3_l1_turbo_batch.py',code)
            meta=json.loads((root/'job/kernel-metadata.json').read_text())
            self.assertEqual(meta['machine_shape'],'NvidiaTeslaT4')

    def test_private_worker_launcher_is_removed_even_if_startup_fails(self):
        execution="import os,pathlib,sys\nenv={}\nargv=[]\nos.execve(sys.executable,argv,env)"
        with patch('huggingface_hub.get_token',return_value='TEST_ONLY_AUTH'):
            code=execution_supervisor(execution,budget=30,output='/tmp/private_vae_test',private_auth=True)
        ast.parse(code)
        self.assertIn('finally:',code)
        self.assertIn("joinpath('bounded_worker.py').unlink(missing_ok=True)",code)


if __name__ == '__main__':
    unittest.main()
