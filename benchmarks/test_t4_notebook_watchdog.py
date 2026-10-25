"""Exercise the outer deadline with CPU-only workers, without Kaggle calls."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from benchmarks.t4_notebook_watchdog import wrap_notebook


BODY='''def run(command,name,cap):
    remaining=min(cap,deadline-time.monotonic());entry=dict(name=name,argv=command)
    with stream:
        process=not_called
        try:entry['exit_code']=process.wait(timeout=remaining)
        except Exception:pass
if False:
    receipts=list(())
    asset_files=list(())
    frozen.prepare_source(None)
    base=prep.inventory();minor='test'
    with archive.open('rb') as stream:pass
    with tarfile.open(archive,'r:') as tar:pass
    prep.install_overlay(None)
    assert prep.inventory()==base
    nvcc=shutil.which('nvcc')
print('TEST_WORKER_COMPLETE',flush=True)
'''


class Watchdog(unittest.TestCase):
    def execute(self,body,budget):
        with tempfile.TemporaryDirectory() as folder:
            code=wrap_notebook(body,budget=budget,output=folder)
            script=Path(folder)/'cell.py';script.write_text(code)
            result=subprocess.run([sys.executable,str(script)],capture_output=True,text=True,timeout=10)
            self.assertEqual(result.returncode,0,result.stderr)
            return json.loads((Path(folder)/'supervision.json').read_text()),result.stdout

    def test_worker_completion_is_not_an_experiment_pass(self):
        receipt,output=self.execute(BODY,3)
        self.assertEqual(receipt['status'],'worker_exited')
        self.assertEqual(receipt['exit_code'],0)
        self.assertIn('notebook_cell_started',output)
        self.assertIn('worker_started',output)
        self.assertIn('TEST_WORKER_COMPLETE',output)

    def test_deadline_covers_preparation_before_any_test(self):
        receipt,output=self.execute('import time;print("TEST_SETUP_STALL",flush=True);time.sleep(30)\n'+BODY,.5)
        self.assertEqual(receipt['status'],'deadline_exceeded')
        self.assertLess(receipt['seconds'],7)
        self.assertIn('TEST_SETUP_STALL',output)
        self.assertNotIn('TEST_WORKER_COMPLETE',output)


if __name__=='__main__':unittest.main()
