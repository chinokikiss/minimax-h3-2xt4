"""Observer identity, publication/readiness and syscall accounting fixtures."""
from concurrent.futures import Future
import gc
from pathlib import Path
import sys
import tempfile
import threading
import unittest

ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'benchmarks'),str(ROOT/'ComfyUI')]
import cpu_causal_trace as cpu


class CPUTraceTests(unittest.TestCase):
    def test_weak_object_identity(self):
        with tempfile.TemporaryDirectory() as folder:
            r=cpu.CPURecorder(folder,0,'cuda:0');f=Future();row=r.identity(f,'future')
            self.assertIs(r.identity(f,'future'),row)
            del f;gc.collect();self.assertIsNone(row['ref']())

    def test_exact_future_publication_link_and_original_result(self):
        with tempfile.TemporaryDirectory() as folder:
            r=cpu.CPURecorder(folder,0,'cuda:0');cpu.install(r);r.enabled=True
            try:
                future=Future();started=threading.Event()
                def producer():
                    started.wait();future.set_result(42)
                thread=threading.Thread(target=producer);thread.start();started.set()
                self.assertEqual(future.result(timeout=10),42);thread.join(10)
                wait=next(e for e in r.events if e['name']=='future_consumer_wait')
                ready=next(e for e in r.events if e['name']=='future_ready_bounds')
                self.assertEqual(wait['future'],ready['future'])
                self.assertLessEqual(ready['lower_ns'],ready['upper_ns'])
                self.assertIn('thread_cpu_running_ns',wait)
                self.assertTrue(any(e['name']=='python_event/set' for e in r.events))
            finally:r.enabled=False;cpu.base.restore()


if __name__=='__main__':unittest.main()
