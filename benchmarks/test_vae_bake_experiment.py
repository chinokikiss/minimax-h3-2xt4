"""Instrumentation transparency and immutable replay contracts (CPU only)."""
import inspect
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
import torch
from benchmarks.profile_tp2_disk_bake import Recorder
from benchmarks.test_tp2_bundle import Bundle


class Recording(unittest.TestCase):
    def test_nested_scopes_exclusive_wall_and_single_invocation(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder=Recorder(directory,'cpu')
            calls=[]
            def function(x):calls.append(x);return x+1
            obj=SimpleNamespace(f=function)
            recorder.wrap(obj,'f','inner')
            with recorder.span('outer','matrix'):
                self.assertEqual(obj.f(3),4)
            recorder.finish()
            self.assertIs(obj.f,function)
            self.assertEqual(calls,[3])
            data=json.loads((Path(directory)/'cpu.json').read_text())
            inner,outer=data['events']
            self.assertEqual(inner['target'],'matrix')
            self.assertEqual(outer['wall_ns'],outer['exclusive_wall_ns']+inner['wall_ns'])

    def test_descriptors_and_exceptions_restore(self):
        class Example:
            @classmethod
            def c(cls,x):return cls.__name__+str(x)
            @staticmethod
            def s(x):raise ValueError(x)
        originals={n:inspect.getattr_static(Example,n) for n in ('c','s')}
        with tempfile.TemporaryDirectory() as directory:
            recorder=Recorder(directory,'cpu')
            for name in originals:recorder.wrap(Example,name,name)
            self.assertEqual(Example().c(4),'Example4')
            with self.assertRaisesRegex(ValueError,'sentinel'):Example.s('sentinel')
            recorder.finish()
        for name,original in originals.items():self.assertIs(inspect.getattr_static(Example,name),original)


class NativeProof(Bundle):
    # Bundle's existing source/LoRA fixtures also exercise actual Kitchen CPU
    # dequantization and quantization, rather than a mocked observer path.
    def test_profiled_merge_keeps_native_bytes(self):
        from patches import offline_lora
        from patches.h3_lora import H3LoRAStore
        from patches.windowed_tp import CheckpointReader
        reader=CheckpointReader(self.source)
        store=H3LoRAStore(((str(self.lora),.7),),reader)
        target='blocks.0.mlp.fc2'
        reference=offline_lora.merged_weight(reader,store,target,'cpu')
        recorder=Recorder(self.root/'profile','cpu')
        recorder.install()
        try:actual=offline_lora.merged_weight(reader,store,target,'cpu')
        finally:recorder.finish()
        for a,b in zip(actual,reference):
            if a is None:self.assertIsNone(b)
            else:self.assertTrue(torch.equal(a,b))
        data=json.loads((self.root/'profile'/'cpu.json').read_text())
        self.assertEqual(data['phases']['production_merge']['calls'],1)
        self.assertGreater(data['phases']['checkpoint_read']['calls'],0)


if __name__=='__main__':unittest.main()
