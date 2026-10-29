"""Original overlay integrity at first consumption; no retired storage backends."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import inspect
import json
import os
import unittest
from unittest.mock import patch

import torch
from safetensors.torch import save_file
import benchmarks.test_tp2_bundle as bundle_fixture
from patches.ptd.bank import FORMAT, PTDBank
from patches.ptd.schedule import hop_sigmas
from patches.tp2_bundle import BundleReader,sha
from patches.tp2_disk_bake import _persist_layer,atomic_json,state_proof


class OverlayContracts(unittest.TestCase):
    def setUp(self):
        fixture=bundle_fixture.Bundle(methodName='runTest'); fixture.setUp(); self.addCleanup(fixture.doCleanups)
        self.fixture=fixture; root=fixture.root/'bank'; root.mkdir()
        base=BundleReader(fixture.descriptor)
        recipe=dict(base=str(fixture.descriptor),base_sha256=sha(fixture.descriptor),
            base_bundle_sha256=sha(base.bundle_file),layers=1,hops=8,steps=32,video_shift=12.,audio_shift=3.)
        digest=hashlib.sha256(json.dumps(recipe,sort_keys=True).encode()).hexdigest()
        self.name='token_refiner.blocks.0.attn.out_proj.weight'
        self.values={self.name:fixture.native[self.name]+.25,
            'final_layer.video_out.weight':torch.ones(3,8),'final_layer.video_out.bias':torch.zeros(3),
            'final_layer.audio_out.weight':torch.ones(2,8),'final_layer.audio_out.bias':torch.zeros(2)}
        hops=[]
        for hop in range(8):
            directory=root/f'hop_{hop:02d}'; directory.mkdir()
            shards=_persist_layer(fixture.local,0,[],directory,base.plan['blocks'],digest)
            overlay=directory/'aux_overlay.safetensors'; save_file(self.values,str(overlay))
            auxiliary=dict(file=overlay.name,sha256=sha(overlay),bytes=overlay.stat().st_size,
                state_sha256=state_proof(self.values),verified_targets=[self.name.removesuffix('.weight')])
            atomic_json(directory/'receipt.json',dict(format=FORMAT,status='complete',hop=hop,
                recipe_sha256=digest,shards=shards,auxiliary=auxiliary,native_verified_targets=auxiliary['verified_targets']))
            hops.append(dict(directory=directory.name,receipt_sha256=sha(directory/'receipt.json')))
        self.path=root/'bank.json'
        atomic_json(self.path,dict(format=FORMAT,status='complete',recipe=recipe,recipe_sha256=digest,
            hops=hops,full_model=False,sigmas=hop_sigmas(8).tolist(),online_lora=False))
        self.bank=PTDBank(self.path,allow_fixture=True,expected_sha256=sha(self.path))

    def test_metadata_admission_defers_hash_until_first_read_then_reuses_it(self):
        path=self.bank.hops[0][0]/'aux_overlay.safetensors'
        with patch('patches.ptd.bank.sha',wraps=sha) as checksum:
            reader=self.bank.reader(0)
            self.assertFalse(reader.overlay_verified)
            self.assertFalse(any(call.args[0]==path for call in checksum.call_args_list))
            for name,expected in self.values.items():
                self.assertTrue(torch.equal(reader.read_tensor(name),expected))
            reader.read_tensor(self.name)
            self.assertEqual(sum(call.args[0]==path for call in checksum.call_args_list),1)

    def test_concurrent_consumers_verify_only_once(self):
        reader=self.bank.reader(0)
        with patch('patches.ptd.bank.sha',wraps=sha) as checksum:
            with ThreadPoolExecutor(max_workers=4) as pool:
                values=list(pool.map(reader.read_tensor,[self.name]*4))
            self.assertTrue(all(torch.equal(v,self.values[self.name]) for v in values))
            self.assertEqual(checksum.call_count,1)

    def test_corruption_is_rejected_on_consumption_even_with_same_size_and_mtime(self):
        reader=self.bank.reader(0); path=reader.overlay.path; before=path.stat()
        data=path.read_bytes(); path.write_bytes(data[:-1]+bytes([data[-1]^1]))
        os.utime(path,ns=(before.st_atime_ns,before.st_mtime_ns))
        with self.assertRaisesRegex(ValueError,'hash mismatch'): reader.read_tensor(self.name)
        self.assertFalse(reader.overlay_verified)

    def test_mutation_after_verification_fails_without_another_full_hash(self):
        reader=self.bank.reader(0); reader.read_tensor(self.name)
        path=reader.overlay.path; before=path.stat()
        os.utime(path,ns=(before.st_atime_ns,before.st_mtime_ns+1000000))
        with patch('patches.ptd.bank.sha',side_effect=AssertionError('No redundant hash')):
            with self.assertRaisesRegex(RuntimeError,'overlay changed'): reader.read_tensor(self.name)

    def test_disabled_roles_keep_native_base_without_reading_overlay(self):
        reader=self.bank.reader(0,refiner=False,heads=False)
        with patch.object(reader,'_verify_overlay',side_effect=AssertionError('Disabled role')):
            self.assertTrue(torch.equal(reader.read_tensor(self.name),self.fixture.native[self.name]))

    def test_retired_layout_and_tampered_receipts_fail_admission(self):
        receipt=self.bank.hops[0][0]/'receipt.json'
        record=json.loads(receipt.read_text()); record['auxiliary']={'format':'retired-layout','units':[]}
        atomic_json(receipt,record)
        with self.assertRaisesRegex(ValueError,'receipt hash'): PTDBank(self.path,allow_fixture=True)
        manifest=json.loads(self.path.read_text()); manifest['hops'][0]['receipt_sha256']=sha(receipt)
        atomic_json(self.path,manifest)
        with self.assertRaisesRegex(ValueError,'monolithic'): PTDBank(self.path,allow_fixture=True)


class SurfaceContracts(unittest.TestCase):
    def test_removed_options_are_not_exposed_and_p1_stays_off(self):
        from patches.ptd import bank
        from patches.ptd.runtime import PTDOptions
        from patches.ptd.nodes import MiniMaxH3TP2PTDBake,MiniMaxH3TP2PTDAdapter
        retired={'split_auxiliary','runtime_layout','baked_refiner'}
        for cls in (MiniMaxH3TP2PTDBake,MiniMaxH3TP2PTDAdapter):
            self.assertFalse(retired & set().union(*(set(v) for v in cls.INPUT_TYPES().values())))
        for function in (bank.bake,bank.disk_plan,PTDOptions):
            self.assertFalse(retired & inspect.signature(function).parameters.keys())
        self.assertFalse(PTDOptions().prefetch_refiner)
        with self.assertRaises(TypeError): PTDOptions(baked_refiner=True)

    def test_future_delivery_prunes_retired_sources_from_historical_capsules(self):
        from benchmarks.ptd_delivery_sources import RETIRED,apply
        files={name:b'historical snapshot' for name in RETIRED}
        files['ComfyUI/immutable.py']=b'Native source bytes'
        changed=apply(files)
        self.assertFalse(set(RETIRED)&files.keys())
        self.assertEqual(files['ComfyUI/immutable.py'],b'Native source bytes')
        self.assertIn('patches/ptd/bank.py',changed)


if __name__=='__main__': unittest.main(verbosity=2)
