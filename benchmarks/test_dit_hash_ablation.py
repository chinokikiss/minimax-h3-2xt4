"""Trusted-bake ablation skips payload work, retaining schema/signature guards."""
from dataclasses import replace
import json
import os
import struct
import unittest
from unittest.mock import patch

import torch
from benchmarks.test_ptd_overlay import OverlayContracts
from patches.ptd.bank import PTDBank
from patches.tp2_bundle import BundleReader,sha


class HashAblation(unittest.TestCase):
    def setUp(self):
        OverlayContracts.setUp(self)
        from benchmarks.test_tp2_bundle import block
        shell=patch('patches.tp2_bundle.shell',side_effect=lambda kind,rank:block(self.fixture.local[rank]))
        shell.start();self.addCleanup(shell.stop)

    def test_default_verifies_once_and_reload_reuses_the_proof(self):
        reader=self.bank.reader(0)
        from patches import tp2_bundle,tp2_disk_bake
        with patch.object(tp2_bundle,'state_hashes',wraps=tp2_bundle.state_hashes) as states, \
                patch.object(tp2_disk_bake,'sha',wraps=sha) as files:
            reader.load_rank(0,0)
            first_files=files.call_count
            self.assertEqual(states.call_count,1)
            reader.load_rank(0,0)
            self.assertEqual(states.call_count,1)
            self.assertEqual(files.call_count,first_files)

    def test_trusted_payloads_skip_file_and_tensor_hashes_and_match_verified_weights(self):
        reference=self.bank.reader(0).load_rank(0,0).state_dict()
        from patches import tp2_bundle,tp2_disk_bake
        def small_metadata_only(path):
            self.assertNotEqual(path.suffix,'.safetensors','Unexpected payload hashing')
            return sha(path)
        with patch.object(tp2_bundle,'sha',side_effect=small_metadata_only), \
                patch.object(tp2_disk_bake,'sha',side_effect=small_metadata_only), \
                patch('patches.ptd.bank.sha',side_effect=small_metadata_only), \
                patch.object(tp2_bundle,'state_hashes',side_effect=AssertionError('Unexpected tensor hashing')):
            bank=PTDBank(self.path,allow_fixture=True,verify_payloads=False)
            reader=bank.reader(0)
            state=reader.load_rank(0,0).state_dict()
            for name,value in reference.items():self.assertTrue(torch.equal(value,state[name]),name)
            self.assertTrue(torch.equal(reader.read_tensor(self.name),self.values[self.name]))
        self.assertNotEqual(reader.identity,self.bank.reader(0).identity)

    def test_same_signature_corruption_is_accepted_only_by_trusted_arm(self):
        bank=PTDBank(self.path,allow_fixture=True,verify_payloads=False)
        trusted=bank.reader(0);verified=self.bank.reader(0)
        path=bank.hops[0][0]/'dit.block00.rank0.safetensors'
        original=path.stat();data=bytearray(path.read_bytes())
        length=struct.unpack('<Q',data[:8])[0];header=json.loads(data[8:8+length])
        offset=8+length+header['qkv_proj.weight']['data_offsets'][0]
        data[offset]^=1;path.write_bytes(data)
        os.utime(path,ns=(original.st_atime_ns,original.st_mtime_ns))
        with self.assertRaisesRegex(ValueError,'payload/proof'):verified.load_rank(0,0)
        changed=trusted.load_rank(0,0)
        self.assertFalse(torch.equal(changed.qkv_proj.weight,self.fixture.local[0]['qkv_proj.weight']))

    def test_trusted_arm_still_rejects_changed_file_signature(self):
        reader=PTDBank(self.path,allow_fixture=True,verify_payloads=False).reader(0)
        reader.load_rank(0,0)
        path=reader.root/'dit.block00.rank0.safetensors';before=path.stat()
        os.utime(path,ns=(before.st_atime_ns,before.st_mtime_ns+1000000))
        with self.assertRaisesRegex(RuntimeError,'changed'):reader.load_rank(0,0)

    def test_queued_readers_override_mode_without_mutating_admitted_base(self):
        base=BundleReader(self.fixture.descriptor)
        trusted=PTDBank(self.path,allow_fixture=True,verify_payloads=False).reader(0,base_reader=base)
        self.assertFalse(trusted.verify_payloads)
        self.assertTrue(base.verify_payloads)
        self.assertIsNot(trusted.verified,base.verified)

    def test_runtime_default_trusts_bake_and_te_cannot_be_opted_out(self):
        from patches.comfy_tp_runtime import RuntimeConfig
        default=RuntimeConfig('te','dit')
        self.assertFalse(default.verify_dit_weights)
        self.assertNotEqual(default,replace(default,verify_dit_weights=True))
        from custom_nodes.minimax_h3_tp2 import MiniMaxH3TP2Loader
        self.assertFalse(MiniMaxH3TP2Loader.INPUT_TYPES()['optional']['verify_dit_weights'][1]['default'])
        with self.assertRaises(ValueError):replace(default,verify_dit_weights=0)
        descriptor=json.loads(self.fixture.descriptor.read_text());descriptor['kind']='te'
        with self.assertRaisesRegex(ValueError,'restricted to DiT'):
            BundleReader(self.fixture.descriptor,descriptor,verify_payloads=False)


if __name__=='__main__':unittest.main(verbosity=2)
