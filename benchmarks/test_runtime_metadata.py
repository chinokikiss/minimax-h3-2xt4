"""Production readers do no content proofs; cheap guards still fail closed."""
from contextlib import ExitStack
import json
import os
from pathlib import Path
import struct
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from safetensors.torch import save_file
from benchmarks import test_tp2_bundle as fixtures
from benchmarks import test_ptd_overlay as overlays
from patches.tp2_bundle import BundleReader, open_checkpoint
from patches.windowed_tp import CheckpointReader


class RuntimeMetadata(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.Bundle(methodName='runTest')
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)

    def forbid_hashes(self):
        stack = ExitStack()
        for name in ('patches.tp2_bundle.sha','patches.tp2_disk_bake.sha',
                     'patches.ptd.bank.sha','patches.tp2_bundle.state_hashes',
                     'patches.rank_local_int8.state_hashes','hashlib.sha256','hashlib.file_digest'):
            stack.enter_context(patch(name,side_effect=AssertionError('Runtime content hash: '+name)))
        stack.enter_context(patch('patches.tp2_bundle.shell',side_effect=lambda kind,rank:fixtures.block(self.fixture.local[rank])))
        return stack

    def test_te_and_dit_load_exact_weights_without_any_sha256(self):
        root = self.fixture.root
        te = json.loads(self.fixture.descriptor.read_text())
        te['kind'] = 'te'
        te['native_header'] = CheckpointReader(self.fixture.aux).header
        te_path = root/'te.json';te_path.write_text(json.dumps(te))
        original = self.fixture.bundle/'dit.block00.rank0.safetensors'
        target = self.fixture.bundle/'te.block00.rank0.safetensors'
        target.write_bytes(original.read_bytes())
        manifest = json.loads(original.with_suffix('.json').read_text());manifest['kind']='te'
        target.with_suffix('.json').write_text(json.dumps(manifest))
        with self.forbid_hashes():
            for path in (te_path,self.fixture.descriptor):
                reader = open_checkpoint(path,verify_weights=False)
                self.assertFalse(reader.verify_payloads)
                self.assertIn('metadata-checked-weights-v1',reader.identity)
                for _ in range(2):
                    state = reader.load_rank(0,0).state_dict()
                    for name,value in self.fixture.local[0].items():
                        self.assertTrue(torch.equal(state[name],value))
                for name,value in self.fixture.native.items():
                    if not name.startswith('blocks.'):
                        self.assertTrue(torch.equal(reader.read_tensor(name),value))

    def test_replacement_with_same_size_and_mtime_is_rejected(self):
        reader = BundleReader(self.fixture.descriptor,verify_payloads=False)
        reader._shard(0,0)
        path = self.fixture.bundle/'dit.block00.rank0.safetensors'
        before = path.stat();replacement=path.with_suffix('.replacement')
        replacement.write_bytes(path.read_bytes())
        os.utime(replacement,ns=(before.st_atime_ns,before.st_mtime_ns))
        replacement.replace(path)
        with self.assertRaisesRegex(RuntimeError,'changed'):reader.check()

    def test_mtime_and_manifest_mutation_are_rejected_without_hashing(self):
        with self.forbid_hashes():
            reader = open_checkpoint(self.fixture.descriptor,verify_weights=False)
            reader.load_rank(0,0)
            manifest = self.fixture.bundle/'dit.block00.rank0.json'
            info = manifest.stat()
            os.utime(manifest,ns=(info.st_atime_ns,info.st_mtime_ns+1_000_000))
            with self.assertRaisesRegex(RuntimeError,'changed'):reader.load_rank(0,0)

    def test_tensor_dtype_is_checked_without_tensor_hashes(self):
        path = self.fixture.bundle/'dit.block00.rank0.safetensors'
        state = dict(self.fixture.local[0]);state['qkv_proj.weight']=state['qkv_proj.weight'].half()
        save_file(state,str(path))
        with self.forbid_hashes():
            reader = open_checkpoint(self.fixture.descriptor,verify_weights=False)
            with self.assertRaisesRegex(ValueError,'schema'):reader.load_rank(0,0)

    def test_safetensors_truncation_and_bad_header_are_rejected(self):
        path = self.fixture.root/'broken.safetensors'
        path.write_bytes(self.fixture.aux.read_bytes()[:-1])
        with self.assertRaisesRegex(ValueError,'range|size'):CheckpointReader(path)
        path.write_bytes(struct.pack('<Q',2**63))
        with self.assertRaisesRegex(ValueError,'header length'):CheckpointReader(path)

    def test_disk_baked_descriptor_and_weights_load_without_hashes(self):
        from benchmarks import test_tp2_disk_bake as disk
        fixture=disk.DiskBake(methodName='runTest');fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        _,descriptor=fixture.export()
        with patch('patches.tp2_bundle.shell',side_effect=lambda kind,rank:fixtures.block(fixture.local[rank])):
            reference=open_checkpoint(descriptor)
            expected=[reference.load_rank(0,rank).state_dict() for rank in (0,1)]
            auxiliary=reference.read_tensor('token_refiner.blocks.0.attn.out_proj.weight')
        # Bake and its publication proofs precede the runtime no-hash boundary.
        with self.forbid_hashes():
            reader=open_checkpoint(descriptor,verify_weights=False)
            for rank in (0,1):
                state=reader.load_rank(0,rank).state_dict()
                self.assertEqual(set(state),set(expected[rank]))
                for name,value in expected[rank].items():self.assertTrue(torch.equal(state[name],value))
            self.assertTrue(torch.equal(reader.read_tensor('token_refiner.blocks.0.attn.out_proj.weight'),auxiliary))
            reader.check()

    def test_legacy_rank_blocks_skip_proof_and_keep_guards(self):
        from patches import rank_local_int8 as rank
        source = open_checkpoint(self.fixture.source,verify_weights=False)
        root = self.fixture.bundle
        path = root/'dit.block00.rank0.json'
        record = json.loads(path.read_text());record['source_identity']=list(source.identity)
        path.write_text(json.dumps(record))
        with self.forbid_hashes(), patch.object(rank,'shell',side_effect=lambda kind,owner:fixtures.block(self.fixture.local[owner])):
            block = rank.load_offline_block(source,'dit',0,0,root)
            self.assertTrue(torch.equal(block.qkv_proj.weight,self.fixture.local[0]['qkv_proj.weight']))
            path.touch()
            stat = path.stat();os.utime(path,ns=(stat.st_atime_ns,stat.st_mtime_ns+1_000_000))
            with self.assertRaisesRegex(RuntimeError,'changed'):rank.load_offline_block(source,'dit',0,0,root)

    def test_runtime_forces_metadata_only_for_both_te_and_dit_even_for_legacy_true(self):
        from patches.comfy_tp_runtime import RankRuntime,RuntimeConfig
        from patches.host_weight_cache import HostBudget
        from patches.ptd.runtime import WorkerAdapter,Controller,PTDOptions
        config = RuntimeConfig('te','dit',verify_dit_weights=True)
        self.assertFalse(config.verify_dit_weights)
        readers = []
        def open_reader(path,**options):
            readers.append(options)
            return SimpleNamespace(header={},identity=path)
        with patch('patches.tp2_bundle.open_checkpoint',side_effect=open_reader):
            runtime = RankRuntime(config,0,'cpu',None,HostBudget(config.host_bytes,pressure_aware=False))
            runtime.cache.close();runtime.startup_cache.close()
        self.assertEqual(readers,[{'verify_weights':False},{'verify_weights':False}])
        for cls in (WorkerAdapter,Controller):
            with patch('patches.ptd.runtime.PTDBank',side_effect=RuntimeError('stop before GPU')) as bank:
                with self.assertRaisesRegex(RuntimeError,'stop before GPU'):
                    cls(SimpleNamespace(config=config),'bank',PTDOptions(enabled=True))
                self.assertIs(bank.call_args.kwargs['verify_payloads'],False)


class PTDMetadata(unittest.TestCase):
    def test_cached_bake_node_does_not_rehash_base_bank_or_checkpoint(self):
        from patches.ptd.nodes import MiniMaxH3TP2PTDBake
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);(root/'bank.json').write_text('{}')
            checkpoint=root/'adapter.safetensors';checkpoint.write_bytes(b'existing adapter')
            bank=SimpleNamespace(base=root/'base.json',recipe={'checkpoint_bytes':checkpoint.stat().st_size})
            base=SimpleNamespace(path=bank.base,header={})
            with patch('patches.tp2_bundle.BundleReader',return_value=base) as base_reader, \
                    patch('patches.ptd.bank.PTDBank',return_value=bank) as bank_reader, \
                    patch('hashlib.sha256',side_effect=AssertionError('Cache hit hash')):
                result=MiniMaxH3TP2PTDBake().prepare(True,bank.base,str(checkpoint),str(root))
                self.assertEqual(result,(str((root/'bank.json').resolve()),))
                self.assertIs(base_reader.call_args.kwargs['verify_payloads'],False)
                self.assertIs(bank_reader.call_args.kwargs['verify_payloads'],False)

    def test_all_hops_and_auxiliary_match_without_any_hashes(self):
        fixture = overlays.OverlayContracts(methodName='runTest');fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        from patches.ptd.bank import PTDBank
        with ExitStack() as stack:
            for name in ('patches.tp2_bundle.sha','patches.tp2_disk_bake.sha','patches.ptd.bank.sha',
                         'patches.tp2_bundle.state_hashes','hashlib.sha256','hashlib.file_digest'):
                stack.enter_context(patch(name,side_effect=AssertionError('Runtime hash: '+name)))
            stack.enter_context(patch('patches.tp2_bundle.shell',side_effect=lambda kind,rank:fixtures.block(fixture.fixture.local[rank])))
            bank = PTDBank(fixture.path,allow_fixture=True,verify_payloads=False)
            peer = PTDBank(fixture.path,allow_fixture=True,verify_payloads=False,expected_sha256=bank.digest)
            self.assertEqual(bank.digest,peer.digest)
            base = BundleReader(fixture.fixture.descriptor,verify_payloads=False)
            for hop in range(8):
                reader = bank.reader(hop,base_reader=base)
                self.assertIsNot(reader.file_checks,base.file_checks)
                for rank in (0,1):
                    state = reader.load_rank(0,rank).state_dict()
                    for name,value in fixture.fixture.local[rank].items():self.assertTrue(torch.equal(state[name],value))
                for name,value in fixture.values.items():self.assertTrue(torch.equal(reader.read_tensor(name),value))
            with self.assertRaisesRegex(ValueError,'metadata differs'):
                PTDBank(fixture.path,allow_fixture=True,verify_payloads=False,expected_sha256=('metadata','wrong'))


class RuntimeArtifacts(unittest.TestCase):
    def test_source_and_native_library_admission_do_not_hash(self):
        from patches import kitchen_attention as kitchen,kitchen_te_attention as te,legacy_runtime
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder);binary=root/'fixture.so';binary.write_bytes(b'library')
            receipt=dict(status='passed',target='sm_75',compiler_version='V12.8.93',exit_code=0,
                         sources={},artifact_sha256='unused',artifact_bytes=7)
            binary.with_name('build_receipt.json').write_text(json.dumps(receipt))
            with patch('hashlib.sha256',side_effect=AssertionError('Runtime hash')):
                self.assertEqual(kitchen.validate_library(binary,'75',verify_hashes=False),binary)
                self.assertEqual(te.validate_library(binary,verify_hashes=False),binary)
                source=Path(__file__).resolve().parents[1]/'ComfyUI'
                legacy_runtime.validate_source(source,verify_hashes=False)
            binary.write_bytes(b'truncated')
            with self.assertRaisesRegex(RuntimeError,'changed'):te.validate_library(binary,verify_hashes=False)


if __name__=='__main__':unittest.main(verbosity=2)
