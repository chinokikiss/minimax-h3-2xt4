"""Synthetic CPU integrity, isolation and legacy-bank compatibility contracts."""
import copy
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from safetensors.torch import save_file

from patches.ptd.auxiliary import HEADS, SplitAuxiliary, write_split
from patches.ptd.bank import FORMAT, PTDBank, repack_split_bank
from patches.ptd.schedule import hop_sigmas
from patches.tp2_bundle import BundleReader, sha
from patches.tp2_disk_bake import _persist_layer, atomic_json, state_proof


def tensors():
    values = {f'token_refiner.blocks.{i}.{target}.weight':
              (torch.arange(16).reshape(4, 4) / 7 + i).to(dtype)
              for i, dtype in enumerate((torch.bfloat16, torch.float16))
              for target in ('attn.qkv_proj', 'attn.out_proj', 'mlp.fc1', 'mlp.fc2')}
    values.update({n: torch.arange(12 if n.endswith('weight') else 3).reshape(
        (3, 4) if n.endswith('weight') else (3,)).float() / 9 for n in HEADS})
    return values


class AuxiliaryContracts(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(); self.addCleanup(temp.cleanup)
        self.root, self.values = Path(temp.name), tensors()
        self.targets = sorted(n.removesuffix('.weight') for n in self.values if n not in HEADS)
        self.recipe = 'a' * 64
        self.proof = write_split(self.values, self.root, self.recipe, 3, self.targets)

    def reader(self, proof=None):
        return SplitAuxiliary(self.root, proof or self.proof, self.recipe, 3)

    def corrupt(self, name, *, header=False):
        unit = self.reader().by_name[name]
        path = self.root / unit['file']; stat = path.stat()
        offset = 10 if header else unit['header_bytes'] + unit['header'][name]['data_offsets'][0]
        with path.open('r+b') as stream:
            stream.seek(offset); byte = stream.read(1); stream.seek(offset); stream.write(bytes([byte[0] ^ 1]))
        os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))

    def test_original_bytes_and_dtypes_and_no_payload_cache(self):
        reader = self.reader()
        for name, value in self.values.items():
            actual = reader.read_tensor(name)
            self.assertEqual(actual.dtype, value.dtype)
            self.assertTrue(torch.equal(actual.view(torch.uint8), value.view(torch.uint8)))
        self.assertFalse(any(isinstance(v, torch.Tensor) for v in vars(reader).values()))
        self.assertEqual(reader.stats['refiner']['verified_tensors'], 8)

    def test_head_consumption_opens_no_refiner_unit(self):
        for unit in self.proof['units']:
            if unit['role'] == 'refiner': (self.root / unit['file']).unlink()
        opened = []
        original = Path.open
        def observe(path, *args, **kwargs):
            opened.append(path); return original(path, *args, **kwargs)
        with patch.object(Path, 'open', observe):
            reader = self.reader()
            for name in HEADS: self.assertTrue(torch.equal(reader.read_tensor(name), self.values[name]))
        self.assertTrue(opened)
        self.assertTrue(all(path.parent.name == 'heads' for path in opened))
        self.assertEqual(reader.stats['refiner']['payload_bytes'], 0)
        self.assertEqual(reader.stats['refiner']['header_bytes'], 0)

    def test_payload_corruption_even_with_restored_timestamp_is_rejected(self):
        for name in ('final_layer.video_out.weight', 'final_layer.audio_out.bias', self.targets[0] + '.weight'):
            with self.subTest(name=name):
                proof = write_split(self.values, self.root, self.recipe, 3, self.targets)
                reader = self.reader(proof); reader.read_tensor(name)
                self.corrupt(name)
                with self.assertRaisesRegex(ValueError, 'checksum'): reader.read_tensor(name)

    def test_unconsumed_corrupt_refiner_does_not_block_valid_head(self):
        self.corrupt(self.targets[0] + '.weight')
        self.assertTrue(torch.equal(self.reader().read_tensor('final_layer.video_out.bias'),
                                   self.values['final_layer.video_out.bias']))
        with self.assertRaisesRegex(ValueError, 'checksum'): self.reader().read_tensor(self.targets[0] + '.weight')

    def test_header_corruption_rejected_before_tensor_allocation(self):
        self.corrupt('final_layer.video_out.weight', header=True)
        with patch('patches.ptd.auxiliary._UnitReader', side_effect=AssertionError('Must not allocate')):
            with self.assertRaisesRegex(ValueError, 'header checksum'):
                self.reader().read_tensor('final_layer.video_out.weight')

    def test_wrong_bake_hop_layout_and_state_proof_are_rejected(self):
        for field, value in (('hop', 4), ('recipe_sha256', 'b' * 64), ('layout', 'unknown'),
                             ('state_sha256', {})):
            proof = copy.deepcopy(self.proof); proof[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError): self.reader(proof)

    def test_missing_duplicate_and_unknown_units_are_rejected(self):
        for change in ('missing', 'duplicate', 'unknown', 'offset', 'shape', 'hash'):
            proof = copy.deepcopy(self.proof)
            unit = proof['units'][0]; name = next(iter(unit['tensors']))
            if change == 'missing': proof['units'].pop()
            elif change == 'duplicate': proof['units'].append(copy.deepcopy(unit))
            elif change == 'unknown': unit['role'] = 'unknown'
            elif change == 'offset': unit['header'][name]['data_offsets'][0] = 1
            elif change == 'shape': unit['header'][name]['shape'] = [2**63, 4]
            else: unit['tensors'][name] = 'invalid'
            with self.subTest(change=change), self.assertRaises(ValueError): self.reader(proof)

    def test_path_escape_is_rejected(self):
        for path in ('../escape.safetensors', str(self.root / 'absolute.safetensors'), r'..\escape.safetensors'):
            proof = copy.deepcopy(self.proof); proof['units'][0]['file'] = path
            with self.subTest(path=path), self.assertRaises(ValueError): self.reader(proof)

    def test_symlink_escape_is_rejected_at_consumption(self):
        unit = self.proof['units'][0]; path = self.root / unit['file']
        with tempfile.TemporaryDirectory() as directory:
            outside = Path(directory) / 'outside.safetensors'; outside.write_bytes(path.read_bytes()); path.unlink()
            try: path.symlink_to(outside)
            except OSError: self.skipTest('Host cannot create symlinks')
            with self.assertRaisesRegex(ValueError, 'escaped'): self.reader().read_tensor(next(iter(unit['tensors'])))

    def test_mutation_during_read_is_rejected(self):
        from patches.windowed_tp import CheckpointReader
        original = CheckpointReader.read_tensor
        def mutate(reader, name):
            value = original(reader, name)
            with reader.path.open('ab') as stream: stream.write(b'x')
            return value
        with patch.object(CheckpointReader, 'read_tensor', mutate):
            with self.assertRaisesRegex(RuntimeError, 'changed during'):
                self.reader().read_tensor('final_layer.video_out.weight')

    def test_split_refiner_matches_native_loading_and_fp16_fold(self):
        from types import SimpleNamespace
        from benchmarks.test_ptd_queue import native_tiny
        from patches.h3_tensor_flow import TensorFlowPolicy, prepare_dit
        from patches.ptd.queue import refiner_group
        source, oracle = native_tiny(), native_tiny()
        values = {n: v for n, v in self.values.items() if n in HEADS}
        for index, block in enumerate(oracle.token_refiner.blocks):
            for part, module in (('attn.qkv_proj', block.attn.base.qkv_proj),
                    ('attn.out_proj', block.attn.base.out_proj), ('mlp.fc1', block.mlp.fc1), ('mlp.fc2', block.mlp.fc2)):
                value = (torch.arange(module.weight.numel()).reshape(module.weight.shape) * .0001 + .02).bfloat16()
                values[f'token_refiner.blocks.{index}.{part}.weight'] = value
                with torch.no_grad(): module.weight.copy_(value)
        proof = write_split(values, self.root, self.recipe, 3, self.targets)
        split = self.reader(proof)
        reader = SimpleNamespace(hop=3, bank=SimpleNamespace(hops=[(None, {'auxiliary': proof})] * 8),
                                 read_tensor=split.read_tensor)
        flow = TensorFlowPolicy('fp16'); prepare_dit(source, flow, rank=0); prepare_dit(oracle, flow, rank=0)
        group = refiner_group({'condition_proj': source.condition_proj, 'token_refiner': source.token_refiner}, reader, flow)
        expected = torch.nn.ModuleDict({'condition_proj': oracle.condition_proj, 'token_refiner': oracle.token_refiner})
        for name, value in expected.state_dict().items(): self.assertTrue(torch.equal(group.state_dict()[name], value), name)
        self.assertEqual(group['token_refiner'].blocks[0].norm1.eps, oracle.token_refiner.blocks[0].norm1.eps)

    def test_split_and_fp32_sidecar_conflict_fails_before_any_weight_loading(self):
        from patches.ptd.nodes import MiniMaxH3TP2PTDBake
        with patch('patches.tp2_bundle.BundleReader', side_effect=AssertionError('Must not load weights')):
            with self.assertRaisesRegex(ValueError, 'FP32 sidecar'):
                MiniMaxH3TP2PTDBake().prepare(True, 'unused', 'unused', 'unused',
                                            runtime_layout=True, split_auxiliary=True)
            self.assertEqual(MiniMaxH3TP2PTDBake().prepare(False, 'unused', 'unused', 'unused',
                             runtime_layout=True, split_auxiliary=True), (None,))


class BankContracts(unittest.TestCase):
    def legacy(self):
        # Reuse the validated tiny bundle generator without inheriting its tests.
        from benchmarks import test_tp2_bundle as fixtures
        fixture = fixtures.Bundle(); fixture.setUp(); self.addCleanup(fixture.doCleanups)
        root = fixture.root / 'bank'; root.mkdir()
        base = BundleReader(fixture.descriptor)
        recipe = dict(base=str(fixture.descriptor), base_sha256=sha(fixture.descriptor),
            base_bundle_sha256=sha(base.bundle_file), checkpoint_sha256='c' * 64,
            layers=1, hops=8, steps=32, video_shift=12., audio_shift=3.)
        digest = hashlib.sha256(json.dumps(recipe, sort_keys=True).encode()).hexdigest()
        values = {n: v for n, v in tensors().items() if n in HEADS}
        values['token_refiner.blocks.0.attn.out_proj.weight'] = fixture.native['token_refiner.blocks.0.attn.out_proj.weight']
        targets = ['token_refiner.blocks.0.attn.out_proj']
        hops = []
        for hop in range(8):
            directory = root / f'hop_{hop:02d}'; directory.mkdir()
            shards = _persist_layer(fixture.local, 0, [], directory, base.plan['blocks'], digest)
            path = directory / 'aux_overlay.safetensors'; save_file(values, str(path))
            aux = dict(file=path.name, sha256=sha(path), bytes=path.stat().st_size,
                       state_sha256=state_proof(values), verified_targets=targets)
            atomic_json(directory / 'receipt.json', dict(format=FORMAT, status='complete', hop=hop,
                recipe_sha256=digest, shards=shards, auxiliary=aux, native_verified_targets=targets))
            hops.append(dict(directory=directory.name, receipt_sha256=sha(directory / 'receipt.json')))
        path = root / 'bank.json'
        atomic_json(path, dict(format=FORMAT, status='complete', recipe=recipe, recipe_sha256=digest,
            hops=hops, full_model=False, sigmas=hop_sigmas(8, 32, 12.).tolist(), online_lora=False))
        return fixture, path, values

    def test_repack_preserves_native_proofs_shards_and_legacy_reader(self):
        fixture, path, values = self.legacy()
        original = {p: sha(p) for p in path.parent.rglob('*') if p.is_file()}
        old = PTDBank(path, allow_fixture=True, expected_sha256=sha(path))
        repacked = repack_split_bank(path, fixture.root / 'split', allow_fixture=True, expected_sha256=old.digest)
        new = PTDBank(repacked, allow_fixture=True, expected_sha256=sha(repacked))
        self.assertNotEqual(old.digest, new.digest)
        self.assertEqual(old.recipe_sha, new.recipe_sha)
        for hop in range(8):
            a, b = old.reader(hop), new.reader(hop)
            self.assertNotEqual(a.identity, b.identity)
            for name, expected in values.items():
                self.assertTrue(torch.equal(a.read_tensor(name), expected))
                self.assertTrue(torch.equal(b.read_tensor(name), expected))
            for rank in (0, 1):
                self.assertEqual(sha(a._shard(0, rank).path), sha(b._shard(0, rank).path))
        self.assertEqual(original, {p: sha(p) for p in original})
        with self.assertRaisesRegex(ValueError, 'admitted manifest'):
            PTDBank(repacked, allow_fixture=True, expected_sha256=old.digest)

    def test_disabled_split_roles_use_original_base(self):
        fixture, path, values = self.legacy()
        repacked = repack_split_bank(path, fixture.root / 'split', allow_fixture=True)
        bank = PTDBank(repacked, allow_fixture=True)
        reader = bank.reader(0, refiner=False, heads=False)
        name = 'token_refiner.blocks.0.attn.out_proj.weight'
        self.assertTrue(torch.equal(reader.read_tensor(name), fixture.native[name]))
        self.assertTrue(all(row['verified_tensors'] == 0 for row in reader.split_auxiliary.stats.values()))

    def test_receipt_tamper_and_untrusted_repack_are_rejected(self):
        fixture, path, values = self.legacy()
        with self.assertRaisesRegex(ValueError, 'admitted manifest'):
            repack_split_bank(path, fixture.root / 'bad', allow_fixture=True, expected_sha256='0' * 64)
        receipt = path.parent / 'hop_00/receipt.json'
        record = json.loads(receipt.read_bytes()); record['auxiliary']['sha256'] = 'f' * 64
        atomic_json(receipt, record)
        with self.assertRaisesRegex(ValueError, 'receipt hash'): PTDBank(path, allow_fixture=True)

    def test_parent_passes_admitted_manifest_digest_to_both_workers(self):
        from types import SimpleNamespace
        from unittest.mock import Mock
        from patches.ptd.runtime import Controller, PTDOptions
        service = Mock()
        bank = SimpleNamespace(digest='d' * 64, sigmas=hop_sigmas(8))
        with patch('patches.ptd.runtime.PTDBank', return_value=bank) as constructor:
            options = PTDOptions(enabled=True)
            controller = Controller(service, 'bank.json', options, expected_bank_sha256=bank.digest)
            controller.begin()
        self.assertEqual(constructor.call_args.kwargs['expected_sha256'], bank.digest)
        self.assertEqual(service.call.call_args.args[-1], bank.digest)


if __name__ == '__main__': unittest.main()
