"""Disk publication, Native equivalence and no inference-time rebaking."""
import json
from pathlib import Path
from unittest.mock import patch

import torch
from safetensors.torch import save_file
from benchmarks.test_tp2_bundle import Bundle, block
from patches.tp2_bundle import open_checkpoint, split_matrix, sha
from patches.tp2_disk_bake import bake, DiskBakedReader
from patches.offline_lora import merged_weight
from patches.h3_lora import H3LoRAStore
from patches.windowed_tp import CheckpointReader
from patches.rank_local_int8 import CONFIG, FORMAT


class DiskBake(Bundle):
    def setUp(self):
        super().setUp()
        plans = json.loads((self.bundle / 'bundle.json').read_text())['plan']['blocks']
        for index in range(1, 50):
            for rank in (0, 1):
                stem = f'dit.block{index:02d}.rank{rank}'
                path = self.bundle / (stem + '.safetensors')
                save_file(self.local[rank], str(path))
                (self.bundle / (stem + '.json')).write_text(json.dumps(dict(format=FORMAT,
                    configuration=CONFIG, **plans[stem], native_tensor_bit_exact=True, file_sha256=sha(path))))

    def export(self):
        output = self.root / 'disk_bake'
        path = bake(self.descriptor, ((str(self.lora), .7),), output, devices=('cpu',))
        return output, path

    def test_disk_bake_is_native_exact_and_has_no_remerge_after_base_payload_removal(self):
        before = {p: sha(p) for p in self.bundle.iterdir()}
        output, descriptor = self.export()
        self.assertEqual(before, {p: sha(p) for p in self.bundle.iterdir()})
        record = json.loads((output / 'bake_receipt.json').read_text())
        self.assertEqual(len(record['shards']), 100)
        self.assertEqual(record['verified_native_targets'], 6)
        self.assertFalse(record['online_lora'] or record['cpu_remerge_during_inference'])
        for path in self.bundle.glob('dit.block*.safetensors'):
            path.unlink()  # Only this test's private temporary copies.
        reader = open_checkpoint(descriptor)
        self.assertIsInstance(reader, DiskBakedReader)
        with patch('patches.tp2_bundle.shell', side_effect=lambda kind, rank: block(self.local[rank])), \
                patch('patches.offline_lora.merged_weight', side_effect=AssertionError('No inference-time merge')):
            results = [reader.load_rank(0, rank).state_dict() for rank in (0, 1)]
            for rank in (0, 1):
                reader.load_rank(0, rank)
            auxiliary = reader.read_tensor('token_refiner.blocks.0.attn.out_proj.weight')
        source = CheckpointReader(self.source)
        store = H3LoRAStore(((str(self.lora), .7),), source)
        from patches.tp2_bundle import DIT_LINEAR
        for name, local in DIT_LINEAR.items():
            value, scale = merged_weight(source, store, 'blocks.0.' + name, 'cpu')
            for rank in (0, 1):
                self.assertTrue(torch.equal(results[rank][local + '.weight'], split_matrix(value, local, rank)))
                if scale is not None:
                    self.assertTrue(torch.equal(results[rank][local + '.weight_scale'], split_matrix(scale, local, rank, scale=True)))
        expected, _ = merged_weight(source, store, 'token_refiner.blocks.0.attn.out_proj', 'cpu')
        self.assertTrue(torch.equal(auxiliary, expected))
        self.assertEqual(reader.stats['merged_matrices'], 0)
        self.assertEqual(reader.workspace_bytes(0), 0)
        self.assertGreater(reader.stats['persistent_baked_bytes'], 0)
        with self.assertRaises(ValueError):
            reader.set_offline_lora(((str(self.lora), .7),))

    def test_cache_recipe_and_corruption_are_rejected(self):
        output, descriptor = self.export()
        self.assertEqual(bake(self.descriptor, ((str(self.lora), .7),), output, devices=('cpu',)), descriptor)
        with self.assertRaisesRegex(ValueError, 'different'):
            bake(self.descriptor, ((str(self.lora), .6),), output, devices=('cpu',))
        path = output / 'dit.block00.rank0.safetensors'
        raw = path.read_bytes()
        path.write_bytes(raw[:-1] + bytes([raw[-1] ^ 1]))
        with self.assertRaisesRegex(ValueError, 'payload'):
            DiskBakedReader(descriptor)._shard(0, 0)

    def test_failure_never_publishes_a_complete_descriptor(self):
        original = sha(self.bundle / 'dit.block00.rank0.safetensors')
        with patch('patches.offline_lora_reference.verify_weight', side_effect=AssertionError('fixture failure')):
            with self.assertRaises(AssertionError):
                self.export()
        self.assertFalse((self.root / 'disk_bake').exists())
        self.assertTrue((self.root / 'disk_bake.partial').is_dir())
        self.assertEqual(sha(self.bundle / 'dit.block00.rank0.safetensors'), original)
