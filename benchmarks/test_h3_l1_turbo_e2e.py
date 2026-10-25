"""Current-default long run and real Turbo header compatibility, CPU only."""
from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import unittest

from benchmarks.run_h3_l1_turbo_e2e import runtime_config, kitchen_denoiser_calls
from benchmarks.run_h3_l1_turbo_batch import release_owned_dit_payloads
from patches.h3_operator_policy import H3OperatorPolicy

ROOT = Path(__file__).resolve().parents[1]


from benchmarks.turbo_e2e_contract import validate_header


class TurboLongRun(unittest.TestCase):
    def test_decode_kitchen_calls_do_not_change_dit_acceptance_count(self):
        rows=[dict(kind='kitchen/int8_attention',command='denoise') for _ in range(200)]
        rows += [dict(kind='kitchen/int8_attention',command='video_pixels') for _ in range(32)]
        rows += [dict(kind='command_gpu_span',command='denoise')]
        self.assertEqual(kitchen_denoiser_calls(dict(gpu=rows)),200)

    def test_current_defaults_and_offline_only_runtime(self):
        config = dict(offline_lora=True, disk_baked_lora=True, width=960, height=544,
                      paths=dict(te='te_bundle.json', dit='dit_baked.json'))
        runtime = runtime_config(config)
        self.assertEqual(runtime.lora_stack, ())
        self.assertFalse(runtime.bundle_offline_lora)
        self.assertEqual(asdict(runtime.operators), asdict(H3OperatorPolicy()))
        self.assertEqual(runtime.operators.dit_attention, 'kitchen')
        self.assertEqual(runtime.operators.column_linear_backend, 'cuda_int8_tilelang')
        self.assertEqual(runtime.tensor_flow.mode, 'fp16')
        self.assertTrue(runtime.adaptive_lookahead and runtime.prefetch_lifecycle and runtime.memory_pressure_aware)
        for overrides in (dict(disk_baked_lora=False), dict(offline_lora=False), dict(height=540)):
            with self.assertRaises(ValueError):
                runtime_config(config | overrides)

    def test_real_header_when_bound_in_frozen_capsule(self):
        header_path = ROOT / 'turbo_lora_header.json'
        if not header_path.exists():
            # The public-file header is bound by the delivery builder; no
            # network or multi-GB model transfer is needed for local unit tests.
            header_path = Path('D:/attention_eval/h3_l1_turbo_05mp_20261024/turbo_header.json')
            native_path = header_path.parent / 'bundle_assets_prior.json'
        else:
            native_path = ROOT / 'bundle_assets.json'
        self.assertTrue(header_path.exists() and native_path.exists(), 'Actual frozen Turbo/base headers required')
        header = json.loads(header_path.read_text())
        native = json.loads(native_path.read_text())['auxiliary']['dit']['native_header']
        self.assertEqual(len(validate_header(header, native)), 208)

    def test_download_cleanup_is_scoped_and_requires_complete_owned_files(self):
        with tempfile.TemporaryDirectory() as folder:
            scratch = Path(folder)
            root = scratch / 'base_bundle'
            root.mkdir()
            with self.assertRaises(ValueError):
                release_owned_dit_payloads(scratch, root)
            (scratch / '.owned_by_turbo_e2e').write_text('fixture')
            paths = [root / f'dit.block{i:02d}.rank{r}.safetensors' for i in range(50) for r in (0, 1)]
            for path in paths:
                path.write_bytes(b'ab')
            keep = root / 'bundle.json'
            keep.write_text('metadata')
            self.assertEqual(release_owned_dit_payloads(scratch, root), 200)
            self.assertTrue(keep.exists())
            self.assertFalse(any(p.exists() for p in paths))
