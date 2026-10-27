"""Fixed workload, reference order, and actual eight-step adapter contracts."""
import copy
import json
from pathlib import Path
import unittest

from benchmarks.anime_e2e_contract import validate
from benchmarks.turbo_e2e_contract import validate_header


class AnimeE2E(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(__file__).resolve().parents[1]
        if not (root/'turbo_plan.json').exists():
            root = Path('D:/attention_eval/h3_anime_e2e_20261027/plan_files')
        cls.root = root
        cls.plan = json.loads((root/'turbo_plan.json').read_text())
        cls.references = json.loads((root/'l1_assets_identity.json').read_text())['images']

    def test_exact_requested_workload_and_native_frame_grid(self):
        result = validate(self.plan, self.references)
        self.assertEqual(result['frames'], 243)
        self.assertEqual(result['video_latent_t'], 72)
        self.assertEqual(result['duration_seconds'], 10.125)

    def test_sampler_resolution_prompt_and_adapter_substitution_rejected(self):
        for name, value in (('sampler', 'lcm'), ('width', 960), ('steps', 4), ('prompt', 'shortened')):
            with self.assertRaises(ValueError):
                validate(self.plan | {name: value}, self.references)
        changed = copy.deepcopy(self.plan)
        changed['turbo_lora']['file'] = changed['turbo_lora']['file'].replace('8step', '4step')
        with self.assertRaises(ValueError):
            validate(changed, self.references)
        with self.assertRaises(ValueError):
            validate(self.plan, self.references[::-1][:1])

    def test_real_eight_step_header_matches_all_native_targets(self):
        header = json.loads((self.root/'turbo_lora_header.json').read_text())
        native = json.loads((self.root/'bundle_assets.json').read_text())['auxiliary']['dit']['native_header']
        self.assertEqual(len(validate_header(header, native)), 208)
