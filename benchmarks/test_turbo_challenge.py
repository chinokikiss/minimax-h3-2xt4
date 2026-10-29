"""New request boundaries and view-only ten-second export, CPU only."""
import copy
import json
from pathlib import Path
import unittest
import torch

from benchmarks.turbo_challenge_contract import from_previous,validate,export_views


class TurboChallenge(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root=Path(__file__).resolve().parents[1]
        if not (root/'turbo_plan.json').exists():root=Path('D:/attention_eval/h3_anime_e2e_20261027/plan_files')
        cls.previous=json.loads((root/'turbo_plan.json').read_text(encoding='utf-8'))

    def test_only_geometry_and_export_change_from_regular_turbo(self):
        before=copy.deepcopy(self.previous);plan=from_previous(self.previous)
        self.assertEqual(self.previous,before)
        for key in ('prompt','prompt_sha256','reference_images','turbo_lora','sampler','steps','seed','video_shift','audio_shift'):
            self.assertEqual(plan[key],before[key])
        self.assertEqual(validate(plan)['generated_video_tokens'],72576)

    def test_native_grid_matches_the_requested_latent_geometry(self):
        from comfy_extras.nodes_minimax_h3 import _empty_av_latent
        plan=from_previous(self.previous);latent,frames=_empty_av_latent(plan['width'],plan['height'],plan['frames'])
        video,audio=latent['samples'].unbind()
        self.assertEqual(frames,243);self.assertEqual(list(video.shape),[1,24,72,48,84])
        self.assertEqual(list(audio.shape),[1,32,2,405])

    def test_online_ptd_lower_resolution_and_wrong_duration_rejected(self):
        plan=from_previous(self.previous)
        for key,value in (('width',1152),('frames',240),('export_frames',243),('sampler','ptd'),
                ('offline_lora',False),('lora_stack',[('adapter',1.)]),('ptd_release',{})):
            with self.assertRaises(ValueError):validate(plan|{key:value})

    def test_export_slices_do_not_copy_or_change_native_decode(self):
        plan=from_previous(self.previous)
        pixels=torch.ones(1,1,1,1,3).expand(1,243,768,1344,3)
        audio=torch.zeros(1,2,324000)
        video,wave=export_views(pixels,audio,plan)
        self.assertEqual(list(video.shape),[1,240,768,1344,3]);self.assertEqual(wave.shape[-1],320000)
        self.assertEqual(video.data_ptr(),pixels.data_ptr());self.assertEqual(wave.data_ptr(),audio.data_ptr())
        self.assertEqual(pixels.shape[1],243);self.assertEqual(audio.shape[-1],324000)

    def test_missing_decoded_frames_or_audio_cannot_pass(self):
        plan=from_previous(self.previous);pixels=torch.ones(1,1,1,3).expand(243,768,1344,3)
        with self.assertRaises(ValueError):export_views(pixels[:239],torch.zeros(2,324000),plan)
        with self.assertRaises(ValueError):export_views(pixels,torch.zeros(2,100),plan)

    def test_runtime_keeps_current_operators_and_trusted_bake_default(self):
        from benchmarks.run_h3_l1_turbo_e2e import runtime_config
        plan=from_previous(self.previous)|dict(paths=dict(te='te',dit='baked'))
        config=runtime_config(plan)
        self.assertFalse(config.verify_dit_weights);self.assertEqual(config.lora_stack,())
        self.assertTrue(config.operators.dit_fused_norm and config.operators.dit_fused_row_gate)
        self.assertTrue(config.operators.dit_fused_gate and config.operators.dit_prepared_attention)


if __name__=='__main__':unittest.main(verbosity=2)
