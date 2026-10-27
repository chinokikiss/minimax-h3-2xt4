import copy
import unittest

from benchmarks.e2e_attention_validation import denoiser_attention_counts


class E2EAttentionValidation(unittest.TestCase):
    def fixture(self):
        rows = []
        for nfe in (1, 2):
            rows.extend([dict(kind='kitchen/int8_attention', command='denoise', nfe=nfe)] * 2)
            rows.append(dict(kind='command_gpu_span', command='denoise', nfe=nfe))
        return dict(commands=[dict(command='denoise', nfe=n) for n in (1, 2)], gpu=rows)

    def test_old_temporal_decode_with_stale_denoise_tag_is_separated(self):
        benchmark = self.fixture()
        benchmark['gpu'].extend([dict(kind='kitchen/int8_attention', command='denoise', nfe=2)] * 5)
        result = denoiser_attention_counts(benchmark)
        self.assertEqual(result['per_step'], {1: 2, 2: 2})
        self.assertEqual(result['total'], 4)
        self.assertEqual(result['outside_command'], 5)

    def test_correctly_tagged_temporal_decode_is_not_denoising(self):
        benchmark = self.fixture()
        benchmark['gpu'].extend([dict(kind='kitchen/int8_attention', command='video_temporal_pair', nfe=2)] * 5)
        result = denoiser_attention_counts(benchmark)
        self.assertEqual(result['total'], 4)
        self.assertEqual(result['outside_command'], 0)

    def test_genuine_extra_attention_inside_a_denoiser_is_still_counted(self):
        benchmark = self.fixture()
        benchmark['gpu'].insert(2, dict(kind='kitchen/int8_attention', command='denoise', nfe=1))
        self.assertEqual(denoiser_attention_counts(benchmark)['per_step'], {1: 3, 2: 2})

    def test_incomplete_or_duplicate_completion_markers_are_rejected(self):
        benchmark = self.fixture()
        for rows in (benchmark['gpu'][:-1], benchmark['gpu'] + [copy.copy(benchmark['gpu'][-1])]):
            with self.assertRaises(ValueError):
                denoiser_attention_counts(benchmark | dict(gpu=rows))
