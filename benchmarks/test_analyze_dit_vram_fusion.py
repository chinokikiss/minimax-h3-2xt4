"""CPU-only analysis fixtures; never hardware measurements."""
import json
from pathlib import Path
import tempfile
import unittest
from benchmarks.analyze_dit_vram_fusion import aggregate,summarize,markdown


def measurements(samples,peak):
    return dict(samples=[dict(cuda_ms=v,wall_ms=v+1) for v in samples],peak_allocated=peak*2**30,
                initial_allocated=5*2**30,peak_reserved=13*2**30,device_used_after=14*2**30)


class Results(unittest.TestCase):
    def test_aligned_worst_rank_then_median(self):
        ranks=[dict(measurements={'a':measurements([10,12,10],12)}),dict(measurements={'a':measurements([15,9,11],11)})]
        row=aggregate(ranks,'a')
        self.assertEqual(row['median_cuda_ms'],12)
        self.assertEqual(row['worst_rank_peak_allocated_gib'],12)

    def test_rejects_short_or_misaligned_trials(self):
        with self.assertRaises(ValueError):aggregate([dict(measurements={'a':measurements([1,2],8)}),dict(measurements={'a':measurements([1,2,3],8)})],'a')

    def test_partial_artifacts_have_no_invented_measurements(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);(root/'result.json').write_text(json.dumps(dict(status='partial_or_failed')))
            result=summarize(root)
            self.assertFalse(result['physical_two_t4_receipt']);self.assertFalse(result['default_promoted'])
            self.assertTrue(all(value['status']=='incomplete' for value in result['sections'].values()))
            self.assertIn('incomplete',markdown(result))


if __name__=='__main__':unittest.main(verbosity=2)
