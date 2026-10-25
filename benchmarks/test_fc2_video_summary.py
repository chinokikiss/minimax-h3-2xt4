"""Reject incomplete rank evidence; execution completion is not Video identity."""
import unittest
from benchmarks.summarize_fc2_video_experiments import complete_matrix,timings


class SummaryEvidence(unittest.TestCase):
    def test_aligned_samples_cannot_be_silently_truncated(self):
        a=dict(samples_ms=[1.]*7,peak_allocated_bytes=1)
        b=dict(samples_ms=[2.]*6,peak_allocated_bytes=1)
        with self.assertRaises(ValueError):timings(a,b)
        b['samples_ms'].append(3.)
        self.assertEqual(timings(a,b)['samples_ms'],[2.]*6+[3.])

    def test_execution_requires_both_ranks_every_case_and_video_stages(self):
        rows=[dict(component=c,layer=l,preparation=m,transport=t,
            status='completed_operator_diagnostic',finite=True,independent_codec_epilogue_exact=True)
            for c,l in (('te',0),('dit',0),('dit',49))
            for m in ('control','fused_stored','fused_recompute') for t in ('int32','scaled_fp16')]
        ranks=[dict(rank=i,status='completed_not_generation_acceptance',cases=rows) for i in (0,1)]
        videos=[dict(rank=i,status='completed_localization_not_identity_acceptance',stages=[{}]*74) for i in (0,1)]
        self.assertTrue(complete_matrix(ranks,rows,videos))
        self.assertFalse(complete_matrix(ranks,rows[:-1],videos))
        self.assertFalse(complete_matrix(ranks[:1],rows,videos))
        rows[0]['independent_codec_epilogue_exact']=False
        self.assertFalse(complete_matrix(ranks,rows,videos))
        rows[0]['independent_codec_epilogue_exact']=True
        videos[1]['stages'].pop()
        self.assertFalse(complete_matrix(ranks,rows,videos))


if __name__=='__main__':unittest.main()
