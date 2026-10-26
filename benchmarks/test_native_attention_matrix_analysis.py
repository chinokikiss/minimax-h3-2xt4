import unittest
from benchmarks.analyze_native_attention_matrix import summarize


def fixture():
    ranks=[]
    for rank in (0,1):
        candidates=[]
        for name,base in (('kitchen_int8',10),('tilelang_group32_dense',20),('native_drop',5)):
            for round_id in (0,1):
                candidates.append(dict(name=name,round=round_id,status='passed',
                    event_median_ms=base+rank+round_id,wall_median_ms=base+rank+round_id+1,
                    peak_allocated=1024,peak_temporary=512))
        ranks.append(dict(rank=rank,cases=[dict(shape=dict(name='test'),candidates=candidates)]))
    return dict(status='completed',ranks=ranks,native_status='qualified')


class Analysis(unittest.TestCase):
    def test_slower_rank_keeps_matched_round_ratios(self):
        result=summarize(fixture())['shapes']['test']['backends']
        self.assertEqual(result['kitchen_int8']['slower_rank_event_ms'],11.5)
        self.assertEqual(result['native_drop']['slower_rank_event_ms'],6.5)
        ratio=result['native_drop']['paired_ratios']['kitchen_int8']
        self.assertAlmostEqual(ratio['min'],12/7)
        self.assertEqual(ratio['max'],2)

    def test_missing_round_withholds_speed_claim(self):
        batch=fixture();batch['ranks'][1]['cases'][0]['candidates'].pop()
        result=summarize(batch)['shapes']['test']['backends']
        self.assertFalse(result['native_drop']['complete_two_rank_two_round'])
        self.assertNotIn('slower_rank_event_ms',result['native_drop'])

    def test_missing_rank_withholds_speed_claim(self):
        batch=fixture();batch['ranks'].pop()
        result=summarize(batch)['shapes']['test']['backends']
        self.assertFalse(result['native_drop']['complete_two_rank_two_round'])


if __name__=='__main__':unittest.main()
