"""Do not call overlapped Future waits GPU stalls or double-count ranks."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import analyze_te_causal_profile as analysis


class Exposure(unittest.TestCase):
    def test_parent_work_and_other_rank_reduce_exposure_and_ranks_are_unioned(self):
        scale=10**6
        def event(name,start,end,**fields):
            return dict(name=name,start_ns=start*scale,end_ns=end*scale,request=1,**fields)
        def load(folder,rank):
            if rank=='parent':
                return dict(pid=99,events=[],gpu_intervals=[],kineto_clock_fit={'anchors':1}),[
                    dict(start_ns=0,end_ns=20*scale,device=0,rank='parent',category='gemm'),
                    dict(start_ns=80*scale,end_ns=100*scale,device=0,rank='parent',category='gemm')],[]
            key=str(('checkpoint','te',2,rank));pid=100+rank
            trace=dict(pid=pid,gpu_intervals=[],kineto_clock_fit={'anchors':1},events=[
                event('future_consumer_wait',20,80,tid=pid,block=key,future='ready',ready_on_entry=False),
                event('future_ready_bounds',79,79,future='ready',lower_ns=78*scale,upper_ns=79*scale),
                event('h2d_enqueue',81,82,tid=pid,block=key)])
            gpu=[] if rank==0 else [dict(start_ns=0,end_ns=30*scale,device=1,rank=1,category='gemm'),
                dict(start_ns=70*scale,end_ns=100*scale,device=1,rank=1,category='gemm')]
            return trace,gpu,[]
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);(root/'result.json').write_text(json.dumps(dict(passed=True,physical=True,
                requests=[dict(request=1,start_ns=0,stages=dict(reference_and_TE=dict(start_ns=0,end_ns=100*scale)))])))
            with patch.object(analysis,'load_trace',load):result=analysis.analyze(root)
            self.assertEqual(result['linked_weight_unavailable_both_idle_union_ms'],40)
            self.assertEqual(sum(r['linked_both_gpu_inactive_ms'] for r in result['blocks']),80)
            self.assertEqual(len(result['blocks']),96)


if __name__=='__main__':unittest.main()
