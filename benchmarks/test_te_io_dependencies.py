"""Verify dependency selection and overlap unions, including competing producers."""
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools.experiments.analyze_te_io_dependencies import dependencies


def read(a,b):
    return dict(start_ns=a,end_ns=b,tid=2,block='block',source='file',offset=0,
        requested_bytes=10,returned_bytes=10,wall_ns=b-a,cpu_ns=1,
        counters=dict(io={},usage={}),samples=dict(states={},wchans={}),
        resident_before=dict(pages=1,resident_pages=0),resident_after=dict(pages=1,resident_pages=1),
        observer_outside_read_ns=0)


class DependencyTests(unittest.TestCase):
    def test_publication_and_thread_select_one_producer(self):
        events=[dict(name='future_consumer_wait',tid=1,start_ns=10,end_ns=30,ready_on_entry=False,
                     future='f',block='block'),
                dict(name='cpu_weight_prepare',tid=2,start_ns=5,end_ns=28,block='block'),
                dict(name='future_ready_bounds',tid=2,start_ns=28,end_ns=29,lower_ns=28,upper_ns=29,future='f'),
                dict(name='h2d_enqueue',tid=1,start_ns=32,end_ns=33)]
        result=dependencies(events,[read(6,12),read(15,22)],1,0,40)
        self.assertAlmostEqual(result['producer_read_wall_overlap_union_s'],9e-9)
        self.assertEqual(result['waits'][0]['next_same_thread_h2d_enqueue']['start_ns'],32)
        # The same block can be prepared speculatively twice. Never silently
        # attribute one of two concurrent candidate producers as proven.
        events.append(dict(name='cpu_weight_prepare',tid=3,start_ns=5,end_ns=28,block='block'))
        result=dependencies(events,[read(6,12)],1,0,40)
        self.assertEqual(result['waits'][0]['read_association'],'unresolved')
        self.assertEqual(result['producer_read_wall_overlap_union_s'],0)


if __name__=='__main__':unittest.main()
