import unittest
from benchmarks.analyze_dit_memory_lifetimes import lifetimes


def fixture(events, shapes=()):
    return dict(initial_segments=[],events=events,markers=[],shape_observations=shapes,
                stacks=[[]],static_tables=[],history_limit_reached=False)


class Lifetimes(unittest.TestCase):
    def test_alias_views_do_not_multiply_allocated_bytes(self):
        p=fixture([dict(action='alloc',addr=7,size=1024,time_us=1,stack=0)],
            [dict(addr=7,time_ns=1500,shape=s,dtype='float16',operator='view') for s in ([512],[8,64])])
        r=lifetimes(p)
        self.assertEqual(r['peak']['allocated_bytes'],1024)
        self.assertEqual(len(r['epochs']),1)
        self.assertEqual(len(r['epochs'][0]['aliases']),2)

    def test_host_release_and_cuda_completion_are_separate(self):
        p=fixture([dict(action='alloc',addr=7,size=1024,time_us=1,stack=0),
                   dict(action='free_requested',addr=7,size=1024,time_us=2),
                   dict(action='free_completed',addr=7,size=1024,time_us=5)])
        r=lifetimes(p)
        self.assertEqual(r['epochs'][0]['host_release_ns'],2000)
        self.assertEqual(r['epochs'][0]['cuda_reusable_ns'],5000)
        self.assertEqual(r['points'][1]['allocated_bytes'],0)
        self.assertEqual(r['points'][1]['pending_bytes'],1024)
        self.assertEqual(r['unknown_free_events'],0)

    def test_address_reuse_creates_a_new_lifetime(self):
        p=fixture([dict(action='alloc',addr=7,size=1024,time_us=1,stack=0),
                   dict(action='free_requested',addr=7,size=1024,time_us=2),
                   dict(action='free_completed',addr=7,size=1024,time_us=3),
                   dict(action='alloc',addr=7,size=4096,time_us=4,stack=0)])
        r=lifetimes(p)
        self.assertEqual([e['bytes'] for e in r['epochs']],[1024,4096])
        self.assertEqual(r['peak']['allocated_bytes'],4096)

    def test_failed_request_is_not_added_to_live_allocated_bytes(self):
        p=fixture([dict(action='segment_alloc',addr=4096,size=8192,time_us=1,stream=0),
                   dict(action='alloc',addr=4096,size=4096,time_us=2,stack=0),
                   dict(action='oom',size=8192,time_us=3,device_free=2048)])
        r=lifetimes(p)
        self.assertEqual(r['peak']['allocated_bytes'],4096)
        self.assertEqual(r['ooms'][0]['allocated_bytes'],4096)
        self.assertEqual(r['ooms'][0]['requested_bytes'],8192)
        self.assertEqual(r['ooms'][0]['largest_inactive_interval_bytes'],4096)

    def test_peak_uses_residency_state_at_that_time(self):
        p=fixture([dict(action='alloc',addr=7,size=1024,time_us=1,stack=0),
                   dict(action='alloc',addr=8,size=4096,time_us=3,stack=0),
                   dict(action='free_requested',addr=8,size=4096,time_us=4),
                   dict(action='free_completed',addr=8,size=4096,time_us=5)])
        p['static_tables']=[[dict(addr=7,category='resident_weights',key='refiner')],
                            [dict(addr=7,category='retired_weights',key='pending')]]
        p['markers']=[dict(time_ns=2000,label='resident',static_table=0),
                      dict(time_ns=6000,label='retired',static_table=1)]
        r=lifetimes(p)
        self.assertEqual(r['peak']['category_bytes']['resident_weights'],1024)
        self.assertNotIn('retired_weights',r['peak']['category_bytes'])


if __name__=='__main__':unittest.main()
