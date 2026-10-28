import unittest
from benchmarks.analyze_vae_cuda13 import attribute,union_us


class TraceAttribution(unittest.TestCase):
    def test_annotations_are_not_kernels_and_extensions_use_launch_correlation(self):
        events=[dict(cat='user_annotation',name='VAE_MEASURED_BLOCK',pid=123,tid=9,ts=0,dur=100),
            dict(cat='user_annotation',name='VAE_LINEAR_SITE/fc1',pid=123,tid=9,ts=10,dur=20),
            dict(cat='cuda_runtime',name='cudaLaunchKernel',pid=123,tid=9,ts=15,dur=1,args=dict(correlation=77)),
            # No External id: extension kernel still belongs to FC1.
            dict(cat='kernel',name='gemm',pid=0,tid=7,ts=40,dur=5,args=dict(correlation=77)),
            dict(cat='gpu_user_annotation',name='VAE_LINEAR_SITE/fc1',pid=0,tid=7,ts=40,dur=5)]
        report=attribute(events)
        self.assertTrue(report['complete']);self.assertEqual(report['kernel_count'],1)
        self.assertEqual(report['excluded_gpu_annotations'],1)
        self.assertEqual(report['phases']['VAE_LINEAR_SITE/fc1']['kernel_work_ms'],.005)
        self.assertEqual(report['kernel_work_ms'],.005)

    def test_missing_launch_is_not_complete(self):
        report=attribute([dict(cat='kernel',name='unmatched',ts=1,dur=3,args=dict(correlation=99))])
        self.assertFalse(report['complete']);self.assertEqual(report['unmatched_kernel_count'],1)
        self.assertEqual(union_us([(1,4),(2,6),(8,10)]),7)


if __name__=='__main__':unittest.main()
