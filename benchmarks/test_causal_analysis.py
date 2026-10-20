"""Known dependency fixtures: overlapping contexts and critical CPU futures."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import analyze_causal_audit as analysis


class CausalityTests(unittest.TestCase):
    def test_cutlass_attention_is_not_gemm(self):
        self.assertEqual(analysis.category('fmha_cutlassF_f32_aligned_32x128_rf_sm75(AttentionKernel)', 'kernel'),'attention')
        self.assertEqual(analysis.category('cutlass::Kernel2<cutlass::gemm::kernel::DefaultGemmWithVisitor>', 'kernel'),'gemm')

    def test_interval_accounting(self):
        self.assertEqual(analysis.union([(0,10),(5,15),(30,40)]),[[0,15],[30,40]])
        self.assertEqual(analysis.overlap([(0,20_000_000)],[(10_000_000,30_000_000)]),10)

    def test_nested_interval_index(self):
        events=[dict(start_ns=a,end_ns=b) for a,b in [(0,100),(5,10),(30,40),(70,80),(100,110)]]
        index=analysis.IntervalIndex(events)
        for start,end in [(10,30),(35,75),(90,100),(100,101),(200,210)]:
            expected=[e for e in events if e['start_ns']<end and e['end_ns']>start]
            self.assertEqual(index.overlapping(start,end),expected)

    def test_parent_activity_and_launch_dependency(self):
        ms=1_000_000
        def event(name,a,b,rank,**fields):
            return dict(name=name,start_ns=a*ms,end_ns=b*ms,rank=rank,request=0,
                        stage='DiT',command=1,nfe=1,pid=12,tid=123,**fields)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            result=dict(passed=True,capture='kineto',case='fixture',host_monitor=dict(samples=[]),requests=[
                dict(request=0,start_ns=100*ms,end_ns=200*ms,latency_ms=100,nfe=1,handoff_gap_ms=None,
                     stages=dict(sampler=dict(start_ns=100*ms,end_ns=200*ms)))])
            (root/'result.json').write_text(json.dumps(result))
            for rank in ('parent',0,1):
                events=[event('request',100,200,rank)]
                kernels=[];runtime=[]
                if rank==0:
                    events += [event('cpu_future_wait',150,165,rank,block='b1',ready_on_entry=False),
                               event('block_compute_enqueue',165,180,rank,block='b1')]
                    kernels=[('gemm',100,110,0,0),('gemm',170,190,0,1)]
                    runtime=[dict(name='cudaLaunchKernel',cat='cuda_runtime',ph='X',
                                  ts=166000,dur=100,tid=123,args=dict(correlation=1))]
                elif rank=='parent':kernels=[('reference_conv',110,150,0,5)]
                else:kernels=[('gemm',105,135,1,2),('ncclAllReduce',150,160,1,3)]
                raw=[dict(name=analysis.label(e),cat='user_annotation',ph='X',
                          ts=e['start_ns']/1000,dur=(e['end_ns']-e['start_ns'])/1000,tid=123)
                     for e in events]
                raw += [dict(name=name,cat='kernel',ph='X',ts=a*1000,dur=(b-a)*1000,
                             args=dict(device=device,stream=7,correlation=correlation))
                        for name,a,b,device,correlation in kernels]
                (root/f'batch_r{rank}_kineto.json').write_text(json.dumps(dict(traceEvents=raw+runtime)))
                (root/f'batch_r{rank}_causal.json').write_text(json.dumps(dict(rank=rank,
                    events=events,gpu_intervals=[],alignment_uncertainty_ns=1)))
            report=analysis.analyze(root)
            self.assertEqual(report['requests'][0]['devices']['0']['busy_union_ms'],70)
            loading=[s for s in report['significant_stalls'] if s['device']==0 and s['start_ns']==150*ms]
            self.assertEqual(len(loading),1)
            self.assertEqual(loading[0]['confirmed_loading_dependencies'][0]['exposed_idle_intersection_ms'],15)
            self.assertFalse(any(s['device']==0 and s['start_ns']==110*ms for s in report['significant_stalls']))
            self.assertTrue((root/'timeline.html').exists())


if __name__=='__main__':unittest.main()
