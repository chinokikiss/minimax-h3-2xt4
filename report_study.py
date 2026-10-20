"""Produce an evidence-linked report from measured rows and canonical traces."""
import json
from pathlib import Path
import statistics

ROOT=Path(__file__).resolve().parent


def table(rows,columns):
    return ['| '+' | '.join(n for n,f in columns)+' |','| '+' | '.join('---' for _ in columns)+' |']+['| '+' | '.join(str(f(r)) for n,f in columns)+' |' for r in rows]


def main():
    phase='optimized' if (ROOT/'kaggle_study_output/optimized/analysis.json').exists() else 'baseline'
    folder=ROOT/'kaggle_study_output'/phase
    data=json.loads((folder/'analysis.json').read_text())
    manifest=json.loads((folder/'study_results/manifest.json').read_text())
    full_folder=ROOT/'kaggle_study_output/full_completion'
    if (full_folder/'analysis.json').exists():
        full_data=json.loads((full_folder/'analysis.json').read_text())
        full_manifest=json.loads((full_folder/'study_results/manifest.json').read_text())
        for kind in ['te','dit']:data['tables']['full_'+kind]=full_data['tables']['full_'+kind]
    else:full_manifest=manifest
    final=full_manifest['status']=='complete' and all({r['mode'] for r in data['tables'].get('full_'+kind,[])} >= {'native','tp_optimized','pp_overlap'} for kind in ['te','dit'])
    report=['# MiniMax-H3 TE / DiT: actual dual-T4 parallel study','',
        'Status: '+('completed' if final else 'optimized block tests complete; full-model comparisons pending')+'. SP and VAE are excluded. Performance evidence comes exclusively from physical Tesla T4 pairs; local 5060 Ti executions establish correctness only.','',
        '[Kaggle run](https://www.kaggle.com/code/chinokiki9/minimax-h3-te-dit-parallel-study). '+
        f'[Measured JSON](kaggle_study_output/{phase}/analysis.json), [trace index](kaggle_study_output/{phase}/TRACE_INDEX.md), [frozen executed source](kaggle_study_output/{phase}/study_source.tar.gz).','',
        '## Why legacy TP lost','',
        'The baseline trace demonstrates communication and layout overhead, rather than a failure to split attention. TE S512 attention falls from 0.689 to 0.364 ms, but TP adds 5.382 ms of AllGather kernels, 5.046 ms of AllReduce kernels and 3.705 ms of layout/copy kernels. DiT S1024 attention falls from 6.853 to 3.477 ms; its AllGather and AllReduce kernels sum to 20.373 and 10.757 ms. These are durations from one profiled rank, not components to subtract from unprofiled wall latency.','',
        'Each row projection gathers lengths, synchronizes four scalar reads per block, pads/copies activations, gathers full feature rows, repeats full-row quantization on both GPUs, transposes weights for `torch._int_mm`, and reduces INT32 accumulators. Native uses different optimized Turing GEMM kernels. At TE S2048 the combined profiled INT8 GEMM time is actually higher in TP (31.123 vs 28.346 ms), despite sharding.','',
        'NCCL initialization logs show `NCCL_P2P_DISABLE=1` and `SHM/direct/direct` channels. Peer eligibility is true, but the baseline explicitly disabled NCCL peer transport. This explains the observed transport; it does not establish that every T4 pair lacks P2P.','',
        '## Targeted changes','',
        '| Change | Purpose and semantic constraint |','|---|---|',
        '| Fixed-size gathers | Remove the length collective, `.item()` synchronization and zero-padding for equal feature shards. |',
        '| Native CUDA INT32 GEMM | Remove weight transposition and padded activation copies. Preserve integer reduction before native dequantization. |',
        '| Global row maximum | Replace full activation AllGather with one FP32 row-maximum AllReduce on native fallback rotation routes. Use native row quantization and discard sentinel columns before GEMM. |',
        '| FP32 rotation geometry | Preserve the full Native rotation GEMM geometry. Local-only FP32 rotation changed accumulation order and failed correctness; zero unowned groups instead. This retains some redundant rotation work. |',
        '| P2P transport experiment | Change the NCCL transport flag independently of the arithmetic; accept only native-equivalent outputs and inspect actual channel logs. |',
        '| Balanced PP scheduling | Use native layers and complete native entry points, two stage streams, independent microrequests, CUDA Events, measured split costs and stream-safe tensor ownership. |','',
        'Two local defects were fixed before submission: PP tensors/metadata needed ownership recorded for each consuming stream; FP32 partial rotation required Native GEMM geometry. Full 50-layer TE/DiT TP and eight PP microrequests repeated three times passed bit-for-bit at S512 in the local two-rank simulation. PP simulation used two distinct trained layers with the complete embedding/refiner/packing/head entry points; physical full-50 PP has its own pre/post Native gates.','',
        '## Block measurements','',
        'Trained checkpoint block 0. TE uses an actual batch dimension; DiT batch 2 means two independent sequential requests because native DiT supports batch 1. No block result estimates full-model latency. PP partitions layers and therefore has no single-block entry in this comparison.','']
    for kind in ['te','dit']:
        report+=['### '+kind.upper(),'']
        rows=sorted(data['tables'][kind],key=lambda r:(r['length'],r['batch'],r['mode']))
        report+=table(rows,[('Strategy',lambda r:r['mode']),('S',lambda r:r['length']),('B/requests',lambda r:r['batch']),('Latency ms',lambda r:f"{r['median_ms']:.3f}"),('Requests/s',lambda r:f"{r['requests_per_second']:.2f}"),('Peak allocated GiB/GPU',lambda r:'/'.join(f'{v/2**30:.3f}' for v in r['peak_allocated_bytes'])),('Correctness',lambda r:'bit-exact' if r['post_accuracy']['is_bit_exact'] else 'gate passed')]);report.append('')
    if phase=='optimized':
        report += ['The optimized row-maximum variant reduces TE S2048 block latency from 100.161 to 51.284 ms and DiT S4096 from 310.376 to 194.996 ms versus legacy TP in the same allocation. These are block improvements, not full-model speedups. TE S512 Native samples shift from about 23 to 17 ms within the sample window; its 22.960 ms median is not a stable baseline for claiming an advantage over optimized TP. Baseline allocation Native S512 was 16.971 ms.','',
                   'The P2P-enabled experiment still initialized SHM/direct/direct channels. Its near-equal timings do not demonstrate direct peer transport. The selected full-model variant is tp_scalemax for both models.','',
                   'Both Native full-model processes reached their 350-second orchestration limit after M4, with no correctness exception. Their completed measurements and traces are preserved below; TP/PP were skipped. The following completion allocation was canceled by the user during setup. A replacement uses the prescribed TE S512/2048/4096 and DiT S8192/16384/32768 matrix with native image/video reference packing; these historical measurements are not results for that matrix.','']
    for kind in ['te','dit']:
        rows=data['tables'].get('full_'+kind,[])
        if not rows:continue
        report+=['## Full '+kind.upper(),'']
        report+=table(rows,[('Strategy',lambda r:r['mode']),('S',lambda r:r['length']),('Requests',lambda r:r['batch']),('Purpose / execution',lambda r:r['purpose']+' / '+r['batch_execution']),('Wall ms',lambda r:f"{r['median_ms']:.3f}"),('Requests/s',lambda r:f"{r['requests_per_second']:.2f}"),('Peak GiB/GPU',lambda r:'/'.join(f'{v/2**30:.3f}' for v in r['peak_allocated_bytes'])),('Correctness',lambda r:'bit-exact' if r['post_accuracy']['is_bit_exact'] else 'gate passed')]);report.append('')
        lat=[r for r in rows if r['purpose']=='latency' and r['length']==512 and r['batch']==1]
        throughput=[r for r in rows if r['purpose']=='throughput' and r['batch']==8]
        if lat and {r['mode'] for r in lat} >= {'native','tp_optimized','pp_overlap'}:
            best=min(lat,key=lambda r:r['median_ms'])
            report += [f"For S512 single-request latency, the measured winner is **{best['mode']}** ({best['median_ms']:.3f} ms).",'']
        if throughput and {r['mode'] for r in throughput} >= {'native','tp_optimized','pp_overlap'}:
            best=min(throughput,key=lambda r:r['median_ms'])
            report += [f"For eight independent S512 requests, the measured winner is **{best['mode']}** ({best['requests_per_second']:.2f} requests/s, complete cohort {best['median_ms']:.3f} ms).",'']
        for r in throughput:
            if not r['pipeline_events']:continue
            s=r['pipeline_events']
            cadence=[]
            for p in s:
                # Events are stored in input order; recover chronological GPU
                # completion intervals even if CPU lock acquisition reordered tasks.
                ordered=sorted(p['completion_offsets_ms']);cadence.extend(b-a for a,b in zip(ordered,ordered[1:]))
            report += [f"PP M8 CUDA-event median stage costs: stage 0 {statistics.median(v for p in s for v in p['stage0_ms']):.3f} ms; stage 1 including handoff and heads {statistics.median(v for p in s for v in p['stage1_including_transfer_and_heads_ms']):.3f} ms. Observed median completion interval {statistics.median(cadence):.3f} ms. Reported cohort wall time includes fill/drain, bubbles, scheduling and return stacking. The ideal 1/(M+1) bubble fraction is not used as measured performance.",'']
    report+=['## Timing, memory and limits','',
        'Block timings use 3 warmups and 9 synchronized samples. Full latency uses 1 warmup and 5 samples; throughput uses 1 warmup and 3 complete cohorts. Distributed wall latency is the maximum rank, with barriers outside the timer; CUDA Events are archived separately. Profiling is a separate diagnostic pass. Inputs and Native references are identical for each accepted comparison; strict pre/post gates require relative L2 ≤ 1e-5, cosine ≥ 0.999999 and max absolute error ≤ 0.005. Both ranks agree before timing.','',
        'Native block timing includes the input clone, with masks/RoPE precomputed equally for all strategies. Full timing includes TE embedding/50 distinct layers and the complete DiT refiner, modality packing, curve interpolation, 50 distinct blocks and both output heads. Outputs are GPU resident; CPU copying, checkpoint loading and sampler/VAE work are excluded. Full weights require two-GPU Native placement; the streamed single-GPU reference validates placement and is never treated as resident latency.','',
        'Throughput cohorts execute the same M independent batch-1 requests for Native, TP and PP. TE true-batch-2 latency is shown separately from PP microbatch-1 execution and is not used as its speedup baseline. Full TE is text-only prefill, without vision/KV decoding; full DiT uses synthetic text conditioning and one audiovisual denoising step without reference conditions or denoise masks. The results do not validate arbitrary multimodal payloads, generation quality or end-to-end sampling.','',
        'Allocated/reserved peaks, resident allocation and peak temporary increments are in the measured JSON. Kernel duration sums, memcpy times, launches and per-device kernel-free gaps are in the trace analysis. Those gaps are activity-timeline evidence, not SM occupancy; NCCL busy kernels can include waiting. Bootstrap intervals describe within-session samples, not independent allocation uncertainty.','']
    if not final:report+=['The final TE/DiT recommendations remain pending physical optimized/full-model results. Baseline evidence supports replacing the measured overhead, not claiming an optimization speedup.','']
    (ROOT/'docs/reports/TE_DIT_STUDY_REPORT.md').write_text('\n'.join(report),encoding='utf-8')
    print(ROOT/'docs/reports/TE_DIT_STUDY_REPORT.md')


if __name__=='__main__':main()
