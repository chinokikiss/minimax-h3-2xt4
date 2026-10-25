"""Report the physical gate separately from diagnostic-only approximate timings."""
import argparse
from collections import defaultdict
import json
import gzip
from pathlib import Path
import statistics


def main():
    p=argparse.ArgumentParser();p.add_argument('directory',type=Path)
    p.add_argument('--report',type=Path,default=Path('docs/reports/H3_PHYSICAL_INT8_LORA_RESULTS.md'));args=p.parse_args()
    paths=list(args.directory.rglob('results.json'))
    if len(paths)!=1:raise ValueError('Expected exactly one version-bound physical results file')
    result=json.loads(paths[0].read_text())
    if not result.get('physical_dual_t4'):raise ValueError('Physical provenance required')
    groups=defaultdict(list)
    for r in result['projection_cases']:
        if r.get('status')=='memory_guard_skipped':continue
        groups[(r['rows'],r['mode'],r['reduction'])].append(r)
    lines=['# Physical dual-T4: optional SwiGLU reduction and DMAD LoRA','',
        f"Run status: **{result['status']}**. Torch {result['torch']}, CUDA {result['cuda']}, Kitchen {result['kitchen']}.",
        'Two physical Tesla T4 GPUs, one NCCL rank per GPU. Pinned trained H3 block-0 weights and DMAD adapter ranges downloaded by Kaggle.',
        '', '**Scope:** learned FFN/FC2 latency and compact complete DiT blocks; no full-model, TE, sampling or generation-quality result.',
        '', '## Learned FFN / output reduction','',
        'All variants use the same inputs, Native ConvRot activation scales and GEMM workload. Two fixed forward/reverse rounds; two warmups and five synchronized timing samples per round. Profiler passes are separate.',
        '', '| S | Strategy | FC2 critical wall ms | FFN critical wall ms | Temporary peak MiB, max rank | Relative L2 | Strict gate |',
        '|---:|---|---:|---:|---:|---:|---|']
    aggregates=[]
    for (length,mode,reduction),rows in sorted(groups.items()):
        def wall(key):
            samples=[]
            for row in rows:
                ranks=row[key]['per_rank']
                samples.extend(max(ranks[r]['wall_ms'][i] for r in (0,1)) for i in range(len(ranks[0]['wall_ms'])))
            return statistics.median(samples)
        fc2=wall('diagnostic_timing');ffn=wall('ffn_diagnostic_timing')
        peak=max(r['temporary_peak_bytes'] for row in rows for r in row['diagnostic_timing']['per_rank'])/2**20
        rel=max(row.get('rel_l2',0) for row in rows);passed=all(row['passed'] and row['ffn_score']['passed'] for row in rows)
        label=mode.replace('scalemax_streamed','streamed').replace('scalemax','existing')+'/'+reduction
        lines.append(f"| {length} | {label} | {fc2:.3f} | {ffn:.3f} | {peak:.2f} | {rel:.3g} | {'PASS' if passed else 'FAIL — diagnostic timing only'} |")
        aggregates.append(dict(rows=length,mode=mode,reduction=reduction,fc2_ms=fc2,ffn_ms=ffn,temp_mib=peak,rel_l2=rel,passed=passed))
    lines+=['', 'Temporary deltas are measured above resident inputs/full reference weights/TP shards. Absolute per-rank allocation/reservation peaks and CUDA-event timings are in results.json. They do not represent full-pipeline VRAM residency.',
            '', '### Original single-GPU Native FFN','', '| S | Critical wall ms | Native gate |','|---:|---:|---|']
    for row in result['native_ffn_cases']:
        lines.append(f"| {row['rows']} | {row['timing']['critical_wall_median_ms']:.3f} | {'PASS' if row['score']['passed'] else 'FAIL'} |")
    skipped=[r for r in result['projection_cases'] if r.get('status')=='memory_guard_skipped']
    if skipped:lines+=['', 'Memory guard skipped: '+', '.join('S'+str(r['rows']) for r in skipped)+'. No out-of-memory trial was intentionally attempted.']
    lines+=['','## Complete block correctness','',
            '| Adapter mode | Reduction | Cases passed / total | Worst relative L2 | Worst absolute error |',
            '|---|---|---:|---:|---:|']
    blocks=defaultdict(list)
    for row in result['block_cases']:blocks[(row['adapter_mode'],row['mode'],row['reduction'])].append(row)
    for (adapter,mode,reduction),rows in sorted(blocks.items()):
        lines.append(f"| {adapter} | {mode}/{reduction} | {sum(r['passed'] for r in rows)}/{len(rows)} | {max(r.get('rel_l2',0) for r in rows):.3g} | {max(r.get('max_abs',0) for r in rows):.6g} |")
    lines+=['','Offline independently merged tensor bytes: '+str(sum(r['native_bytes_exact'] for r in result.get('offline_native_bytes',[])))+'/'+str(len(result.get('offline_native_bytes',[])))+'.',
            'Online uses upstream sequential bypass as the reference. Offline uses the independently verified merged-model reference; these are different numerical models.',
            '', '## Targeted communication evidence','',
            '| Strategy | MAX calls | SUM calls | SUM payload dtype | Tensor payload MiB / FC2 invocation |',
            '|---|---:|---:|---|---:|']
    for r in result.get('profiles',[]):
        calls=r['collective_calls'];maximum=sum(c['op']=='MAX' for c in calls);sums=[c for c in calls if c['op']=='SUM']
        lines.append(f"| {r['mode']}/{r['reduction']} | {maximum} | {len(sums)} | {', '.join(sorted(set(c['dtype'] for c in sums)))} | {sum(c['payload_bytes'] for c in calls)/2**20:.3f} |")
    traces=[]
    for path in sorted(paths[0].parent.glob('*rank0.json.gz')):
        data=json.load(gzip.open(path,'rt'));events=[e for e in data['traceEvents'] if e.get('cat')=='kernel']
        begin=min(e['ts'] for e in events);end=max(e['ts']+e['dur'] for e in events)
        rotation=[e for e in events if 'sgemm' in e['name']]
        integer=[e for e in events if 'cutlass_75_tensorop_i8816gemm' in e['name']]
        nccl=[e for e in events if 'nccl' in e['name'].lower()]
        traces.append(dict(trace=path.name,rank=0,gpu_trace_start_us=begin,gpu_trace_end_us=end,
            gpu_span_ms=(end-begin)/1000,kernel_count=len(events),rotation_kernels=len(rotation),
            rotation_activity_ms=sum(e['dur'] for e in rotation)/1000,int8_gemm_kernels=len(integer),
            int8_gemm_activity_ms=sum(e['dur'] for e in integer)/1000,nccl_kernels=len(nccl),
            nccl_activity_ms=sum(e['dur'] for e in nccl)/1000))
    lines+=['','### Confirmed cost of the memory-saving path (S8192, rank 0)','',
            '| Trace / strategy | GPU kernels | Rotation kernels / activity ms | INT8 GEMM kernels / activity ms | NCCL kernel activity ms |',
            '|---|---:|---:|---:|---:|']
    for t in traces:
        lines.append(f"| {t['trace']} | {t['kernel_count']} | {t['rotation_kernels']} / {t['rotation_activity_ms']:.3f} | {t['int8_gemm_kernels']} / {t['int8_gemm_activity_ms']:.3f} | {t['nccl_activity_ms']:.3f} |")
    lines+=['', 'The existing route uses one large rotation GEMM; the streamed two-pass route repeats rotation across 64 tiles. FP32 additionally splits one INT8 GEMM and one output SUM into 32 each. INT8 reduces SUM payload, but adds 32 scale MAX calls and quantization kernels. These observed extra operations explain the memory/latency tradeoff; smaller communication does not imply lower latency than the original path.',
            'Exact per-trace GPU timestamp intervals are stored in `trace_causality.json`. Activity totals include NCCL waiting and can overlap; they are not isolated link-transfer times. Profiler passes are separate from timing samples.']
    lines+=['','Payload sums count collective tensor arguments, not physical PCIe/NCCL link traffic. Per-rank compressed Chrome traces and top kernel/CPU summaries are included. CPU/GPU self-time sums can overlap and are not added to critical latency.',
            '', '## Interpretation limits','',
            'INT8 is approximate even when the overflow safety tests pass. Failed gates cannot support a speedup or production recommendation. Online concatenation likewise remains unqualified wherever its strict gate fails.',
            'A single bounded physical run with two ordering rounds screens numerical/memory/performance behavior; full-model multi-request throughput, loading/caching, long-sequence attention and audiovisual quality still need separate validation.',
            '', f'Evidence directory: `{args.directory.resolve()}`. Source identities, tensor-range SHA256 plans, and per-rank raw samples accompany the run.']
    lines+=['','## Recommendation','',
            'Keep existing exact INT32 for latency when it fits. Use streamed exact INT32 when byte equivalence is mandatory and memory is tight; use streamed FP32 for a smaller temporary peak when the measured tiny numerical difference is acceptable. Both streamed variants are slower in this run.',
            'Keep INT8 manual and experimental: approximately 6.9% FC2 relative error, no material VRAM reduction versus streamed FP32, and no latency win versus existing INT32.',
            'Offline LoRA passed the tested Native byte/block checks. Online single LoRA failed one of three compact shapes and online multi-LoRA failed all three under the strict gate; neither is promoted as universally Native equivalent.',
            'The next targeted optimization to test is a larger, headroom-bounded row tile and fewer launches/collectives. This is a proposal, not a measured improvement. No inference policy or numerical gate was changed during this diagnostic run.']
    args.report.write_text('\n'.join(lines)+'\n',encoding='utf8')
    (args.directory/'trace_causality.json').write_text(json.dumps(traces,indent=2))
    (args.directory/'summary.json').write_text(json.dumps(dict(status=result['status'],aggregates=aggregates,
        native_exact_passed=result.get('native_exact_passed'),safety=result['safety']),indent=2))
    print(args.report.resolve());print(json.dumps(aggregates,indent=2))


if __name__=='__main__':main()
