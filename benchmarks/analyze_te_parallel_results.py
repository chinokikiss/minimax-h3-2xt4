"""Reproduce the retained, two-allocation TE matrix and CUDA trace summaries."""
import argparse
from collections import defaultdict
import gzip
import json
from pathlib import Path


def load(path):
    raw = path.read_bytes()
    if path.suffix == '.gz':raw = gzip.decompress(raw)
    return json.loads(raw)


def profile_summary(folder, rank):
    events = load(folder/f'attempt2_rank{rank}_tp2_trace.json.gz')['traceEvents']
    probe = load(folder/f'attempt2_rank{rank}_probe_trace.json.gz')['traceEvents']
    probe_kernels = [e for e in probe if e.get('cat') == 'kernel' and e.get('ph') == 'X']
    assert probe_kernels, 'Missing CUDA probe coverage'
    scopes = {}
    for label in ('TE/tp2_actual_collectives', 'TE/tp2_exact_collective_result_replay'):
        # Kineto emits GPU annotations with the same names. Count only the
        # CPU user scopes; selecting both annotations would double-count work.
        marks = [e for e in events if e.get('cat') == 'user_annotation' and e.get('ph') == 'X' and e.get('name') == label]
        assert len(marks) == 1
        scope = marks[0]
        kernels = [e for e in events if e.get('cat') == 'kernel' and e.get('ph') == 'X'
                   and scope['ts'] <= e['ts'] and e['ts']+e['dur'] <= scope['ts']+scope['dur']]
        families = defaultdict(lambda: dict(events=0, duration_sum_ms=0.))
        by_name = defaultdict(lambda: dict(events=0, duration_sum_ms=0.))
        for event in kernels:
            name = event['name']
            if 'nccl' in name.lower():family = 'nccl'
            elif 'direct_copy_kernel_cuda' in name and 'signed char' in name:family = 'int8_copy'
            elif 'igemm_int8' in name or 'i8816gemm_s8' in name:family = 'int8_gemm'
            else:family = 'other'
            for group in (families[family], by_name[name]):
                group['events'] += 1;group['duration_sum_ms'] += event['dur']/1000
        assert kernels
        scopes[label] = dict(cpu_scope_ms=scope['dur']/1000, kernel_events=len(kernels),
            kernel_duration_sum_ms=sum(e['dur'] for e in kernels)/1000, families=dict(families),
            top_kernel_names=[dict(name=name, **value) for name,value in sorted(by_name.items(),key=lambda p:-p[1]['duration_sum_ms'])[:8]])
    assert scopes['TE/tp2_actual_collectives']['families']['nccl']['events'] == 4
    assert 'nccl' not in scopes['TE/tp2_exact_collective_result_replay']['families']
    return dict(rank=rank, probe_kernel_events=len(probe_kernels), scopes=scopes)


def analyze(folder):
    attempts = {attempt:[load(folder/f'attempt{attempt}_rank{rank}.json') for rank in (0,1)] for attempt in (1,2)}
    for ranks in attempts.values():
        assert len({r['uuid'] for r in ranks}) == 2
        assert all(r['capability'] == [7,5] and 'T4' in r['device'] for r in ranks)
    assert all(r['status'] == 'complete' and len(r['cases']) == 18 for r in attempts[2])
    rows = []
    for length in (128,353,1306,4096):
        attempt = 1 if length == 128 else 2
        ranks = attempts[attempt]
        for backend in ('native','tilelang_group32','kitchen_causal'):
            single = [next(c for c in r['cases'] if c['tokens'] == length and c['backend'] == backend and c['topology'] == 'single_gpu') for r in ranks]
            tp = [next(c for c in r['cases'] if c['tokens'] == length and c['backend'] == backend and c['topology'] == 'tp2_nccl') for r in ranks]
            assert tp[0]['output_sha256'] == tp[1]['output_sha256']
            assert all(c['replay_byte_exact_to_nccl'] for c in tp)
            assert all(c['attention_vs_fp32']['finite'] and c['block_vs_native_fp16_flow']['finite'] for c in single+tp)
            assert all(c['block_vs_single_gpu_same_backend']['finite'] for c in tp)
            real = max(c['block']['median_wall_ms'] for c in tp)
            replay = max(c['collective_result_replay']['median_wall_ms'] for c in tp)
            operations = [c['diagnostic']['collectives'] for c in tp]
            assert all(len(v) == 4 and all(e['operation'] == 'all_reduce' for e in v) for v in operations)
            assert all(sum('MAX' in e['reduction'] for e in v) == 2 and sum('SUM' in e['reduction'] for e in v) == 2 for v in operations)
            for entries in operations:
                for entry in entries:
                    is_max = 'MAX' in entry['reduction']
                    assert entry['output_shapes'] == [[length, 1 if is_max else 5120]]
                    assert entry['output_dtypes'] == ['torch.float32' if is_max else 'torch.float16']
                    assert entry['operand_bytes'] == length * (4 if is_max else 5120 * 2)
            rows.append(dict(tokens=length, backend=backend, attempt=attempt, uuid=[r['uuid'] for r in ranks],
                single_gpu_ms=[c['block']['median_wall_ms'] for c in single], tp2_ms=real, replay_ms=replay,
                speedup_vs_single=[c['block']['median_wall_ms']/real for c in single],
                replay_delta_ms=real-replay, replay_delta_percent=100*(real-replay)/real,
                attention_single_gpu_ms=[c['attention']['median_wall_ms'] for c in single],
                attention_tp2_ms=[c['attention']['median_wall_ms'] for c in tp],
                attention_relative_l2_percent=[100*c['attention_vs_fp32']['relative_l2'] for c in tp],
                attention_max_abs=max(c['attention_vs_fp32']['max_abs'] for c in tp),
                tp2_vs_unsharded_block_relative_l2_percent=[100*c['block_vs_single_gpu_same_backend']['relative_l2'] for c in tp],
                collective_operand_bytes_per_rank=[sum(e['operand_bytes'] for e in v) for v in operations],
                collective_entry_skew_ms=[abs(a['start_ns']-b['start_ns'])/1e6 for a,b in zip(*operations)],
                peak_allocated_bytes_per_rank=[c['block']['peak_allocated_bytes'] for c in tp],
                samples=dict(single=[[s['wall_ms'] for s in c['block']['samples']] for c in single],
                    tp2=[[s['wall_ms'] for s in c['block']['samples']] for c in tp])))
    batch = load(folder/'attempt2_result.json')
    assert batch['status'] == 'complete' and batch['worker_exit_code'] == 0
    assert load(folder/'attempt2_execution_supervision.json')['exit_code'] == 0
    padding = [c[key] for rank in attempts[2] for c in rank['cases']
               for key in ('padding_boolean', 'padding_additive') if key in c]
    assert len(padding) == 24 and all(control['finite'] for control in padding)
    return dict(status='complete', scope='Resident unsharded/TP2 adapter block; not Native full-encoder latency',
        rows=rows, all_48_block_attention_cases_finite=True, replay_exact_all_12_tp2_configurations=True,
        finite_padding_controls=len(padding),
        tp2_faster_than_both_unsharded_controls=all(min(row['speedup_vs_single'])>1 for row in rows),
        profiles=[profile_summary(folder, rank) for rank in (0,1)], profiler_tools=batch['profiler_tools'],
        limits=['S128 and the other lengths use separate allocations; per-length controls remain paired.',
            'Single GPU uses the TP block adapter at world_size=1; it is not Native ComfyUI whole-layer/stage timing.',
            'world_size gates differ: TP2 enables fused quantization/transport unavailable to world_size=1.',
            'Backend order is fixed and short-shape NCCL/rank skew varies; block-level backend rankings are not isolated attention gains.',
            'Replay includes cached GPU copies/cache footprint and is not a pure wire-time or speedup upper bound.',
            'Profiled kernel sums include waiting/overlap and cannot be subtracted from unprofiled wall time.',
            'Seeded block0 inputs are not a real full-TE distribution or generation-quality test.',
            'Nsight Systems is absent; Nsight Compute exists but was not invoked. No hardware-counter stall attribution.'])


def plot(result, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names=('native','tilelang_group32','kitchen_causal');labels=('Native SDPA','Group32 Masked','Kitchen causal')
    colors=('#3379ae','#9a6bbc','#dc7f35')
    fig,axes=plt.subplots(1,3,figsize=(14,3.8),layout='constrained')
    for name,label,color in zip(names,labels,colors):
        rows=[r for r in result['rows'] if r['backend']==name];x=[r['tokens'] for r in rows]
        axes[0].plot(x,[min(r['speedup_vs_single']) for r in rows],marker='o',label=label,color=color)
        axes[1].plot(x,[max(r['attention_tp2_ms']) for r in rows],marker='o',color=color)
        axes[2].plot(x,[r['replay_delta_ms'] for r in rows],marker='o',color=color)
    axes[0].axhline(1,color='gray',linestyle='--',linewidth=1)
    axes[0].set(title='Block speedup vs faster unsharded control',ylabel='Speedup (x)')
    axes[1].set(title='Attention only: slower TP2 owner',ylabel='Median milliseconds',yscale='log')
    axes[2].set(title='Real TP2 minus exact-result replay',ylabel='Observed delta (ms), not pure wire time')
    for axis in axes:
        axis.set_xscale('log');axis.set_xticks([128,353,1306,4096],['128','353','1306','4096'])
        axis.set_xlabel('Tokens');axis.spines[['top','right']].set_visible(False)
    axes[0].legend(frameon=False,fontsize=8)
    fig.suptitle('Physical 2x T4 | trained TE block0 | fixed seeded inputs | S128 retained from allocation 1')
    fig.savefig(output,dpi=150);plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--evidence',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--plot',type=Path)
    args=parser.parse_args();result=analyze(args.evidence)
    args.output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    if args.plot:plot(result,args.plot)
    print(json.dumps(dict(status=result['status'],configurations=len(result['rows']),tp2_faster_than_both_controls=result['tp2_faster_than_both_unsharded_controls'])))
