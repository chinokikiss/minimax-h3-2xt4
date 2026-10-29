"""Independently check the saved T4 P2 evidence; no Kaggle/network/GPU calls."""
import argparse
from collections import defaultdict
import gzip
import hashlib
import json
from pathlib import Path
from statistics import mean


def read(path):
    raw = path.read_bytes()
    return json.loads(gzip.decompress(raw) if path.suffix == '.gz' else raw)


def choose(root, downloaded, archived):
    path = root / downloaded
    return path if path.exists() else root / archived


def totals(rows):
    return dict(count=len(rows), wall_seconds=sum(r['wall_seconds'] for r in rows),
                thread_cpu_seconds=sum(r['thread_cpu_seconds'] for r in rows))


def analyze(root, submission):
    result_path = choose(root, 'ptd_qualification/result.json', 'ptd_result.json.gz')
    result = read(result_path)
    binding = read(choose(root, 'light_setup/delivery_binding.json', 'delivery_binding.json'))
    supervision = read(choose(root, 'ptd_supervision/supervision.json', 'supervision.json'))
    source = submission['source_binding']['publication']
    assert binding['capsule_sha256'] == result['capsule_sha256'] == source['capsule_sha256']
    assert result['status'] == 'complete' and result['handoff_outputs_byte_exact']
    assert supervision['status'] == 'worker_exited' and supervision['exit_code'] == 0
    assert result['contract_tests'] == 48 and result['runtime_layout_bytes'] == 0
    assert result['verified_targets_per_hop'] == [16] * 8 and result['deployment_fixture_rejected']
    hardware = result['hardware']
    assert len(hardware['devices']) == 2 and len({d['uuid'] for d in hardware['devices']}) == 2
    assert all('T4' in d['name'] and d['capability'] == [7, 5] for d in hardware['devices'])
    assert hardware['kitchen'] == '0.2.37' and hardware['tilelang'] == '0.1.13' and hardware['cuda'] == '12.8'
    assert result['full_pipeline'] is False and result['no_local_lora_download']
    assert result['auxiliary_comparison']['native_tensor_proofs_equal']
    assert not result['auxiliary_comparison']['lora_remerged']
    assert not result['auxiliary_comparison']['fp32_sidecar']
    cases = result['smoke_cases']
    assert [c['case'] for c in cases] == ['monolithic', 'split', 'split_repeat', 'monolithic_repeat']
    rows = []
    worker_reports_checked = 0
    for index, case in enumerate(cases):
        split = case['case'].startswith('split')
        assert case['status'] == 'complete' and case['finite'] and case['hops'] == list(range(8))
        assert case['cpu_options'] == dict(future_cache=True, prefetch_refiner=True)
        assert case['real_trained_layers'] == 2 and not case['full_pipeline'] and not case['quality_qualified']
        assert case['hop_output_sha256'] == cases[0]['hop_output_sha256']
        assert case['final_sha256'] == cases[0]['final_sha256']
        assert len(case['hop_output_sha256']) == 8 and all(len(h) == 2 for h in case['hop_output_sha256'])
        assert len({s['window_id'] for s in case['queue_states']}) == 1
        assert len({s['shell_id'] for s in case['queue_states']}) == 1
        ranks = []
        for rank, state in enumerate(case['restored_stats']):
            assert state['rank'] == rank
            probe = state['ptd_probe']
            relative = f'{index}_{case["case"]}/worker_rank{rank}.json'
            worker = read(choose(root, 'ptd_qualification/' + relative, relative))
            for key in ('rank', 'spans', 'cpu_preparation_spans', 'auxiliary_consumption',
                        'peak_allocated', 'peak_reserved'):
                assert worker[key] == probe[key], ('Independent worker receipt mismatch', relative, key)
            worker_reports_checked += 1
            assert len(probe['spans']) == 16 and all(s['default_stream_interval_ms'] is not None for s in probe['spans'])
            cache = state['host_cache']
            assert cache['cache_bytes'] == cache['pending'] == cache['protected_groups'] == 0
            assert cache['shared_budget']['reserved_bytes'] == cache['shared_budget']['pending_preparation_bytes'] == 0
            assert cache['shared_budget']['preparing'] == 0
            assert all(n == 1 for n in cache['preparation_counts'].values())
            stages = defaultdict(list)
            for span in probe['cpu_preparation_spans']: stages[span['stage']].append(span)
            accesses = probe['auxiliary_consumption']
            roles = {}
            for role in ('refiner', 'video_head', 'audio_head'):
                values = [r for r in accesses if r['role'] == role]
                assert all(r['payload_verified'] for r in values)
                roles[role] = dict(totals(values), payload_bytes=sum(r['payload_bytes'] for r in values),
                    header_bytes=sum(r['header_bytes'] for r in values))
            if split:
                assert not stages.get('raw_overlay_verify')
                assert roles['refiner']['count'] == (64 if rank == 0 else 0)
                assert roles['video_head']['count'] == roles['audio_head']['count'] == 18
            else:
                assert len(stages['raw_overlay_verify']) == 8 and not accesses
            windows = [e['window'] for e in state['events'] if e.get('window')]
            window = windows[-1]
            ranks.append(dict(rank=rank, stages={k:totals(v) for k,v in stages.items()}, roles=roles,
                cuda_peak_allocated_bytes=probe['peak_allocated'], cuda_peak_reserved_bytes=probe['peak_reserved'],
                rss_snapshot_bytes=probe['rss_bytes'],
                shared_host_peak_reserved_bytes=cache['shared_budget']['peak_reserved_bytes'],
                cpu_cache_peak_bytes=cache['peak_cache_bytes'], cache_wait_seconds=cache['wait_ms']/1000,
                window_host_wait_seconds=window['host_wait_ms']/1000, h2d_bytes=window['h2d_bytes'],
                h2d_copy_event_seconds=sum(c['cuda_ms'] for c in window['copies'])/1000,
                budget_admission_denials=cache['shared_budget']['admission_denials'],
                scheduler_yields=cache['pressure_yields'], consumed_releases=cache['consumed_releases'],
                future_evictions=cache['future_evictions'], all_preparations_once=True, ownership_drained=True))
        hops = [a+b for a,b in zip(case['hop_seconds'],case['selection_seconds'])]
        rows.append(dict(case=case['case'],variant='split' if split else 'monolithic',
            rpc_seconds=sum(hops),later_hops_rpc_seconds=sum(hops[1:]),hop_rpc_seconds=hops,
            max_later_denoise_seconds=max(case['hop_seconds'][1:]),aot_wall_seconds=case['aot_wall_seconds'],ranks=ranks))
    groups = {}
    for rank in (0, 1):
        for key in ('h2d_bytes', 'cuda_peak_allocated_bytes', 'cuda_peak_reserved_bytes'):
            assert len({row['ranks'][rank][key] for row in rows}) == 1, ('Changed GPU footprint/work', rank, key)
    for label in ('monolithic', 'split'):
        selected = [r for r in rows if r['variant'] == label]
        groups[label] = dict(repetitions=len(selected),mean_rpc_seconds=mean(r['rpc_seconds'] for r in selected),
            mean_later_hops_rpc_seconds=mean(r['later_hops_rpc_seconds'] for r in selected),
            max_later_denoise_seconds=max(r['max_later_denoise_seconds'] for r in selected),
            shared_host_peak_reserved_range_bytes=[min(r['ranks'][0]['shared_host_peak_reserved_bytes'] for r in selected),
                max(r['ranks'][0]['shared_host_peak_reserved_bytes'] for r in selected)],
            ranks=[])
        for rank in (0,1):
            rank_rows = [r['ranks'][rank] for r in selected]
            group_stages = {}
            for stage in sorted({s for r in rank_rows for s in r['stages']}):
                values = [r['stages'].get(stage,dict(count=0,wall_seconds=0.,thread_cpu_seconds=0.)) for r in rank_rows]
                group_stages[stage] = {key:mean(v[key] for v in values) for key in ('count','wall_seconds','thread_cpu_seconds')}
            groups[label]['ranks'].append(dict(rank=rank,mean_stages=group_stages,
                mean_window_host_wait_seconds=mean(r['window_host_wait_seconds'] for r in rank_rows),
                mean_h2d_copy_event_seconds=mean(r['h2d_copy_event_seconds'] for r in rank_rows),
                h2d_bytes=rank_rows[0]['h2d_bytes'],cuda_peak_allocated_bytes=rank_rows[0]['cuda_peak_allocated_bytes'],
                cuda_peak_reserved_bytes=rank_rows[0]['cuda_peak_reserved_bytes']))
    raw = result_path.read_bytes()
    if result_path.suffix == '.gz': raw = gzip.decompress(raw)
    return dict(status='verified',source_commit=source['commit'],capsule_sha256=source['capsule_sha256'],
        original_result_sha256=hashlib.sha256(raw).hexdigest(),hardware=hardware,worker_reports_checked=worker_reports_checked,
        denoiser_calls=32,output_hashes_identical=True,native_merge_targets_verified=128,
        full_pipeline=False,quality_qualified=False,
        no_duplicate_preparations=True,all_ownership_drained=True,supervision=supervision,
        bake_seconds=result['bake_seconds'],repack_seconds=result['repack_seconds'],
        baked_bytes=result['baked_bytes'],auxiliary_comparison=result['auxiliary_comparison'],
        cases=rows,groups=groups,
        split_rpc_change_percent=(groups['split']['mean_rpc_seconds']/groups['monolithic']['mean_rpc_seconds']-1)*100,
        metric='Sum of hop-selection and denoiser RPCs; excludes load/bake/repack/AOT/stats/service startup/teardown',
        limitations=['Two repetitions each; ABBA reduces but does not remove order/page-cache effects',
            'Two trained blocks, 24 seeded tokens; no full-model speed or generation-quality claim',
            'Nested CPU spans overlap; CUDA intervals include waits and are not Nsight kernel traces',
            'Host peaks are managed reservations; RSS is sampled; CUDA allocator peaks omit external allocations'])


def figure(summary, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes=plt.subplots(1,3,figsize=(15,4.6),gridspec_kw={'width_ratios':[1.4,1.1,.8]})
    colors={'monolithic':'#27739a','split':'#c25a36'}
    for index,row in enumerate(summary['cases']):
        label=row['variant']+(' repeat' if index>=2 else '')
        axes[0].plot(range(8),row['hop_rpc_seconds'],marker='o',markersize=4,
                     linestyle='--' if index>=2 else '-',color=colors[row['variant']],label=label)
    axes[0].set(title='Per-hop selection + denoiser RPC',xlabel='Hop',ylabel='Seconds',ylim=(0,27),xticks=range(8))
    axes[0].legend(frameon=False,fontsize=8)
    values=[]
    for label in ('monolithic','split'):
        stage=summary['groups'][label]['ranks'][0]['mean_stages']['refiner_total']
        values += [stage['wall_seconds'],stage['thread_cpu_seconds']]
    axes[1].bar(range(4),values,color=['#27739a','#8fb7cc','#c25a36','#e3ad97'])
    axes[1].set(title='Rank 0: seven refiner preparations',ylabel='Seconds (mean of two runs)',
        xticks=range(4),xticklabels=['Mono\nwall','Mono\nthread CPU','Split\nwall','Split\nthread CPU'],ylim=(0,105))
    for x,value in enumerate(values): axes[1].text(x,value+2,f'{value:.1f}',ha='center',fontsize=9)
    values=[summary['groups'][label]['ranks'][1]['mean_stages']['raw_aux_read']['wall_seconds']
            for label in ('monolithic','split')]
    axes[2].bar(range(2),values,color=[colors['monolithic'],colors['split']])
    axes[2].set(title='Rank 1: auxiliary read + verify',ylabel='Wall seconds (mean of two runs)',
                xticks=range(2),xticklabels=['Monolithic','Split'],ylim=(0,55))
    for x,value in enumerate(values): axes[2].text(x,value+1,f'{value:.2f}',ha='center',fontsize=9)
    for axis in axes:
        axis.spines[['top','right']].set_visible(False)
        axis.grid(axis='y',alpha=.2);axis.set_axisbelow(True)
    fig.suptitle('PTD P2 on 2 x Tesla T4: correct separation, slower critical path',fontsize=14)
    fig.text(.5,.01,'Two real trained blocks, 24 seeded tokens; CPU spans overlap and are not additive to RPC time.',
             ha='center',fontsize=9,color='#555555')
    fig.tight_layout(rect=(0,.05,1,.94));fig.savefig(path,dpi=170);plt.close(fig)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--submission',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--figure',type=Path)
    args=parser.parse_args()
    summary=analyze(args.root,read(args.submission))
    args.output.write_text(json.dumps(summary,indent=2),encoding='utf-8')
    if args.figure: figure(summary,args.figure)
    print(json.dumps({k:summary[k] for k in ('status','denoiser_calls','worker_reports_checked',
        'output_hashes_identical','split_rpc_change_percent')}))


if __name__=='__main__': main()
