"""Generate a source-verified report from downloaded physical T4 evidence."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]


def analyze(folder):
    result=json.loads((folder/'results.json').read_text())
    hardware=json.loads((folder/'hardware.json').read_text())
    assert result['physical'] and result['compute_capability']==[7,5]
    assert 'T4' in result['gpu']
    assert result['completed'], 'Incomplete matrix'
    for name,expected in result['source_sha256'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==expected, 'Source mismatch: '+name
    modes=defaultdict(list)
    for row in result['cases']:modes[row['mode']].append(row)
    passed=[r for r in result['cases'] if r.get('passed')]
    ptchecks=[]
    for path in sorted(folder.glob('*.ptx')):
        text=path.read_text()
        ptchecks.append(dict(file=path.name,sm75='.target sm_75' in text,
                             int8_mma='mma.sync.aligned.m8n8k16' in text,
                             fp16_mma='mma.sync.aligned.m16n8k8' in text,
                             forbidden=any(x in text for x in ['cp.async','wgmma.','mma.sp.'])))
    assert ptchecks and all(r['sm75'] and r['int8_mma'] and not r['forbidden'] for r in ptchecks)
    summary=dict(physical=True,passed=result['passed'],records=len(result['cases']),passed_records=len(passed),
                 worst_implementation_relative_l2=max(r['implementation_error']['rel_l2'] for r in passed),
                 worst_implementation_max_abs=max(r['implementation_error']['max_abs'] for r in passed),
                 worst_lse_max_abs=max(r['lse_max_abs'] for r in passed),ptx_checks=ptchecks)
    (folder/'summary.json').write_text(json.dumps(summary,indent=2))
    lines=['# TileLang INT8 sparse attention: physical T4 experiments',
           '', 'Experiment date: 2026-10-13. Submitted and collected using the Kaggle CLI.',
           '', f"Hardware: {result['gpu']} / SM75. Runtime: PyTorch {hardware['torch']}, TileLang {hardware['tilelang']}, CUDA {hardware['torch_cuda']}. The Kaggle session exposes two T4s; these operator experiments execute on GPU 0 only.",
           '', f"**{len(passed)}/{len(result['cases'])} cases passed the prespecified implementation gates.** All records finished: {result['completed']}. Overall pass: {result['passed']}.",
           '', f"Worst passing implementation error: relative L2 {summary['worst_implementation_relative_l2']:.6g}, max absolute {summary['worst_implementation_max_abs']:.6g}, LSE max absolute {summary['worst_lse_max_abs']:.6g}.",
           '', '## What was implemented', '',
           '- TileLang INT8 Tensor Core QK with INT32 accumulation, per-token Q/K scales and shared K centering.',
           '- FP32 online softmax/normalization; FP16 PV or block/channel-scaled INT8 PV with block-local probability quantization.',
           '- Sol pooled residual blocks using FP16 K/V means and explicit valid counts; selected-block-only top-k and token-order local/global controls.',
           '- D=64/128, BHND FP16 tensors, padded/ragged lengths and strided input preparation. Matching non-causal self-attention only; no GQA, dropout, arbitrary masks or ComfyUI integration in this qualification.',
           '- Preparation, summaries and routing are currently PyTorch GPU operations. TileLang implements the tiled attention operator; preparation fusion and performance tuning are deferred.',
           '', '## Implementation correctness versus approximation accuracy', '',
           'The implementation gate compares each TileLang result with an independent streaming PyTorch reference using identical quantized operands, routes, pooling rules and probability-cast rules. Its limits were set before the first hardware run: relative L2 <=0.002, max absolute <=0.015 and LSE max absolute <=0.003. No NxN score matrix is used by this reference.',
           '', 'Dense-FP16 fidelity is reported separately against FP32 unquantized attention rounded to FP16. A passing implementation can still be a poor approximation to dense attention. The inputs are synthetic; these results do not certify MiniMax H3 generation quality.',
           '', '| Approach | Passed cases | Worst implementation relative L2 |', '|---|---:|---:|']
    for mode,rows in modes.items():
        good=[r for r in rows if r.get('passed')]
        worst=max((r['implementation_error']['rel_l2'] for r in good),default=float('nan'))
        lines.append(f'| {mode} | {len(good)}/{len(rows)} | {worst:.6g} |')
    for name in ['random2048_h4','local2048_h4','local8192_h28']:
        rows=[r for r in result['cases'] if r['case']['name']==name]
        lines+=['',f'## {name}', '',
                '| Approach | Selected blocks | Relative L2 vs dense | Kernel median ms | Complete call median ms |',
                '|---|---:|---:|---:|---:|']
        for row in rows:
            if row.get('passed'):
                lines.append(f"| {row['mode']} | {row['selected_density']*100:.2f}% | {row['vs_dense_fp16']['rel_l2']:.6g} | {row['forward']['median_ms']:.3f} | {row['end_to_end']['median_ms']:.3f} |")
            else:lines.append(f"| {row['mode']} | — | {row['status']} | — | — |")
        baseline=next((r for r in result.get('pytorch_baselines',[]) if r['case']['name']==name),None)
        if baseline:
            lines.append(f"| PyTorch FP16 SDPA | 100% | {baseline['vs_dense_fp32']['rel_l2']:.6g} | {baseline['forward']['median_ms']:.3f} | — |")
    lines+=['','## Interpretation and limits', '',
            'These are warmed, untuned CUDA-event measurements, with compilation reported separately in JSON. Complete-call timing includes PyTorch preparation/routing, packing, allocation, validation and the TileLang kernel. Preparation currently builds V8 buffers for every approach; this is a common correctness scaffold and unnecessary overhead for FP16-PV modes. Peak memory is the benchmark process allocation while references and inputs are retained, not an isolated operator-memory claim.',
            '', 'The local-signal input gives each leaf a strong shared Q/K signature, making its own keys dominant. The random input has diffuse attention and makes omitted/pooled remote keys consequential. Equal outputs on the local-signal case do not demonstrate equivalent quality on a real video model. Denser tau/budget settings are included to expose the accuracy/work tradeoff.',
            '', 'The first hardware version passed 42/50 checks. Its eight D128 INT8-PV launch failures requested 69,120 bytes of shared memory. The corrected implementation reduces the summary-key tile to 32 columns and stages its distinct fragment layout through shared memory; this changes staging, not the mathematical pooling contract.',
            '', 'PTX exports confirm `.target sm_75`, INT8 `m8n8k16` and FP16 `m16n8k8` instruction families. No cp.async, WGMMA or hardware sparse-MMA path was found in those exports. Numerical checks ran on a physical T4; Compute Sanitizer and captured H3 tensor/generation qualification remain separate future checks.',
            '', '## Reproduce', '', 'Run from the workspace with an authenticated Kaggle CLI:', '',
            '```powershell', "& 'C:\\ProgramData\\miniconda3\\envs\\comfyui\\python.exe' -m unittest benchmarks.test_tilelang_attention_contract -v",
            "& 'C:\\ProgramData\\miniconda3\\envs\\comfyui\\python.exe' run_tilelang_t4_kaggle.py", '```', '',
            'The runner packages only the two experiment modules into a private notebook, pins TileLang 0.1.15, requests NvidiaTeslaT4, records hardware/source hashes and downloads the outputs. It does not modify existing model benchmark notebooks.',
            '', 'Minimal operator usage:', '', '```python',
            'from patches.tilelang_sparse_attention import prepare, make_routes, execute',
            'p = prepare(q, k, v)  # FP16 [B,H,N,D] on an SM75 CUDA device',
            "routes = make_routes(p, 'sol', tau=1.0)",
            'out, lse = execute(p, routes, pooled=True, int8_pv=False)', '```', '',
            f'Raw evidence directory: `{folder.resolve()}`. The source hashes in results.json must match the local operator and benchmark for this report generator to accept the run.', '']
    report=ROOT/'docs/reports/TILELANG_T4_ATTENTION_RESULTS.md'
    report.write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps(summary,indent=2));print(report)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('folder',type=Path)
    analyze(parser.parse_args().folder)
