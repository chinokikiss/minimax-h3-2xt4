"""Freeze CPU-only optimization evidence without contacting Kaggle."""
import hashlib
import json
from pathlib import Path


def main():
    root=Path(__file__).resolve().parents[1]
    base=Path('D:/attention_eval/operator_optimization_tp2_20261016')
    out=base/'offline_optimization'
    frozen=json.loads((base/'v1/tp2_operators/source_hashes.json').read_text())
    unchanged={name:hashlib.sha256((root/name).read_text(encoding='utf8').encode()).hexdigest()==frozen[name]
        for name in ['patches/tilelang_grouped_tp.py','patches/tilelang_grouped_fused.py','patches/tilelang_sparse_attention.py']}
    assert all(unchanged.values()),'The physical-T4 baseline changed'
    lower=json.loads((out/'lowering.json').read_text())['cases']
    compiled=json.loads((out/'compilation.json').read_text())['cases']
    real=json.loads((out/'real_operand_identity.json').read_text())['cases']
    assert len(lower)==len(compiled)==8 and len(real)==10
    assert all(c['exit_code']==0 and c['spill_store_bytes']==c['spill_load_bytes']==0 for c in compiled)
    names=['patches/tilelang_grouped_tp_optimized.py','benchmarks/test_grouped_tp_optimized.py',
           'benchmarks/lower_grouped_tp_optimized.py','benchmarks/compile_grouped_tp_offline.py',
           'benchmarks/tp2_nccl_microbench.py','benchmarks/tp2_timing.py']
    result={'status':'offline_checks_complete','scope':'This continuation only','new_Kaggle_GPU_jobs':0,'full_H3_generations':0,
        'T4_execution_verified':False,'performance_claims':False,'production_dispatch_changed':False,
        'sparse_policy':'Current tau0 routes, prefix/diagonal/global protections, drop and scalar pooling are frozen.',
        'quality_policy':'No sparse precision work or dense-relative L1/L2 quality gate. Full-generation distribution/semantic validation deferred.',
        'frozen_baseline_unchanged':unchanged,'real_CPU_cases':len(real),'compiled_SM75_variants':len(compiled),
        'next_hardware_gate':'Current TP2 shapes, operator equivalence, warmed whole-API timings; no SM120 substitution.',
        'source_sha256':{name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in names}}
    (out/'state.json').write_text(json.dumps(result,indent=2))
    print('OFFLINE_VERIFIED: unchanged T4 baseline; 10 real CPU identity cases; 8 SM75 cubins; zero new GPU jobs')


if __name__=='__main__':main()
