"""Produce an immutable derived builder for 1-3 KC, 4-8 Sol, Video VAE KC.

The derivation is exact replacement of one previously completed physical 2xT4
E2E builder; do not mutate its source, notebook or original baselines.
"""
from pathlib import Path
import ast
root=Path(__file__).resolve().parents[1]
src=root/"benchmarks/build_sol_v2_production_e2e.py"
dst=root/"benchmarks/build_h3_kc_sol_8step_long_e2e.py"
text=src.read_text(encoding="utf-8")
replacements=[
("Build one production Sol-v2 0.85 MP / 10 s long-sequence E2E from the completed baseline notebook.",
 "Build one 8-NFE KC/Sol 0.85 MP long E2E on frozen previous input."),
("dit_attention_schedule='2-7:sol@1.0',","dit_attention_schedule='1-3:kitchen,4-8:sol@1.0',"),
("plan['name']='turbo_085mp_10s_sol_v2_production'",
 "plan['name']='turbo_085mp_10s_kc1to3_sol4to8_video_kc'"),
("TE Kitchen causal; DiT NFE1/NFE8 Kitchen and NFE2-7 Sol-v2 All-INT8 tau=1.0; Video VAE native-D64 Sol-v2 tau=1.0. ",
 "TE Kitchen causal; DiT NFE1-3 Kitchen INT8 and NFE4-8 restored pre-math-fix Sol-v2 All-INT8 tau=1.0; Video VAE Kitchen INT8. "),
("Production Sol-v2 long-sequence E2E overlay after physical-T4 microgate and Real-H3 attention qualification",
 "Physical 2xT4 8-NFE hybrid long E2E: NFE1-3 KC, NFE4-8 restored old Sol, Video VAE KC"),
("/kaggle/working/sol_v2_production_e2e","/kaggle/working/h3_kc_sol_8step_long_e2e"),
("/tmp/h3_sol_v2_production_20261030","/tmp/h3_kc_sol_8step_long_20261031"),
("/kaggle/working/sol_v2_production_supervision","/kaggle/working/h3_kc_sol_8step_long_supervision"),
("pamcks/h3-sol-v2-production-e2e-20261030","pamcks/h3-kc-sol-8step-long-e2e-20261031"),
("H3 Sol V2 Production E2E 20261030","H3 KC1-3 Sol4-8 VAE KC Long E2E 20261031"),
("result_relative='sol_v2_production_e2e/batch_results.json'","result_relative='h3_kc_sol_8step_long_e2e/batch_results.json'"),
("new_cpu_smoke_run=True,","new_cpu_smoke_run=False,"),
("prior_cpu_qualification_reused_for_environment_only=False,","prior_cpu_qualification_reused_for_environment_only=True,"),
("changed_source_not_claimed_cpu_qualified=False,","changed_source_not_claimed_cpu_qualified=True,"),
]
for old,new in replacements:
    count=text.count(old)
    expected=2 if old=='/kaggle/working/sol_v2_production_e2e' else 1
    if count!=expected:raise ValueError(f"Expected {expected} occurrences got {count} for {old}")
    text=text.replace(old,new)
# Preserve original input exactly except attention policy + identifiers.
point="    validate(plan)\n\n    hashes={}"
insert="""    validate(plan)
    from patches.attention_schedule import parse_attention_schedule,select_attention_policy
    scheduled=parse_attention_schedule(plan['operators']['dit_attention_schedule'])
    actual=[select_attention_policy(scheduled,n,'kitchen',plan['operators']['sol_tau'])
            for n in range(1,9)]
    assert actual==([('kitchen',None)]*3+[('sol',1.0)]*5),actual
    assert plan['operators']['video_attention']=='kitchen'
    assert plan['operators']['te_attention']=='kitchen_causal'
    # Fail closed if somebody reintroduced the rejected math-parity patch.
    restore_manifest=json.loads((ROOT/'docs/data/h3_sol_v2_active_source_identity.json').read_text())
    for name,record in restore_manifest['paths'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==record['sha256'],name

    hashes={}"""
assert text.count(point)==1
text=text.replace(point,insert)
# Exact immutable attention defaults in Kaggle worker process.
point="env=dict(os.environ);env.update(json.loads(pathlib.Path('/kaggle/working/light_setup/execution_env.json').read_text()))"
new=point+"""
env['H3_SOL_WORKSPACE_POOL']='0'
env['H3_SOL_LUT_BITMAP']='0'
"""
assert text.count(point)==1
text=text.replace(point,new)
# Fix provenance notes: E2E is not yet qualified; only source/E2E assumptions are tested.
point="sm75_native_compile='Current native source is source-locked and AOT-compiled in the job; current full production overlay still requires physical-T4 E2E qualification',"
new="sm75_native_compile='Native SM75 ABI12 compile is required in the job; this new exact hybrid E2E has no prior physical qualification',"
assert text.count(point)==1
text=text.replace(point,new)
point="    plan['source_note']=("
assert text.count(point)==1
assert "Video VAE Kitchen INT8" in text and "dit_attention_schedule='1-3:kitchen,4-8:sol@1.0'" in text
ast.parse(text)
dst.write_text(text,encoding="utf-8")
print("BUILDER_WRITTEN",dst.relative_to(root),"BYTES",dst.stat().st_size)
