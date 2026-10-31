"""Derive a separate all-Kitchen long E2E builder from the source-locked hybrid E2E.

Keep the exact successful 2026-10-29 0.85MP 10s 8-step payload and the same
source closure as the recent hybrid job. The only algorithmic policy change
vs the recent hybrid job is DiT NFE4-8 -> Kitchen. VAE already Kitchen.
"""
import ast
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
src=ROOT/"benchmarks/build_h3_kc_sol_8step_long_e2e.py"
dst=ROOT/"benchmarks/build_h3_kc_all_8step_long_e2e.py"
data=src.read_text(encoding="utf-8")
replacements=[
("Build one 8-NFE KC/Sol 0.85 MP long E2E on frozen previous input.",
 "Build one all-Kitchen INT8 8-NFE 0.85 MP long E2E on frozen previous input."),
("dit_attention_schedule='1-3:kitchen,4-8:sol@1.0',", "dit_attention_schedule='',"),
("plan['name']='turbo_085mp_10s_kc1to3_sol4to8_video_kc'",
 "plan['name']='turbo_085mp_10s_kc_all_8step_video_kc'"),
("'Exact completed 0.85MP regular Turbo workload inputs replayed against the promoted production Sol-v2 path. '",
 "'Exact completed 0.85MP regular Turbo workload inputs replayed on the same frozen source closure as the hybrid E2E. '"),
("'Production Sol semantics: native D64/D128, H3 token-space temporal/local routing, exact generated-audio prefix, '",
 "'The all-KC attention policy is the independent controlled comparison to the partial-Sol variant. '"),
("'frame-boundary protection, sync-free LUT, and metadata-only runtime checkpoint admission. '",
 "'No model/input/seed/LoRA/sample/geometry/shift changes. '"),
("'TE Kitchen causal; DiT NFE1-3 Kitchen INT8 and NFE4-8 restored pre-math-fix Sol-v2 All-INT8 tau=1.0; Video VAE Kitchen INT8. '",
 "'TE Kitchen causal; all eight DiT NFEs Kitchen INT8; Video VAE Kitchen INT8, Audio VAE native. '"),
("assert actual==([('kitchen',None)]*3+[('sol',1.0)]*5),actual",
 "assert actual==[('kitchen',None)]*8,actual"),
("h3_kc_sol_8step_long_e2e","h3_kc_all_8step_long_e2e"),
("h3_kc_sol_8step_long_20261031","h3_kc_all_8step_long_20261031"),
("h3_kc_sol_8step_long_supervision","h3_kc_all_8step_long_supervision"),
("Physical 2xT4 8-NFE hybrid long E2E: NFE1-3 KC, NFE4-8 restored old Sol, Video VAE KC",
 "Physical 2xT4 8-NFE long E2E: NFE1-8 Kitchen INT8, Video VAE Kitchen INT8"),
("pamcks/h3-kc-sol-8step-long-e2e-20261031","pamcks/h3-kc-all-e2e-20261031"),
("H3 KC1-3 Sol4-8 VAE KC Long E2E 20261031","H3 KC All E2E 20261031"),
]
for old,new in replacements:
    n=data.count(old)
    if n<1:raise AssertionError((n,old))
    data=data.replace(old,new)
# The old baseline metadata still holds an id_no from a 2026-10-29 kernel. It
# overrides the textual id on push and previously caused a v4 update, not new
# notebook creation. Remove id_no in this new job only.
needle="    meta.update(id='pamcks/h3-kc-all-e2e-20261031',"
assert data.count(needle)==1
data=data.replace(needle,"    meta.pop('id_no',None) # Prevent updating the old historical notebook ID.\n"+needle)
# New job is an all-dense KC reference. The source still includes pre-fix Sol
# files for identical closure/provenance, but zero Sol dispatch is expected.
data=data.replace("sm75_native_compile='Native SM75 ABI12 compile is required in the job; this new exact hybrid E2E has no prior physical qualification',",
"sm75_native_compile='All-Kitchen runtime is selected; frozen pre-math Sol code is included for comparable source closure but must not dispatch',")
data=data.replace("source-locked experimental Sol-v2 path","source-locked full E2E runtime")
data=data.replace("    assert plan['operators']['video_attention']=='kitchen'\n",
"    assert plan['operators']['video_attention']=='kitchen'\n    assert plan['operators']['dit_attention_schedule']==''\n")
data=data.replace("    assert plan['operators']['te_attention']=='kitchen_causal'\n",
"    assert plan['operators']['te_attention']=='kitchen_causal'\n    assert plan['operators']['dit_attention']=='kitchen'\n")
assert data.count("meta.pop('id_no',None)")==1
assert "dit_attention_schedule=''" in data
assert "actual==[('kitchen',None)]*8" in data
ast.parse(data)
dst.write_text(data,encoding="utf-8")
print("GENERATED",dst.relative_to(ROOT),dst.stat().st_size)
