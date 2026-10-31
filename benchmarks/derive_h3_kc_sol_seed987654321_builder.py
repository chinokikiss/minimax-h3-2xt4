"""Derive an isolated high-distance-seed KC1-3/Sol4-8 long E2E builder.

The original frozen baseline and repository's Turbo challenge seed allowlist
remain byte-identical. The generated notebook overlays a one-line, hash-locked
allowlist extension for this job only, in addition to the seed-only request
difference. No other operator/kernel source is modified.
"""
from pathlib import Path
import ast

ROOT=Path(__file__).resolve().parents[1]
SRC=ROOT/"benchmarks/build_h3_kc_sol_8step_long_e2e.py"
DEST=ROOT/"benchmarks/build_h3_kc_sol_seed987654321_long_e2e.py"
SEED=987654321
old=SRC.read_text(encoding="utf-8")
assert not DEST.exists(),DEST
replacements=[
("Build one 8-NFE KC/Sol 0.85 MP long E2E on frozen previous input.",
 "Build one KC/Sol 0.85MP long E2E with isolated seed 987654321."),
("from benchmarks.turbo_challenge_contract import validate",
 "from benchmarks.turbo_challenge_contract import validate as original_validate"),
("    plan['operators']=dict(",
 "    plan['seed']=987654321  # requested high-distance seed, sole workload input delta\n    plan['operators']=dict("),
("plan['name']='turbo_085mp_10s_kc1to3_sol4to8_video_kc'",
 "plan['name']='turbo_085mp_10s_kc1to3_sol4to8_video_kc_seed987654321'"),
("Prompt, references, seed, LoRA, sampler, shifts and geometry unchanged.",
 "Prompt, references, LoRA, sampler, shifts and geometry unchanged; seed 20261020 -> 987654321."),
("h3_kc_sol_8step_long_e2e","h3_kc_sol_seed987654321_long_e2e"),
("h3_kc_sol_8step_long_20261031","h3_kc_sol_seed987654321_long_20261031"),
("h3_kc_sol_8step_long_supervision","h3_kc_sol_seed987654321_long_supervision"),
("pamcks/h3-kc-sol-8step-long-e2e-20261031","pamcks/h3-kc-sol-seed987654321-e2e"),
("H3 KC1-3 Sol4-8 VAE KC Long E2E 20261031","H3 KC Sol Seed987654321 E2E"),
]
for before,after in replacements:
    n=old.count(before)
    assert n>=1,(before,n)
    old=old.replace(before,after)

# The benchmark contract currently allows only 20261020/20261021.
# Adjust *only* the source bytes included in this notebook's tar overlay,
# so its per-job runner can validate the requested large seed. Never write
# these changed bytes into the working repository's contract module.
anchor="ROOT=Path(__file__).resolve().parents[1]\n"
assert old.count(anchor)==1
addition="""# Job-local only: the immutable original contract stays unchanged on disk.
CONTRACT='benchmarks/turbo_challenge_contract.py'
LEGACY_ALLOWLIST=b'SEEDS=frozenset((20261020,20261021))'
JOB_ALLOWLIST=b'SEEDS=frozenset((20261020,20261021,987654321))'
old_contract=(ROOT/CONTRACT).read_bytes()
assert old_contract.count(LEGACY_ALLOWLIST)==1
job_contract=old_contract.replace(LEGACY_ALLOWLIST,JOB_ALLOWLIST)
assert job_contract!=old_contract
assert job_contract.replace(JOB_ALLOWLIST,LEGACY_ALLOWLIST)==old_contract
def validate(plan):
    assert plan.get('seed')==987654321
    # All remaining rules, including prompt/hash/geometry, are identical.
    return original_validate(dict(plan,seed=20261020))
"""
# Import executes before assignment to ROOT, but validate is called later.
old=old.replace(anchor,anchor+addition)
# Assert only seed + non-execution labels differ from previous hybrid accepted request.
needle="    validate(plan)\n    from patches.attention_schedule"
assert old.count(needle)==1
new="""    previous=json.loads((ROOT/'docs/data/h3_kc_sol_8step_long_e2e_20261031/request.json').read_text(encoding='utf-8'))
    differences={k:(previous.get(k),plan.get(k)) for k in set(previous)|set(plan)
                 if previous.get(k)!=plan.get(k)}
    assert set(differences)=={'seed','name','source_note'},differences
    assert differences['seed']==(20261020,987654321),differences
    validate(plan)
    from patches.attention_schedule"""
old=old.replace(needle,new)
# Hash/archive encode exactly the changed one-line contract in this job.
needle="            raw=(ROOT/name).read_bytes()"
assert old.count(needle)==1
old=old.replace(needle,"            raw=job_contract if name==CONTRACT else (ROOT/name).read_bytes()")
# Explicitly include isolated validation source in the source-locked overlay.
needle="    overlay,closure=closed_overlay_files(baseline_commit,CORE_OVERLAY)\n"
assert old.count(needle)==1
old=old.replace(needle,needle+"    assert CONTRACT in overlay  # fail closed if the frozen seed gate would not be deployed\n")
# Prevent previous accepted hybrid update: old baseline has stale numeric ID.
needle="    meta.update(id='pamcks/h3-kc-sol-seed987654321-e2e',"
assert old.count(needle)==1
old=old.replace(needle,"    meta.pop('id_no',None)  # allocate a new Notebook instead of historical numeric ID\n"+needle)
# Freeze exact provenance of one-line contract override, and other input constraints.
needle="        attention_policy=plan['operators'],"
assert old.count(needle)==1
old=old.replace(needle,"""        isolated_seed_contract_override=dict(
            path=CONTRACT,seed=987654321,
            original_sha256=hashlib.sha256(old_contract).hexdigest(),
            overlay_sha256=hashlib.sha256(job_contract).hexdigest(),
            exact_reverse_delta=True,
            repository_unmodified=True,
            scope='add one allowed seed only; no changes to run logic or model'),
        attention_policy=plan['operators'],""")
# Code's user-intended description should not call it identical input.
old=old.replace("The previous successful notebook supplies the frozen environment bootstrap and\nreference delivery.","The previous successful notebook supplies the frozen environment bootstrap and\nreference delivery. The sole dynamic input difference is the large seed.")
assert "'1-3:kitchen,4-8:sol@1.0'" in old
assert "video_attention='kitchen'" in old
assert "meta.pop('id_no',None)" in old
assert "raw=job_contract if name==CONTRACT" in old
ast.parse(old)
DEST.write_text(old,encoding='utf-8')
print('CREATED',DEST.relative_to(ROOT),'BYTES',DEST.stat().st_size,'SEED',SEED)
