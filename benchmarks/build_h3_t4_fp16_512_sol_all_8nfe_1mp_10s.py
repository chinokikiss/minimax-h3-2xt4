"""Build one 1MP/10s dual-T4 FP16/512, debug-off, Sol-v2 all-NFE E2E.

Reuses the pinned 2026-11-01 0.85MP Turbo source publication and CPU
qualification for environment setup, then overlays changed local production
sources. Never submits or polls Kaggle; submission is a separate one-shot step.
"""
from __future__ import annotations
import ast
import base64
import hashlib
import io
import json
import lzma
from pathlib import Path
import re
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[1]
IMAGE = ROOT / "bench_assets/public_reference_images/cd4cbfd7137b80e85e64ada86ffd93e512b874de84cb4291fc0278b747461e03.png"
EXPECTED_SHA = "cd4cbfd7137b80e85e64ada86ffd93e512b874de84cb4291fc0278b747461e03"
EXPECTED_BYTES = 7703256
MIRROR = (
    "https://raw.githubusercontent.com/chinokikiss/minimax-h3-2xt4/"
    "225da31d025d510bb152683f53cf6f0a9843e0d2/"
    "bench_assets/public_reference_images/" + EXPECTED_SHA + ".png"
)

# This is inserted only into the one-shot Kaggle worker, after the verified
# source overlay and before run_h3_l1_turbo_batch.py. The frozen image identity
# and production code remain untouched.
REFERENCE_STAGE = r"""
# Benchmark-only reference delivery. Do not alter l1_assets_identity.json.
import requests,time
_rows=json.loads((source/'l1_assets_identity.json').read_text())['images']
_matches=[r for r in _rows if r['file']=='reference_shrine.png']
assert len(_matches)==1
_ref=_matches[0]
assert (_ref['sha256'],_ref['bytes'],_ref['size'])==(SHRINE_SHA,SHRINE_BYTES,[4096,2048])
_target=source/'bench_assets/l1_current/reference_shrine.png'
_target.parent.mkdir(parents=True,exist_ok=True)
if not _target.is_file():
    for _attempt in range(3):
        try:
            _response=requests.get(SHRINE_MIRROR,
                headers={'User-Agent':'H3-REF2VA-evaluation/1.0'},
                timeout=(20,90))
            _response.raise_for_status()
            _data=_response.content
            break
        except requests.RequestException as _error:
            _status=getattr(getattr(_error,'response',None),'status_code',None)
            if _attempt==2 or (_status is not None and _status not in (408,429,500,502,503,504)):
                raise
            _delay=(5,15)[_attempt]
            print('H3_REFERENCE_MIRROR_RETRY',_attempt+1,_status,_delay,flush=True)
            time.sleep(_delay)
    if len(_data)!=SHRINE_BYTES or hashlib.sha256(_data).hexdigest()!=SHRINE_SHA:
        raise ValueError('Immutable shrine mirror has different bytes')
    _target.write_bytes(_data)
if _target.stat().st_size!=SHRINE_BYTES or hashlib.sha256(_target.read_bytes()).hexdigest()!=SHRINE_SHA:
    raise ValueError('Staged shrine reference has wrong bytes')
(out/'reference_delivery_v2.json').write_text(json.dumps(dict(
    mode='commit_pinned_github_mirror',local_source_sha256=SHRINE_SHA,
    source_url=SHRINE_MIRROR,bytes=SHRINE_BYTES,verified=True),indent=2))
print('H3_REFERENCE_STAGED',SHRINE_SHA,SHRINE_BYTES,flush=True)
""".replace("SHRINE_SHA", repr(EXPECTED_SHA)).replace(
    "SHRINE_BYTES", repr(EXPECTED_BYTES)).replace("SHRINE_MIRROR", repr(MIRROR))

EVIDENCE = ROOT / "docs/data/h3_t4_fp16_512_sol_all_8nfe_1mp_10s_20261102"
BASE = ROOT / "docs/data/h3_t4_fp16_512_sol_all_8nfe_10s_20261101/base_job"
JOB = EVIDENCE / "gpu_job"

sys.path.insert(0, str(ROOT))
from patches.attention_schedule import parse_attention_schedule, select_attention_policy
from benchmarks.source_overlay_closure import overlay_files, assert_closed
from benchmarks.t4_notebook_watchdog import supervise_worker

SEEDS = (
    "patches/comfy_tp_runtime.py",
    "patches/h3_tensor_flow.py",
    "patches/diagnostic_metrics.py",
    "patches/host_weight_cache.py",
    "patches/weight_residency.py",
    "patches/runtime_diagnostics.py",
    "patches/staged_vae.py",
    "patches/ptd/runtime.py",
    "patches/h3_operator_policy.py",
    "patches/h3_operator_aot.py",
    "patches/h3_operator_dependencies.py",
    "patches/sol_attention.py",
    "patches/sol_sm75_backend.py",
    "patches/tilelang_row_quant.py",
    "patches/tilelang_swiglu_quant.py",
    "patches/tilelang_swiglu_inplace.py",
    "patches/tilelang_transport_p2.py",
    "patches/tilelang_dit_fusion.py",
    "patches/dit_kitchen_turing_a1.py",
    "benchmarks/ref2va_e2e_run.py",
    "benchmarks/run_h3_l1_turbo_e2e.py",
    "benchmarks/run_h3_l1_turbo_batch.py",
    "benchmarks/e2e_attention_validation.py",
    "benchmarks/turbo_challenge_contract.py",
    "custom_nodes/minimax_h3_tp2/__init__.py",
)
RETIRED = (
    "patches/tilelang_group32_masked.py",
    "patches/tilelang_te_preparation.py",
    "patches/tilelang_attention_p2.py",
    "patches/tilelang_attention_tradeoffs.py",
    "patches/tilelang_dense_preparation.py",
    "patches/tilelang_grouped_fused.py",
    "patches/tilelang_grouped_qk.py",
    "patches/tilelang_grouped_tp.py",
    "patches/tilelang_grouped_tp_kitchen.py",
    "patches/tilelang_grouped_tp_kitchen_v2.py",
    "patches/tilelang_grouped_tp_optimized.py",
    "patches/tilelang_sage_preparation.py",
    "patches/tilelang_sage_qk.py",
    "patches/tilelang_scalar_pool.py",
    "patches/tilelang_sol_drop_fused.py",
    "patches/tilelang_sparse_attention.py",
)


def main():
    if JOB.exists():
        raise FileExistsError("Preserve prepared/submitted job; never overwrite: " + str(JOB))
    bind = json.loads((BASE / "source_binding.json").read_text(encoding="utf-8"))
    publication = bind["publication"]
    previous = json.loads((ROOT / "docs/data/h3_t4_fp16_512_sol_all_8nfe_10s_20261101/request.json").read_text(encoding="utf-8"))
    assert publication["commit"] == "d7c7528515d2a7961de5e6f961232d81d21cdc8b"
    assert publication["published"] and bind["mode"] == "e2e" and bind["turbo_e2e"]
    assert (previous["width"], previous["height"], previous["frames"], previous["fps"]) == (1216, 704, 243, 24)
    assert (previous["seed"], previous["steps"], previous["sampler"], previous["scheduler"]) == (20261020, 8, "euler", "simple")
    assert (previous["video_shift"], previous["audio_shift"], previous["cfg"]) == (6, 3, 1)
    assert previous["offline_lora"] and previous["disk_baked_lora"]
    assert previous["turbo_lora"]["strength"] == 1
    assert previous["turbo_lora"]["file"] == "minimax_h3_ref2v_turbo_8step_v1.0_768p_comfyui_bf16.safetensors"

    plan = dict(previous)
    plan.update(
        name="H3_FP16_512_SOL_V2_ALL_8_NFE_1MP_10S_P0_DEFAULTS",
        width=1344,
        height=768,
        debug=False,
        residual_exponent=9,
        operators=dict(
            te_attention="kitchen_causal",
            dit_attention="sol",
            dit_attention_schedule="1-8:sol@1.0",
            sol_tau=1.0,
            sol_topology_v2=True,
            sol_sync_free_lut=True,
            sol_collect_route_stats=False,
            sol_temporal_radius=1,
            sol_local_radius_tokens=64,
            sol_protect_frame_boundaries=True,
            video_attention="kitchen",
        ),
        source_note=(
            "1MP challenge 1344x768/243f derived from completed 1216x704, seed-20261020, 8-step Euler/Simple "
            "Turbo workload; current FP16 residual 1/512, explicit debug=False, "
            "Sol v2 tau=1.0 on every NFE1-8, current P0-A1/B/C and Sol key-only path. Same TE, inputs, "
            "Turbo LoRA, shifts, and decoder semantics retained."
        ),
    )
    assert all(plan[key] == previous[key] for key in previous
               if key not in ("name", "source_note", "operators", "width", "height"))
    assert (plan["width"],plan["height"]) == (1344,768)
    assert plan["residual_exponent"] == 9 and plan["debug"] is False
    from patches.h3_operator_policy import H3OperatorPolicy
    assert all((H3OperatorPolicy().dit_p0_2_lifetime,H3OperatorPolicy().dit_p0_3_lifetime,
                H3OperatorPolicy().dit_p0_a1_turing,H3OperatorPolicy().dit_p0_b_inplace,
                H3OperatorPolicy().dit_p0_c_local_rotation))
    schedule = parse_attention_schedule(plan["operators"]["dit_attention_schedule"])
    selected = [select_attention_policy(schedule, n, "sol", 1.0) for n in range(1, 9)]
    assert selected == [("sol", 1.0)] * 8, selected

    from benchmarks.turbo_challenge_contract import validate as validate_turbo
    validate_turbo(plan)
    overlay, closure = overlay_files(publication["commit"], SEEDS)
    assert_closed(publication["commit"], overlay, SEEDS)
    hashes = {}
    memory = io.BytesIO()
    with tarfile.open(fileobj=memory, mode="w:") as archive:
        for name in overlay:
            raw = (ROOT / name).read_bytes()
            hashes[name] = hashlib.sha256(raw).hexdigest()
            entry = tarfile.TarInfo(name)
            entry.size = len(raw)
            entry.mtime = 0
            entry.uid = 0
            entry.gid = 0
            entry.uname = ""
            entry.gname = ""
            archive.addfile(entry, io.BytesIO(raw))
    packed = lzma.compress(memory.getvalue(), preset=9 | lzma.PRESET_EXTREME)
    encoded = base64.b85encode(packed).decode()
    packed_sha = hashlib.sha256(packed).hexdigest()

    source_notebook = json.loads((BASE / "experiment.ipynb").read_text(encoding="utf-8"))
    assert len(source_notebook["cells"]) == 1
    old = "".join(source_notebook["cells"][0]["source"])
    marker = "print('T4_SUPERVISOR notebook_cell_started',flush=True)"
    assert old.count(marker) == 2
    prefix = old[:old.rfind(marker)]
    execution = r'''import base64,hashlib,io,json,lzma,os,pathlib,sys,tarfile
env=dict(os.environ)
env.update(json.loads(pathlib.Path('/kaggle/working/light_setup/execution_env.json').read_text()))
env['MINIMAX_H3_PROFILE_DETAILS']='0'
env['H3_SOL_WORKSPACE_POOL']='0'
env['H3_SOL_LUT_BITMAP']='0'
source=pathlib.Path('/kaggle/temp/h3_light_source')
out=pathlib.Path('/kaggle/working/h3_t4_fp16_512_sol_v2_all_8nfe_1mp_10s');out.mkdir()
manifest=json.loads(pathlib.Path('/kaggle/temp/h3_light_distribution/delivery_manifest.json').read_text())
for name,expected in manifest['source_hashes'].items():
    path=source/name
    assert path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest()==expected,('Baseline source mismatch',name)
raw=base64.b85decode(OVERLAY)
assert hashlib.sha256(raw).hexdigest()==OVERLAY_SHA
with tarfile.open(fileobj=io.BytesIO(lzma.decompress(raw)),mode='r:') as archive:
    for entry in archive:
        rel=pathlib.PurePosixPath(entry.name)
        assert entry.isfile() and not rel.is_absolute() and '..' not in rel.parts
        destination=source.joinpath(*rel.parts)
        destination.parent.mkdir(parents=True,exist_ok=True)
        with archive.extractfile(entry) as stream:destination.write_bytes(stream.read())
for name,expected in HASHES.items():
    assert hashlib.sha256((source/name).read_bytes()).hexdigest()==expected,('Overlay mismatch',name)
for name in RETIRED:
    target=source/name
    if target.exists():target.unlink()
(out/'request.json').write_text(json.dumps(REQUEST,indent=2))
(out/'source_overlay.json').write_text(json.dumps(dict(baseline_commit=COMMIT,
    overlay_hashes=HASHES,scope='FP16/512 debug-off all-eight-NFE Sol-v2 1MP 10s Turbo E2E P0 defaults'),indent=2))
argv=[sys.executable,'-u',str(source/'benchmarks/run_h3_l1_turbo_batch.py'),
    '--source',str(source),'--output',str(out),'--plan',str(out/'request.json'),
    '--scratch','/tmp/h3_t4_fp16_512_sol_v2_1mp_10s_20261102','--e2e-timeout','5400']
os.execve(sys.executable,argv,env)
'''
    for key, value in (
        ("OVERLAY_SHA", repr(packed_sha)),
        ("OVERLAY", repr(encoded)),
        ("HASHES", repr(hashes)),
        ("RETIRED", repr(RETIRED)),
        ("REQUEST", repr(plan)),
        ("COMMIT", repr(publication["commit"])),
    ):
        execution = re.sub(r"\b" + key + r"\b", lambda _match: value, execution)
    # Same commit-pinned, hash-verified shrine staging as the successful v2 E2E.
    assert IMAGE.is_file() and len(IMAGE.read_bytes())==EXPECTED_BYTES
    assert hashlib.sha256(IMAGE.read_bytes()).hexdigest()==EXPECTED_SHA
    assert execution.count("(out/'request.json').write_text(")==1
    execution=execution.replace("(out/'request.json').write_text(",
        REFERENCE_STAGE+"(out/'request.json').write_text(",1)
    code = prefix + supervise_worker(execution, budget=7200,
        output="/kaggle/working/h3_t4_fp16_512_sol_v2_1mp_10s_supervision")
    code += ("assert json.loads(pathlib.Path('/kaggle/working/"
        "h3_t4_fp16_512_sol_v2_all_8nfe_1mp_10s/batch_results.json').read_text())"
        "['status']=='complete'\n")
    ast.parse(code)
    assert not re.search(r"(?:hf_[A-Za-z0-9]{25,}|gh[pousr]_[A-Za-z0-9]{30,}|KGAT_[A-Za-z0-9_-]{20,})", code)
    source_notebook["cells"][0]["source"] = code.splitlines(keepends=True)

    JOB.mkdir(parents=True, exist_ok=False)
    notebook = JOB / "experiment.ipynb"
    notebook.write_text(json.dumps(source_notebook), encoding="utf-8")
    if notebook.stat().st_size >= 900_000:
        raise ValueError("Notebook exceeds known Kaggle save size envelope")
    metadata = json.loads((BASE / "kernel-metadata.json").read_text(encoding="utf-8"))
    metadata.pop("id_no", None)
    metadata.update(
        id="pamcks/h3-1mp-fp16-512-sol-v2-8nfe-10s-p0-20261102",
        title="H3 1MP FP16 512 Sol v2 8NFE 10s P0 20261102",
        code_file="experiment.ipynb",
    )
    assert metadata["is_private"] and metadata["enable_gpu"]
    assert metadata["machine_shape"] == "NvidiaTeslaT4"
    assert not metadata["dataset_sources"] and not metadata["kernel_sources"]
    (JOB / "kernel-metadata.json").write_text(json.dumps(metadata,indent=2),encoding="utf-8")
    binding = dict(
        baseline_source_binding=bind,
        baseline_commit=publication["commit"],
        overlay_hashes=hashes,
        closure=list(closure),
        overlay_sha256=packed_sha,
        plan=plan,
        attention_selection=selected,
        benchmark_debug=False,
        runtime_debug=False,
        expected_sol_per_rank=8*50,
        expected_kitchen_dit_per_rank=0,
        gpu_count=2,
        status_polling=False,
        live_log_polling=False,
        final_collection_count=0,
    )
    (JOB / "source_binding.json").write_text(json.dumps(binding,indent=2),encoding="utf-8")
    (EVIDENCE / "request.json").write_text(json.dumps(plan,indent=2),encoding="utf-8")
    (EVIDENCE / "build_receipt.json").write_text(json.dumps(dict(
        status="ready",kernel=metadata["id"],notebook_bytes=notebook.stat().st_size,
        notebook_sha256=hashlib.sha256(notebook.read_bytes()).hexdigest(),
        overlay_count=len(hashes),overlay_sha256=packed_sha,
        all_sol_nfe=selected,debug=False,residual_exponent=9,
    ),indent=2),encoding="utf-8")
    print(json.dumps(dict(status="built",kernel=metadata["id"],
        notebook_bytes=notebook.stat().st_size,overlay_count=len(hashes),
        gpu_count=2,sol_nfe=selected,debug=False,exponent=9),indent=2))


if __name__ == "__main__":
    main()
