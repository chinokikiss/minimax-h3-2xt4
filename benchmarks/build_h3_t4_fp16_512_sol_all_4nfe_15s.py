"""Build one 15s dual-T4 FP16/512, debug-off, Sol-v2 all-NFE E2E.

Reuses the original 2026-10-24 0.5MP Turbo source publication and CPU
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
EVIDENCE = ROOT / "docs/data/h3_t4_fp16_512_sol_all_4nfe_15s_20261101"
BASE = EVIDENCE / "base_job"
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
    "benchmarks/ref2va_e2e_run.py",
    "benchmarks/run_h3_l1_turbo_e2e.py",
    "benchmarks/run_h3_l1_turbo_batch.py",
    "benchmarks/e2e_attention_validation.py",
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
    previous = json.loads((EVIDENCE / "baseline_15s_request.json").read_text(encoding="utf-8"))
    assert publication["commit"] == "11cec15df6c3e3b7c03c482f78b831daa62b20a3"
    assert publication["published"] and bind["mode"] == "e2e" and bind["turbo_e2e"]
    assert (previous["width"], previous["height"], previous["frames"], previous["fps"]) == (960, 544, 362, 24)
    assert (previous["seed"], previous["steps"], previous["sampler"], previous["scheduler"]) == (20261020, 4, "lcm", "simple")
    assert (previous["video_shift"], previous["audio_shift"], previous["cfg"]) == (12, 3, 1)
    assert previous["offline_lora"] and previous["disk_baked_lora"]
    assert previous["turbo_lora"]["strength"] == 1
    assert previous["turbo_lora"]["file"] == "minimax_h3_ref2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors"

    plan = dict(previous)
    plan.update(
        name="H3_FP16_512_SOL_V2_ALL_4_NFE_15S_DEBUG_OFF",
        debug=False,
        residual_exponent=9,
        operators=dict(
            te_attention="kitchen_causal",
            dit_attention="sol",
            dit_attention_schedule="1-4:sol@1.0",
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
            "Exact previously completed 960x544/362f, seed-20261020, 4-step LCM/Simple "
            "Turbo workload; current FP16 residual 1/512, explicit debug=False, "
            "Sol v2 tau=1.0 on every NFE1-4. Original frozen TE, input references, "
            "Turbo LoRA, shifts, and decoder semantics retained."
        ),
    )
    assert all(plan[key] == previous[key] for key in previous
               if key not in ("name", "source_note", "operators"))
    schedule = parse_attention_schedule(plan["operators"]["dit_attention_schedule"])
    selected = [select_attention_policy(schedule, n, "sol", 1.0) for n in range(1, 5)]
    assert selected == [("sol", 1.0)] * 4, selected

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
out=pathlib.Path('/kaggle/working/h3_t4_fp16_512_sol_v2_all_4nfe_15s');out.mkdir()
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
    overlay_hashes=HASHES,scope='FP16/512 debug-off all-four-NFE Sol-v2 15s Turbo E2E'),indent=2))
argv=[sys.executable,'-u',str(source/'benchmarks/run_h3_l1_turbo_batch.py'),
    '--source',str(source),'--output',str(out),'--plan',str(out/'request.json'),
    '--scratch','/tmp/h3_t4_fp16_512_sol_v2_15s_20261101','--e2e-timeout','5400']
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
    code = prefix + supervise_worker(execution, budget=7200,
        output="/kaggle/working/h3_t4_fp16_512_sol_v2_15s_supervision")
    code += ("assert json.loads(pathlib.Path('/kaggle/working/"
        "h3_t4_fp16_512_sol_v2_all_4nfe_15s/batch_results.json').read_text())"
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
        id="pamcks/h3-fp16-512-sol-v2-all-4nfe-15s-debug-off",
        title="H3 15s FP16 512 All Sol v2 Debug Off",
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
        expected_sol_per_rank=4*50,
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
