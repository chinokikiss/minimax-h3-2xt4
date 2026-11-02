"""Build independent frozen CPU/T4 P0-A1/P0-C Kaggle notebooks.

The established Sol-v2 E2E capsule and setup are immutable references.
Overlay source-bound current production P0-2/P0-3 plus the isolated A1/C
harness. Never submit, poll Kaggle or alter a historical notebook here.
"""
from __future__ import annotations

import ast
import base64
import hashlib
import io
import json
import lzma
from pathlib import Path
import tarfile

from benchmarks.t4_notebook_watchdog import supervise_worker

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "docs/data/h3_t4_fp16_512_sol_all_8nfe_10s_20261101"
EVIDENCE = ROOT / "docs/data/h3_dit_p0ac_t4_20261102"
EXTRA = tuple("patches/" + name for name in (
    "tp_dit.py", "dit_column_linear.py", "tilelang_dit_fusion.py",
    "tilelang_transport_p2.py", "tilelang_row_quant.py", "tilelang_swiglu_quant.py"
)) + tuple("benchmarks/" + name for name in (
    "dit_p0ac_t4_candidates.py", "run_dit_p0ac_case.py",
    "run_dit_p0ac_batch.py", "test_dit_p0ac_t4_preflight.py",
    "test_dit_p0_source_contracts.py", "test_dit_p0_production_defaults.py"
))

SETUP_END = "assert receipt['status']=='ready_for_device_gate'\n"

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

def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()

def source_archive(names):
    archive_buffer = io.BytesIO()
    manifest = {}
    with tarfile.open(fileobj=archive_buffer, mode="w:") as archive:
        for name in names:
            rel = Path(name)
            if rel.is_absolute() or ".." in rel.parts:
                raise AssertionError("Invalid source path: " + name)
            content = (ROOT / rel).read_bytes()
            if name.endswith(".py"):
                ast.parse(content, filename=name)
            manifest[name] = {"sha256": sha(content), "bytes": len(content)}
            entry = tarfile.TarInfo(name)
            entry.size = len(content)
            entry.mtime = entry.uid = entry.gid = 0
            entry.uname = entry.gname = ""
            archive.addfile(entry, io.BytesIO(content))
    payload = lzma.compress(archive_buffer.getvalue(), preset=9 | lzma.PRESET_EXTREME)
    return payload, manifest

def build():
    previous = json.loads((BASE / "build_receipt_v2.json").read_text(encoding="utf-8"))
    baseline = (BASE / "gpu_job_v2/experiment.ipynb").read_bytes()
    assert sha(baseline) == previous["notebook_sha256"]
    notebook = json.loads(baseline)
    source = "".join(notebook["cells"][0]["source"])
    assert source.count(SETUP_END) == 1
    setup = source[:source.index(SETUP_END) + len(SETUP_END)]
    ast.parse(setup)
    e2e = json.loads((BASE / "gpu_job_v2/source_binding.json").read_text(encoding="utf-8"))
    assert e2e["plan"]["residual_exponent"] == 9
    assert e2e["plan"]["operators"]["dit_attention_schedule"] == "1-8:sol@1.0"
    closure = set(e2e["overlay_hashes"])
    assert "patches/h3_operator_policy.py" in closure
    assert "patches/sol_attention.py" in closure
    assert "patches/h3_tensor_flow.py" in closure
    names = sorted(closure | set(EXTRA))
    payload, manifest = source_archive(names)
    encoded = base64.b85encode(payload).decode("ascii")
    assert "patches/h3_operator_policy.py" in manifest
    assert "patches/sol_sm75_backend.py" in manifest
    EVIDENCE.mkdir(parents=True, exist_ok=False)
    for mode in ("cpu", "gpu"):
        out = EVIDENCE / ("cpu_job" if mode == "cpu" else "gpu_job")
        overlay = f'''import base64,hashlib,io,json,lzma,os,pathlib,sys,tarfile
source=pathlib.Path('/kaggle/temp/h3_light_source')
out=pathlib.Path('/kaggle/working/dit_p0ac')
out.mkdir(parents=True,exist_ok=True)
raw=base64.b85decode({encoded!r})
assert hashlib.sha256(raw).hexdigest()=={sha(payload)!r}
manifest={manifest!r}
retired={RETIRED!r}
audit={{'mode':{mode!r},'frozen_setup_notebook_sha256':{sha(baseline)!r},
       'previous_e2e_overlay_count':{len(closure)},'files':{{}}}}
with tarfile.open(fileobj=io.BytesIO(lzma.decompress(raw)),mode='r:') as archive:
    members=archive.getmembers()
    assert len(members)==len(manifest)=={len(names)}
    for entry in members:
        rel=pathlib.PurePosixPath(entry.name)
        assert entry.isfile() and not rel.is_absolute() and '..' not in rel.parts
        name=str(rel)
        record=manifest[name]
        with archive.extractfile(entry) as stream:content=stream.read()
        assert len(content)==record['bytes'] and hashlib.sha256(content).hexdigest()==record['sha256']
        target=source.joinpath(*rel.parts)
        old=target.read_bytes() if target.is_file() else b''
        target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes(content)
        audit['files'][name]=dict(previous_sha256=hashlib.sha256(old).hexdigest() if old else None,
            overlaid_sha256=record['sha256'],bytes=len(content))
for name in retired:
    target=source/name
    if target.exists():target.unlink()
assert 'sol' in source.joinpath('patches/h3_operator_policy.py').read_text()
(out/'overlay_receipt.json').write_text(json.dumps(audit,indent=2))
env=dict(os.environ)
env.update(json.loads(pathlib.Path('/kaggle/working/light_setup/execution_env.json').read_text()))
argv=[sys.executable,'-u',str(source/'benchmarks/run_dit_p0ac_batch.py'),
      '--source',str(source),'--output',str(out)]
if {mode=='cpu'!r}:argv+=['--cpu-preflight']
os.execve(sys.executable,argv,env)
'''
        ast.parse(overlay)
        code = setup + supervise_worker(
            overlay, budget=240 if mode == "cpu" else 2600,
            output="/kaggle/working/dit_p0ac_supervision"
        )
        expected = "cpu_ready" if mode == "cpu" else "complete"
        code += (f"\nassert json.loads(pathlib.Path('/kaggle/working/dit_p0ac/result.json').read_text())"
                 f"['status']=={expected!r}\n")
        ast.parse(code)
        candidate = json.loads(baseline)
        candidate["cells"][0]["source"] = code.splitlines(keepends=True)
        notebook_bytes = json.dumps(candidate).encode()
        assert len(notebook_bytes) < 900000, (mode, len(notebook_bytes))
        metadata = dict(
            id="pamcks/h3-dit-p0ac-" + ("cpu" if mode == "cpu" else "t4") + "-20261102",
            title="H3 DiT P0-A1 P0-C " + ("CPU" if mode == "cpu" else "T4") + " 20261102",
            code_file="experiment.ipynb", language="python", kernel_type="notebook",
            is_private=True, enable_gpu=mode == "gpu", enable_tpu=False,
            enable_internet=True, docker_image_pinning_type="original",
            dataset_sources=[], kernel_sources=[], competition_sources=[], model_sources=[]
        )
        if mode == "gpu":
            metadata["machine_shape"] = "NvidiaTeslaT4"
        out.mkdir()
        (out / "experiment.ipynb").write_bytes(notebook_bytes)
        (out / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        (out / "source_binding.json").write_text(json.dumps(dict(
            baseline_frozen_setup_sha256=sha(baseline), mode=mode,
            proven_e2e_overlay_keys=sorted(closure),
            source_overlay_sha256=sha(payload), source_overlay_manifest=manifest,
            all_production_operator_paths_covered=True,
            full_pipeline=False, benchmark_only=True, residual_exponent=9,
            rows=[59134,65166], tests=["baseline","p0_a1","p0_c","p0_a1_p0_c"],
            no_status_polling=True, no_live_log_polling=True,
            submission_count=0
        ), indent=2), encoding="utf-8")
        print(mode, "notebook_bytes", len(notebook_bytes),
              "notebook_sha256", sha(notebook_bytes), flush=True)
    (EVIDENCE / "build_manifest.json").write_text(json.dumps(dict(
        status="prepared_not_submitted",
        study="P0-A1 Kitchen unused output, P0-C O-Proj local rotation",
        source_files=len(manifest), previous_e2e_overlay_files=len(closure),
        source_overlay_sha256=sha(payload), compressed_bytes=len(payload),
        source_overlay_manifest=manifest, retired_sources=list(RETIRED), environment="native Torch, Kitchen 0.2.37, TileLang 0.1.13",
        no_status_polling=True, no_live_log_polling=True, user_authorized_new_t4_validation=True
    ), indent=2), encoding="utf-8")

if __name__ == "__main__":
    build()
