"""Build source-hashed private CPU/T4 notebooks from the frozen 1/512 setup.

No dependency upgrades, live Git ref, Kaggle status call, GPU submission,
or modification of an existing E2E notebook. The experiment overlays six P0
candidate sources and four independent test/runner files AFTER frozen setup.
"""
from __future__ import annotations

import ast
import base64
import hashlib
import json
from pathlib import Path

from benchmarks.t4_notebook_watchdog import supervise_worker

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/"docs/data/h3_t4_fp16_512_sol_all_8nfe_10s_20261101"
EVIDENCE=ROOT/"docs/data/h3_dit_p0_lifetime_20261102_v2"
PATHS=tuple("patches/"+n for n in (
    "tp_dit.py","dit_column_linear.py","tilelang_dit_fusion.py",
    "tilelang_transport_p2.py","tilelang_row_quant.py","tilelang_swiglu_quant.py"))+tuple(
    "benchmarks/"+n for n in (
    "test_dit_p0_local.py","test_dit_p0_source_contracts.py",
    "run_dit_p0_case.py","run_dit_p0_batch.py"))
SETUP_END="assert receipt['status']=='ready_for_device_gate'\n"


def sha(b:bytes)->str:return hashlib.sha256(b).hexdigest()


def build():
    previous=json.loads((BASE/"build_receipt_v2.json").read_text())
    frozen=(BASE/"gpu_job_v2/experiment.ipynb").read_bytes()
    assert sha(frozen)==previous["notebook_sha256"],"Frozen source mismatch"
    notebook=json.loads(frozen)
    source="".join(notebook["cells"][0]["source"])
    assert source.count(SETUP_END)==1
    setup=source[:source.index(SETUP_END)+len(SETUP_END)]
    ast.parse(setup)
    binding=json.loads((BASE/"gpu_job_v2/source_binding.json").read_text())
    assert binding["plan"]["residual_exponent"]==9
    assert binding["plan"]["operators"]["dit_attention_schedule"]=="1-8:sol@1.0"
    files={}
    for name in PATHS:
        raw=(ROOT/name).read_bytes()
        ast.parse(raw,filename=name)
        files[name]=dict(sha256=sha(raw),bytes=len(raw),payload=base64.b64encode(raw).decode())
    provenance={name:dict(sha256=row["sha256"],bytes=row["bytes"]) for name,row in files.items()}
    EVIDENCE.mkdir(parents=True,exist_ok=True)
    for mode in ("cpu","gpu"):
        dest=EVIDENCE/("cpu_job" if mode=="cpu" else "gpu_job")
        if dest.exists():
            raise FileExistsError("Refusing to overwrite a prepared Kaggle job: "+str(dest))
        overlay=f"""import base64,hashlib,json,os,pathlib,sys
source=pathlib.Path('/kaggle/temp/h3_light_source')
out=pathlib.Path('/kaggle/working/dit_p0')
out.mkdir(parents=True,exist_ok=True)
files={repr(files)}
audit={{'mode':{mode!r},'frozen_setup_notebook_sha256':{sha(frozen)!r},'files':{{}}}}
for name,record in files.items():
    target=source/name
    old=target.read_bytes() if target.is_file() else b''
    raw=base64.b64decode(record['payload'],validate=True)
    if len(raw)!=record['bytes'] or hashlib.sha256(raw).hexdigest()!=record['sha256']:
        raise RuntimeError('P0 overlay integrity mismatch: '+name)
    target.parent.mkdir(parents=True,exist_ok=True)
    target.write_bytes(raw)
    audit['files'][name]=dict(previous_sha256=hashlib.sha256(old).hexdigest() if old else None,
                              overlaid_sha256=record['sha256'],bytes=len(raw))
(out/'overlay_receipt.json').write_text(json.dumps(audit,indent=2))
env=dict(os.environ)
env.update(json.loads(pathlib.Path('/kaggle/working/light_setup/execution_env.json').read_text()))
argv=[sys.executable,'-u',str(source/'benchmarks/run_dit_p0_batch.py'),
      '--source',str(source),'--output',str(out)]
if {mode=='cpu'!r}:argv+=['--cpu-preflight']
os.execve(sys.executable,argv,env)
"""
        ast.parse(overlay)
        budget=240 if mode=="cpu" else 2600
        code=setup+supervise_worker(overlay,budget=budget,
            output="/kaggle/working/dit_p0_supervision")
        expected="cpu_ready" if mode=="cpu" else "complete"
        code+=f"\nassert json.loads(pathlib.Path('/kaggle/working/dit_p0/result.json').read_text())['status']=={expected!r}\n"
        ast.parse(code)
        candidate=json.loads(frozen)
        candidate["cells"][0]["source"]=code.splitlines(keepends=True)
        encoded=json.dumps(candidate).encode()
        assert len(encoded)<900000
        meta=dict(id="pamcks/h3-dit-p0-lifetime-"+("cpu" if mode=="cpu" else "t4")+"-v2-20261102",
                  title="H3 DiT P0 Lifetime "+("CPU" if mode=="cpu" else "T4")+" V2 20261102",
                  code_file="experiment.ipynb",language="python",kernel_type="notebook",
                  is_private=True,enable_gpu=mode=="gpu",enable_tpu=False,enable_internet=True,
                  docker_image_pinning_type="original",dataset_sources=[],kernel_sources=[],
                  competition_sources=[],model_sources=[])
        if mode=="gpu":meta["machine_shape"]="NvidiaTeslaT4"
        dest.mkdir(parents=True)
        (dest/"experiment.ipynb").write_bytes(encoded)
        (dest/"kernel-metadata.json").write_text(json.dumps(meta,indent=2))
        (dest/"source_binding.json").write_text(json.dumps(dict(
            frozen_notebook_sha256=sha(frozen),
            frozen_publication_sha256=binding["baseline_source_binding"]["publication"]["sha256"],
            frozen_capsule_sha256=binding["baseline_source_binding"]["publication"]["capsule_sha256"],
            mode=mode,full_pipeline=False,benchmark_only=True,tests=["p0_3","p0_2","p0_1","all_p0"],
            residual_exponent=9,rows=[59134,65166],source_overlay=provenance,
            kaggle_mounts=0,source_hashes_verified=True,
            no_production_deployment=True),indent=2))
        print(mode,"notebook_bytes",len(encoded),"sha256",sha(encoded))
    (EVIDENCE/"build_manifest.json").write_text(json.dumps(dict(
        status="prepared",baseline_notebook_sha256=sha(frozen),
        source_overlay=provenance,environment="frozen Kaggle-native Torch, Kitchen 0.2.37, TileLang 0.1.13",
        local_conda="C:/ProgramData/miniconda3/envs/comfyui/python.exe",
        cpu_before_gpu=True,no_status_polling=True,no_live_log_polling=True),indent=2))


if __name__=="__main__":build()
