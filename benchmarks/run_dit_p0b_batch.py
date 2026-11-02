"""Frozen-capsule CPU preflight or bounded T4 P0-B A/B study."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import traceback
import unittest


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--source",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--cpu-preflight",action="store_true")
    args=parser.parse_args()
    source=args.source.resolve()
    out=args.output.resolve()
    out.mkdir(parents=True,exist_ok=True)
    sys.path[:0]=[str(source),str(source/"ComfyUI"),str(source/"benchmarks")]
    import comfy.cli_args
    if args.cpu_preflight:comfy.cli_args.args.cpu=True
    from patches.legacy_runtime import validate_packages,validate_source
    validate_packages()
    validate_source(source/"ComfyUI")
    receipt=dict(status="starting",mode="cpu" if args.cpu_preflight else "gpu",
                 no_lora=True,no_sample=True,device_setup_only=False,full_pipeline=False)
    def save():(out/"result.json").write_text(json.dumps(receipt,indent=2))
    try:
        if args.cpu_preflight:
            from unittest import defaultTestLoader,TextTestRunner
            suite=unittest.TestSuite([defaultTestLoader.loadTestsFromName(n) for n in (
                "benchmarks.test_dit_p0_source_contracts",
                "benchmarks.test_dit_p0_production_defaults",
                "benchmarks.test_dit_p0_ac_dispatch",
                "benchmarks.test_dit_p0b_t4_preflight")])
            result=TextTestRunner(verbosity=2).run(suite)
            if not result.wasSuccessful():raise AssertionError("P0 CPU qualification failed")
            receipt.update(status="cpu_ready",cpu_tests=result.testsRun,errors=0,failed=0)
            return
        import torch
        if torch.cuda.device_count()!=2:raise RuntimeError("Two T4 devices required")
        devices=[dict(name=torch.cuda.get_device_name(i),
                      capability=list(torch.cuda.get_device_capability(i))) for i in range(2)]
        if any("T4" not in row["name"] or row["capability"]!=[7,5] for row in devices):
            raise RuntimeError("SM75 T4 fingerprint mismatch")
        receipt.update(hardware=dict(devices=devices,torch=str(torch.__version__),
                                     cuda=torch.version.cuda),status="weight_download")
        assets=json.loads((source/"bundle_assets.json").read_text())
        objects={name:row for name,row in assets["objects"].items()
                 if name in ("dit.block00.rank0.safetensors","dit.block00.rank1.safetensors")}
        if len(objects)!=2:raise RuntimeError("Missing frozen trained rank0/rank1 shards")
        from concurrent.futures import ThreadPoolExecutor
        from benchmarks.tp2_public_dit_assets import download
        weights=Path("/tmp/h3_p0b_trained_shards_20261102")
        weights.mkdir(exist_ok=False)
        def fetch(item):
            name,row=item
            return download(dict(file=name,repository=assets["publication"]["repo"],
                revision=assets["publication"]["commit"],**row),weights)
        with ThreadPoolExecutor(max_workers=2) as pool:list(pool.map(fetch,objects.items()))
        receipt["weight_download_bytes"]=sum(row["bytes"] for row in objects.values())
        receipt["status"]="t4_running";save()
        nvcc=shutil.which("nvcc")
        if not nvcc:raise RuntimeError("Frozen T4 setup lacks nvcc")
        env=dict(os.environ,OMP_NUM_THREADS="2",MKL_NUM_THREADS="2",OPENBLAS_NUM_THREADS="2",
                 NCCL_P2P_DISABLE="1",NCCL_DEBUG="WARN",
                 NCCL_MIN_NCHANNELS="4",NCCL_MAX_NCHANNELS="4",
                 CUDACXX=nvcc,CUDA_HOME=str(Path(nvcc).resolve().parent.parent),
                 TILELANG_CACHE_DIR="/kaggle/temp/h3_dit_p0b_cache")
        argv=[sys.executable,"-m","torch.distributed.run","--standalone","--nproc_per_node=2",
              str(source/"benchmarks/run_dit_p0b_case.py"),
              "--source",str(source),"--weights",str(weights),"--output",str(out/"tp2")]
        t0=time.monotonic()
        with (out/"worker.log").open("w") as log:
            process=subprocess.Popen(argv,cwd=source,env=env,stdout=log,
                                     stderr=subprocess.STDOUT,start_new_session=True)
            try:code=process.wait(timeout=2100)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid,signal.SIGTERM)
                try:process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid,signal.SIGKILL);process.wait(timeout=5)
                code=-9
        receipt.update(worker_exit_code=code,seconds=time.monotonic()-t0)
        if code:raise RuntimeError("TP2 P0 worker failed; inspect retained per-rank evidence")
        ranks=[json.loads((out/"tp2"/f"rank{r}"/"result.json").read_text()) for r in (0,1)]
        complete=all(r["status"]=="complete" for r in ranks)
        precision=all(v['all_bit_exact'] for r in ranks for v in r['per_geometry'])
        receipt.update(status="complete" if complete else "partial",exact_qualified=precision,
                       p0b_intermediate_qualified=all(
                           all(p['all_bit_exact'] for p in r['intermediate']) for r in ranks),
                       ran_shapes=[59134,65166],rank_receipts=[str(out/"tp2"/f"rank{r}"/"result.json") for r in (0,1)])
    except BaseException as e:
        receipt.update(status="failed",error=repr(e))
        (out/"exception.txt").write_text(traceback.format_exc())
        raise
    finally:
        save()
        print("DIT_P0B_BATCH_FINAL",receipt["status"],flush=True)


if __name__=="__main__":main()
