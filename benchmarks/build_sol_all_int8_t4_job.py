"""Build and optionally submit the isolated Sol All-INT8 versus Kitchen T4 benchmark."""
import argparse
import ast
import base64
import datetime as dt
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

from t4_notebook_watchdog import supervise_worker

ROOT = Path(__file__).resolve().parents[1]


def experiment_files():
    base = ROOT / "experiments" / "sol_all_int8"
    return tuple(
        p.relative_to(ROOT).as_posix()
        for p in sorted(base.rglob("*"))
        if p.is_file() and "__pycache__" not in p.parts
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--publication", type=Path, required=True)
    p.add_argument("--qualification", type=Path, required=True)
    p.add_argument("--build-receipt", type=Path, required=True)
    p.add_argument("--evidence", type=Path, required=True)
    p.add_argument("--job-name", default="gpu_job_v2")
    p.add_argument("--min-length", type=int, default=0)
    p.add_argument("--submit", action="store_true")
    args = p.parse_args()

    pub = json.loads(args.publication.read_text(encoding="utf-8-sig"))
    qualification = json.loads(args.qualification.read_text(encoding="utf-8-sig"))
    build_receipt = json.loads(args.build_receipt.read_text(encoding="utf-8-sig"))
    assert pub["published"] and qualification["status"] == "cpu_ready"
    assert qualification["capsule_sha256"] == pub["capsule_sha256"]
    assert qualification["core_pins"]["comfy-kitchen"] == "0.2.37"
    assert build_receipt["status"] == "passed" and build_receipt["target"] == "sm_75"

    args.evidence.mkdir(parents=True, exist_ok=True)
    job = args.evidence / args.job_name
    if job.exists():
        raise FileExistsError("Preserve an existing prepared/submitted job")
    job.mkdir()

    payload = {}
    hashes = {}
    for name in experiment_files():
        raw = (ROOT / name).read_bytes()
        payload[name] = base64.b64encode(raw).decode()
        hashes[name] = hashlib.sha256(raw).hexdigest()

    local_sources = build_receipt["sources"]
    for name, digest in local_sources.items():
        if name == "STAR7_LICENSE":
            project_name = "experiments/sol_all_int8/STAR7_LICENSE"
        else:
            project_name = "experiments/sol_all_int8/" + name
        assert hashes[project_name] == digest, project_name

    worker = r'''import base64,hashlib,io,json,os,pathlib,shutil,statistics,subprocess,sys,tarfile,urllib.request,time
PUB=PUBLICATION
QUAL=QUALIFICATION
FILES=EMBEDDED_FILES
HASHES=EMBEDDED_HASHES
EXPECTED_NATIVE_SOURCES=EXPECTED_SOURCES
MIN_LENGTH=MINIMUM_LENGTH
OUT=pathlib.Path('/kaggle/working/sol_all_int8');OUT.mkdir(parents=True,exist_ok=True)
SETUP=pathlib.Path('/kaggle/working/sol_setup');SETUP.mkdir(parents=True,exist_ok=True)
DIST=pathlib.Path('/kaggle/temp/h3_sol_distribution');DIST.mkdir(parents=True,exist_ok=True)
SRC=pathlib.Path('/kaggle/temp/h3_sol_source');SRC.mkdir(parents=True,exist_ok=True)
print('SOL_BENCH source_download',flush=True)
with urllib.request.urlopen(PUB['url'],timeout=90) as response:
    raw=response.read(PUB['bytes']+1)
assert len(raw)==PUB['bytes'] and hashlib.sha256(raw).hexdigest()==PUB['sha256']
with tarfile.open(fileobj=io.BytesIO(raw),mode='r:gz') as archive:archive.extractall(DIST,filter='data')
capsule=(DIST/'capsule.tar.gz.bin').read_bytes()
assert hashlib.sha256(capsule).hexdigest()==PUB['capsule_sha256']==QUAL['capsule_sha256']
with tarfile.open(fileobj=io.BytesIO(capsule),mode='r:gz') as archive:archive.extractall(SRC,filter='data')
(SETUP/'cpu_qualification.json').write_text(json.dumps(QUAL))
print('SOL_BENCH frozen_setup',flush=True)
subprocess.run([sys.executable,'-u',str(SRC/'benchmarks/prepare_kaggle_direct_pip.py'),
    '--distribution',str(DIST),'--source',str(SRC),'--output',str(SETUP),
    '--mode','operators','--qualification',str(SETUP/'cpu_qualification.json')],
    check=True,timeout=600)
setup=json.loads((SETUP/'setup_receipt.json').read_text())
assert setup['status']=='ready_for_device_gate'
for name,encoded in FILES.items():
    raw=base64.b64decode(encoded)
    assert hashlib.sha256(raw).hexdigest()==HASHES[name]
    target=SRC/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(raw)
env=dict(os.environ);env.update(json.loads((SETUP/'execution_env.json').read_text()))
env['PYTHONPATH']=str(SRC)+os.pathsep+env['PYTHONPATH']
nvcc=shutil.which('nvcc')
assert nvcc, 'Kaggle T4 image must provide nvcc'
version=subprocess.check_output([nvcc,'--version'],text=True)
assert 'V12.8.93' in version,version
print('SOL_BENCH native_build',flush=True)
build_dir=pathlib.Path('/kaggle/temp/h3_sol_native_build')
subprocess.run([sys.executable,'-u','-m','experiments.sol_all_int8.build',
    '--output',str(build_dir),'--nvcc',nvcc],cwd=SRC,env=env,check=True,timeout=360)
receipt=json.loads((build_dir/'build_receipt.json').read_text())
assert receipt['status']=='passed' and receipt['target']=='sm_75'
assert receipt['sources']==EXPECTED_NATIVE_SOURCES
LIB=build_dir/'h3_sol_all_int8_sm75.so'
env['H3_SOL_ALL_INT8_LIBRARY']=str(LIB)
print('SOL_BENCH launch_dual_t4',flush=True)
procs=[]
for rank in (0,1):
    path=OUT/f'rank{rank}.json'
    cmd=[sys.executable,'-u','-m','experiments.sol_all_int8.benchmark',
         '--output',str(path),'--library',str(LIB),'--rank',str(rank),
         '--min-length',str(MIN_LENGTH)]
    log=(OUT/f'rank{rank}.log').open('w')
    procs.append((subprocess.Popen(cmd,cwd=SRC,env=env,stdout=log,stderr=subprocess.STDOUT),log))
deadline=time.monotonic()+1500
codes=[]
for proc,log in procs:
    remaining=max(1,deadline-time.monotonic())
    try: codes.append(proc.wait(timeout=remaining))
    finally: log.close()
assert codes==[0,0],codes
ranks=[json.loads((OUT/f'rank{rank}.json').read_text()) for rank in (0,1)]
assert all(r['status']=='completed' for r in ranks)
rows=[]
for length in ranks[0]['lengths']:
    per=[]
    for rank,result in enumerate(ranks):
        case=next(c for c in result['cases'] if c['length']==length)
        s=case['summary']
        sol_rounds=[x for x in case['rounds'] if x['name']=='sol_all_int8_tau1']
        ck_rounds=[x for x in case['rounds'] if x['name']=='ck_int8_dense']
        per.append(dict(rank=rank,ck_median_ms=s['ck_median_ms'],sol_median_ms=s['sol_median_ms'],
            speedup_vs_ck=s['speedup_vs_ck'],
            route_density=statistics.median([x['route_density_median'] for x in sol_rounds]),
            ck_peak_temporary=max(x['peak_temporary'] for x in ck_rounds),
            sol_peak_temporary=max(x['peak_temporary'] for x in sol_rounds)))
    rows.append(dict(length=length,ranks=per,
        ck_median_ms=statistics.median([x['ck_median_ms'] for x in per]),
        sol_median_ms=statistics.median([x['sol_median_ms'] for x in per]),
        speedup_vs_ck=statistics.median([x['speedup_vs_ck'] for x in per]),
        route_density=statistics.median([x['route_density'] for x in per])))
summary=dict(status='completed',gpu_count=2,gpus=[r['gpu'] for r in ranks],
    comparison='Complete raw-input FP16 BHND calls: Sol All-INT8 tau=1.0 vs comfy-kitchen 0.2.37 dense INT8',
    heads=28,head_dim=128,rows=rows,source_hashes=HASHES,
    native_build=dict(compiler_version=receipt['compiler_version'],
        artifact_sha256=receipt['artifact_sha256'],artifact_bytes=receipt['artifact_bytes']),
    limitations=['Synthetic operator inputs; no H3 generation-quality claim.',
                 'Sol includes routing, centroid construction and Q/K/V quantization.',
                 'Results are opt-in evidence only; production dispatch is unchanged.'])
(OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print('SOL_BENCH completed',flush=True)
'''.replace("PUBLICATION",repr(pub)).replace("QUALIFICATION",repr(qualification)).replace(
    "EMBEDDED_FILES",repr(payload)).replace("EMBEDDED_HASHES",repr(hashes)).replace(
    "EXPECTED_SOURCES",repr(local_sources)).replace("MINIMUM_LENGTH",repr(args.min_length))

    code = supervise_worker(worker, budget=2100, output="/kaggle/working/sol_supervision")
    code += """
guard=json.loads(pathlib.Path('/kaggle/working/sol_supervision/supervision.json').read_text())
assert guard['status']=='worker_exited' and guard['exit_code']==0,guard
assert json.loads(pathlib.Path('/kaggle/working/sol_all_int8/summary.json').read_text())['status']=='completed'
"""
    ast.parse(code)
    notebook = dict(nbformat=4,nbformat_minor=5,
        metadata=dict(kernelspec=dict(name="python3",language="python",display_name="Python 3")),
        cells=[dict(id="sol-all-int8",cell_type="code",execution_count=None,metadata={},outputs=[],
                    source=code.splitlines(keepends=True))])
    metadata = dict(id="pamcks/h3-sol-all-int8-t4-20261030",
        title="H3 Sol All INT8 T4 20261030",code_file="experiment.ipynb",language="python",
        kernel_type="notebook",is_private=True,enable_gpu=True,enable_tpu=False,
        enable_internet=True,machine_shape="NvidiaTeslaT4",docker_image_pinning_type="original",
        dataset_sources=[],kernel_sources=[],competition_sources=[],model_sources=[])
    (job/"experiment.ipynb").write_text(json.dumps(notebook),encoding="utf-8")
    (job/"kernel-metadata.json").write_text(json.dumps(metadata,indent=2),encoding="utf-8")
    binding = dict(publication=pub,
        qualification_sha256=hashlib.sha256(args.qualification.read_bytes()).hexdigest(),
        runtime_sources=hashes,native_source_manifest=local_sources,
        local_compile_reference=build_receipt,
        benchmark_scope="isolated operator benchmark only; no production dispatch changes",
        min_length=args.min_length,kaggle_native_compile=True,gpu_count=2,zero_mounts=True,
        status_polling=False,live_log_polling=False,final_collection_count=0)
    (job/"source_binding.json").write_text(json.dumps(binding,indent=2),encoding="utf-8")
    print(json.dumps(dict(status="built",job=str(job),kernel=metadata["id"],
                          notebook_bytes=(job/"experiment.ipynb").stat().st_size),indent=2))
    if not args.submit:
        return
    receipt_path=args.evidence/"gpu_submission_v2.json"
    state=dict(kernel=metadata["id"],status="submitting",
        submitted_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        source_binding=binding,run_timeout_seconds=2400,final_collection_count=0,
        result_relative="sol_all_int8/summary.json")
    with receipt_path.open("x",encoding="utf-8") as f:json.dump(state,f,indent=2)
    proc=subprocess.run([sys.executable,str(ROOT/"benchmarks/kaggle_tls12_runner.py"),
        "--cli","kernels","push","-p",str(job)],capture_output=True,text=True,timeout=180)
    output=proc.stdout+proc.stderr
    state.update(returncode=proc.returncode,response=output,
                 status="accepted" if proc.returncode==0 else "acceptance_uncertain")
    m=re.search(r"Kernel version (\d+)",output)
    if m:state["version"]=int(m.group(1))
    u=re.search(r"https://www\.kaggle\.com/code/[\w-]+/[\w-]+",output)
    if u:state["url"]=u.group(0)
    receipt_path.write_text(json.dumps(state,indent=2)+"\n",encoding="utf-8")
    (args.evidence/"gpu_submission_v2_cli.log").write_text(output,encoding="utf-8")
    print(output[-2000:])
    if proc.returncode:
        raise RuntimeError("Single corrected Kaggle submission attempt failed; no automatic retry")


if __name__=="__main__":
    main()
