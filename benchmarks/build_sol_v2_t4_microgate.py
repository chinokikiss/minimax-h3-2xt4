"""Build/submit a zero-weight physical-T4 Sol v2 micro-gate."""
import argparse,ast,base64,datetime as dt,hashlib,io,json,lzma,math,re,subprocess,sys,tarfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
FILES=[
    "benchmarks/sol_v2_t4_microgate.py",
    "benchmarks/sol_bitmap_route_probe.py",
    "benchmarks/sol_workspace_pool_probe.py",
    "patches/sol_attention.py",
    "patches/sol_sm75_backend.py",
]
NATIVE=ROOT/"patches"/"sol_attention_native"
FILES += [p.relative_to(ROOT).as_posix() for p in sorted(NATIVE.rglob("*")) if p.is_file()]

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--evidence",type=Path,required=True);ap.add_argument("--submit",action="store_true")
    ap.add_argument("--workspace-ab",action="store_true",help="same-instance baseline/workspace-pool A/B, no model weights")
    ap.add_argument("--bitmap-ab",action="store_true",help="same-instance ordered-int32/bitmap LUT A/B")
    args=ap.parse_args()
    if sum([args.workspace_ab,args.bitmap_ab])>1:
        raise ValueError("--workspace-ab and --bitmap-ab are separate gates")
    args.evidence.mkdir(parents=True,exist_ok=True)
    job=args.evidence/"gpu_job"
    if job.exists():raise FileExistsError(job)
    job.mkdir()
    hashes={};buf=io.BytesIO()
    with tarfile.open(fileobj=buf,mode="w:") as tf:
        for name in FILES:
            raw=(ROOT/name).read_bytes();hashes[name]=hashlib.sha256(raw).hexdigest()
            info=tarfile.TarInfo(name);info.size=len(raw);info.mtime=0;info.uid=info.gid=0;info.uname=info.gname=""
            tf.addfile(info,io.BytesIO(raw))
    packed=lzma.compress(buf.getvalue(),preset=9|lzma.PRESET_EXTREME)
    encoded=base64.b85encode(packed).decode()
    worker=r'''import base64,hashlib,io,json,lzma,math,os,pathlib,subprocess,sys,tarfile,time
ROOT=pathlib.Path('/kaggle/working/sol_v2_micro_source');ROOT.mkdir()
raw=lzma.decompress(base64.b85decode(PAYLOAD))
with tarfile.open(fileobj=io.BytesIO(raw),mode='r:') as tf:
    for member in tf.getmembers():
        rel=pathlib.PurePosixPath(member.name)
        assert member.isfile() and not rel.is_absolute() and '..' not in rel.parts
        target=ROOT.joinpath(*rel.parts);target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes(tf.extractfile(member).read())
for name,expected in HASHES.items():
    path=ROOT/name
    assert hashlib.sha256(path.read_bytes()).hexdigest()==expected,(name,'hash mismatch')
import torch,shutil
assert torch.cuda.is_available()
assert torch.cuda.get_device_capability(0)==(7,5),(torch.cuda.get_device_name(0),torch.cuda.get_device_capability(0))
nvcc=shutil.which('nvcc');assert nvcc
version=subprocess.check_output([nvcc,'--version'],text=True)
assert 'release 12.8' in version,version
out=pathlib.Path('/kaggle/working/sol_v2_microgate');out.mkdir()
env=dict(os.environ);env['PYTHONPATH']=str(ROOT)
# A/B default control is deterministic even if the notebook inherited flags.
env['H3_SOL_WORKSPACE_POOL']='0'
env['H3_SOL_LUT_BITMAP']='0'
cmd=[sys.executable,'-u',str(ROOT/'benchmarks/sol_v2_t4_microgate.py'),'--output',str(out/'result.json')]
begin=time.monotonic()
proc=subprocess.run(cmd,cwd=ROOT,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=900)
(out/'stdout.log').write_text(proc.stdout)
summary=dict(exit_code=proc.returncode,seconds=time.monotonic()-begin,nvcc=version.splitlines()[-1],
             torch=torch.__version__,gpu=torch.cuda.get_device_name(0),source_hashes=HASHES)
(out/'runner.json').write_text(json.dumps(summary,indent=2))
print(proc.stdout[-12000:],flush=True)
assert proc.returncode==0,summary
result=json.loads((out/'result.json').read_text())
assert result['status']=='complete',result
print('SOL_V2_MICROGATE_PASS',json.dumps(result['acceptance'],sort_keys=True),flush=True)
if __RUN_BITMAP_AB__:
    import torch.nn.functional as F
    cases={}
    for mode,enabled in (('baseline','0'),('bitmap','1')):
        variant_env=dict(env)
        variant_env['H3_SOL_WORKSPACE_POOL']='0'
        variant_env['H3_SOL_LUT_BITMAP']=enabled
        cmd=[sys.executable,'-u',str(ROOT/'benchmarks/sol_bitmap_route_probe.py'),
             '--out',str(out/('bitmap_'+mode))]
        proc=subprocess.run(cmd,cwd=ROOT,env=variant_env,stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT,text=True,timeout=1200)
        (out/('bitmap_'+mode+'_stdout.log')).write_text(proc.stdout)
        print('bitmap_'+mode,proc.stdout[-3000:],flush=True)
        assert proc.returncode==0,(mode,proc.returncode,proc.stdout[-5000:])
        cases[mode]=json.loads((out/('bitmap_'+mode)/'result.json').read_text())
        assert cases[mode]['abi']==12
    comparison={}
    for name in ('d64_vae','real_shape_d128'):
        a=torch.load(out/'bitmap_baseline'/(name+'.pt'),map_location='cpu',weights_only=True)
        b=torch.load(out/'bitmap_bitmap'/(name+'.pt'),map_location='cpu',weights_only=True)
        route_exact=torch.equal(a['exact_mask'],b['exact_mask'])
        count_exact=torch.equal(a['row_count'],b['row_count'])
        lut_exact=(cases['baseline']['cases'][name]['lut_sha256'] ==
                   cases['bitmap']['cases'][name]['lut_sha256'])
        x=a['output'].float();y=b['output'].float()
        rel_l2=float(torch.linalg.vector_norm(x-y)/
                     torch.linalg.vector_norm(x).clamp_min(1.e-12))
        cosine=float(F.cosine_similarity(x.flatten(),y.flatten(),dim=0))
        baseline_ms=cases['baseline']['cases'][name]['complete']['median_ms']
        bitmap_ms=cases['bitmap']['cases'][name]['complete']['median_ms']
        baseline_lut=cases['baseline']['cases'][name]['lut_shape']
        bitmap_lut=cases['bitmap']['cases'][name]['lut_shape']
        count_lut_baseline=math.prod(baseline_lut)*4
        count_lut_bitmap=math.prod(bitmap_lut)*4
        good=route_exact and count_exact and lut_exact and rel_l2<=1.e-6 and cosine>=0.9999
        comparison[name]=dict(
            route_exact=route_exact,count_exact=count_exact,lut_exact=lut_exact,
            rel_l2=rel_l2,cosine=cosine,
            baseline_ms=baseline_ms,bitmap_ms=bitmap_ms,
            speedup=baseline_ms/bitmap_ms,
            baseline_lut_bytes=count_lut_baseline,bitmap_lut_bytes=count_lut_bitmap,
            lut_bytes_saved=count_lut_baseline-count_lut_bitmap,
            baseline_peak=cases['baseline']['cases'][name]['peak_temporary_bytes'],
            bitmap_peak=cases['bitmap']['cases'][name]['peak_temporary_bytes'],
            semantic_gate=good)
    ab=dict(status='complete',scope='physical-SM75 opt-in bitmap LUT, not H3 E2E',
            variants=cases,comparison=comparison)
    (out/'bitmap_ab_result.json').write_text(json.dumps(ab,indent=2))
    print('SOL_BITMAP_LUT_AB',json.dumps(comparison,sort_keys=True),flush=True)
    assert all(x['semantic_gate'] for x in comparison.values()),comparison
if __RUN_WORKSPACE_AB__:
    import torch.nn.functional as F
    cases={}
    for mode,enabled in (('baseline','0'),('pool','1')):
        variant_env=dict(env)
        variant_env['H3_SOL_WORKSPACE_POOL']=enabled
        cmd=[sys.executable,'-u',str(ROOT/'benchmarks/sol_workspace_pool_probe.py'),
             '--out',str(out/('workspace_'+mode))]
        proc=subprocess.run(cmd,cwd=ROOT,env=variant_env,stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT,text=True,timeout=1200)
        (out/('workspace_'+mode+'_stdout.log')).write_text(proc.stdout)
        print('workspace_'+mode,proc.stdout[-3000:],flush=True)
        assert proc.returncode==0,(mode,proc.returncode,proc.stdout[-5000:])
        cases[mode]=json.loads((out/('workspace_'+mode)/'result.json').read_text())
        assert cases[mode]['abi']==12
    comparison={}
    for name in ('d64_vae','real_shape_d128'):
        a=torch.load(out/'workspace_baseline'/(name+'.pt'),map_location='cpu',weights_only=True)
        b=torch.load(out/'workspace_pool'/(name+'.pt'),map_location='cpu',weights_only=True)
        route_exact=torch.equal(a['exact_mask'],b['exact_mask'])
        count_exact=torch.equal(a['row_count'],b['row_count'])
        lut_exact=(cases['baseline']['cases'][name]['lut_sha256'] ==
                   cases['pool']['cases'][name]['lut_sha256'])
        x=a['output'].float();y=b['output'].float()
        rel_l2=float(torch.linalg.vector_norm(x-y)/
                     torch.linalg.vector_norm(x).clamp_min(1.e-12))
        cosine=float(F.cosine_similarity(x.flatten(),y.flatten(),dim=0))
        baseline_ms=cases['baseline']['cases'][name]['complete']['median_ms']
        pool_ms=cases['pool']['cases'][name]['complete']['median_ms']
        base_mem=cases['baseline']['cases'][name]['memory']
        pool_mem=cases['pool']['cases'][name]['memory']
        good=route_exact and count_exact and lut_exact and rel_l2<=1.e-6 and cosine>=0.9999
        comparison[name]=dict(
            route_exact=route_exact,count_exact=count_exact,lut_exact=lut_exact,rel_l2=rel_l2,
            cosine=cosine,baseline_ms=baseline_ms,pool_ms=pool_ms,
            speedup=baseline_ms/pool_ms,
            baseline_cold_peak=base_mem['cold_peak_over_initial'],
            pool_cold_peak=pool_mem['cold_peak_over_initial'],
            pool_resident=pool_mem['resident_after_first'],
            pool_steady_peak_over_initial=pool_mem['steady_peak_over_initial'],
            semantic_gate=good)
    ab=dict(status='complete',
            scope='zero-weight physical-SM75 workspace A/B; persistent bytes explicit',
            variants=cases,comparison=comparison)
    (out/'workspace_ab_result.json').write_text(json.dumps(ab,indent=2))
    print('SOL_WORKSPACE_AB',json.dumps(comparison,sort_keys=True),flush=True)
    assert all(x['semantic_gate'] for x in comparison.values()),comparison
'''.replace("PAYLOAD",repr(encoded)).replace("HASHES",repr(hashes)).replace("__RUN_WORKSPACE_AB__",repr(args.workspace_ab)).replace("__RUN_BITMAP_AB__",repr(args.bitmap_ab))
    ast.parse(worker)
    nb=dict(nbformat=4,nbformat_minor=5,metadata=dict(kernelspec=dict(name="python3",language="python",display_name="Python 3")),
            cells=[dict(id="sol-v2-microgate",cell_type="code",execution_count=None,metadata={},outputs=[],source=worker.splitlines(keepends=True))])
    if args.workspace_ab:
        kernel_id,title="pamcks/h3-sol-workspace-ab-20261031","H3 Sol Workspace AB 20261031"
    elif args.bitmap_ab:
        kernel_id,title="pamcks/h3-sol-bitmap-ab-20261031","H3 Sol Bitmap LUT AB 20261031"
    else:
        kernel_id,title="pamcks/h3-sol-v2-t4-microgate-20261030","H3 Sol V2 T4 Microgate 20261030"
    meta=dict(id=kernel_id,title=title,
              code_file="experiment.ipynb",language="python",kernel_type="notebook",is_private=True,
              enable_gpu=True,enable_tpu=False,enable_internet=False,machine_shape="NvidiaTeslaT4",
              docker_image_pinning_type="original",dataset_sources=[],kernel_sources=[],competition_sources=[],model_sources=[])
    (job/"experiment.ipynb").write_text(json.dumps(nb),encoding="utf-8")
    (job/"kernel-metadata.json").write_text(json.dumps(meta,indent=2),encoding="utf-8")
    binding=dict(scope="zero-weight single-T4 Sol v2 micro-gate",source_hashes=hashes,
                 workspace_ab=args.workspace_ab,bitmap_ab=args.bitmap_ab,
                 payload_sha256=hashlib.sha256(packed).hexdigest(),payload_bytes=len(packed),
                 notebook_sha256=hashlib.sha256((job/"experiment.ipynb").read_bytes()).hexdigest(),
                 full_pipeline=False,model_weights=False,internet=False,status_polling=False,live_log_polling=False)
    (job/"source_binding.json").write_text(json.dumps(binding,indent=2),encoding="utf-8")
    receipt=dict(status="built",kernel=meta["id"],notebook_bytes=(job/"experiment.ipynb").stat().st_size,source_binding=binding)
    (args.evidence/"build_receipt.json").write_text(json.dumps(receipt,indent=2),encoding="utf-8")
    print(json.dumps(receipt,indent=2))
    if not args.submit:return
    state=dict(kernel=meta["id"],status="submitting",submitted_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
               result_relative="sol_v2_microgate/result.json",final_collection_count=0,source_binding=binding)
    sub=args.evidence/"gpu_submission.json"
    with sub.open("x",encoding="utf-8") as f:json.dump(state,f,indent=2)
    proc=subprocess.run([sys.executable,str(ROOT/"benchmarks/kaggle_tls12_runner.py"),"--cli","kernels","push","-p",str(job)],
                        capture_output=True,text=True,timeout=180)
    output=proc.stdout+proc.stderr
    m=re.search(r"Kernel version (\d+)",output)
    provider_error=bool(re.search(r"(?im)^\s*(?:Kernel push error:|Error:)",output))
    accepted=proc.returncode==0 and m is not None and not provider_error
    state.update(status="accepted" if accepted else "acceptance_uncertain",
                 returncode=proc.returncode,response=output,provider_error=provider_error)
    if m:state["version"]=int(m.group(1))
    u=re.search(r"https://www\.kaggle\.com/code/[\w-]+/[\w-]+",output)
    if u:state["url"]=u.group(0)
    sub.write_text(json.dumps(state,indent=2)+"\n",encoding="utf-8")
    (args.evidence/"gpu_submission_cli.log").write_text(output,encoding="utf-8")
    print(output[-3000:])
    if not accepted:raise RuntimeError("Kaggle submission was not accepted; do not retry automatically")

if __name__=="__main__":main()
