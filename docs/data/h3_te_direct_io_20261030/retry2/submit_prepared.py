import ast,base64,datetime as dt,hashlib,json,re,subprocess,sys
from pathlib import Path
root=Path('D:/attention_eval/h3_te_direct_io_20261030/retry2');job=root/'gpu_job'
meta=json.loads((job/'kernel-metadata.json').read_text(encoding='utf-8'));binding=json.loads((job/'source_binding.json').read_text(encoding='utf-8'))
nbpath=job/meta['code_file'];code=''.join(json.loads(nbpath.read_text(encoding='utf-8'))['cells'][0]['source']);ast.parse(code)
assert hashlib.sha256(nbpath.read_bytes()).hexdigest()==binding['notebook_sha256']
assert meta['id']=='pamcks/h3-te-direct-io-t4-20261030' and meta['is_private'] and meta['enable_gpu'] and not meta['dataset_sources'] and not meta['kernel_sources']
workers=[]
for node in ast.walk(ast.parse(code)):
 if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr=='b64decode' and isinstance(node.args[0],ast.Constant):
  try:workers.append(base64.b64decode(node.args[0].value).decode())
  except (ValueError,UnicodeError):pass
execution=next(body for body in workers if "run_te_direct_io.py" in body)
tree=ast.parse(execution);encoded=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='FILES' for t in n.targets))
assert set(encoded)==set(binding['benchmark_sources'])
for name,value in encoded.items():assert hashlib.sha256(base64.b64decode(value)).hexdigest()==binding['benchmark_sources'][name]
assert not any(term in execution for term in ['kernels_status','kernels_list','kernels_output'])
receipt=root/'gpu_submission.json'
state=dict(kernel=meta['id'],status='submitting',submitted_utc=dt.datetime.now(dt.timezone.utc).isoformat(),source_binding=binding,run_timeout_seconds=5100,earliest_final_collection_seconds=5100,final_collection_count=0,experiment_kind='te_direct_io',result_relative='te_direct_io/result.json')
with receipt.open('x',encoding='utf-8') as stream:json.dump(state,stream,indent=2)
try:
 process=subprocess.run([sys.executable,'-m','kaggle','kernels','push','-p',str(job),'--accelerator','NvidiaTeslaT4','-t','5100'],capture_output=True,text=True,timeout=180)
 output=process.stdout+process.stderr
 state.update(status='accepted' if process.returncode==0 else 'acceptance_uncertain',returncode=process.returncode)
 version=re.search(r'Kernel version (\d+)',output)
 if version:state['version']=int(version.group(1))
 url=re.search(r'https://www\.kaggle\.com/code/[\w-]+/[\w-]+',output)
 if url:state['url']=url.group(0)
except subprocess.TimeoutExpired:
 output='Submission acknowledgement timed out; preserve claim, no automatic retry.';state.update(status='acceptance_uncertain',returncode=-1)
receipt.write_text(json.dumps(state,indent=2),encoding='utf-8');(root/'gpu_submission_cli.log').write_text(output,encoding='utf-8');print(json.dumps({k:state.get(k) for k in ['kernel','status','version','url','returncode']}),flush=True)
