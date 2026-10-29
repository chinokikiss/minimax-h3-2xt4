import json,os,pathlib,sys
source=pathlib.Path('/kaggle/temp/h3_light_source')
env=dict(os.environ);env.update(json.loads(pathlib.Path('/kaggle/working/light_setup/execution_env.json').read_text()))
argv=[sys.executable,'-u',str(source/'benchmarks/run_dit_vram_batch.py'),'--source',str(source),'--output','/kaggle/working/dit_vram']
if False:argv+=['--cpu-preflight']
os.execve(sys.executable,argv,env)
