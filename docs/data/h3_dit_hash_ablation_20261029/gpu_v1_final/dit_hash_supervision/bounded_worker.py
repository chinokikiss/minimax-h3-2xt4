import json,os,pathlib,sys
env=dict(os.environ);env.update(json.loads(pathlib.Path('/kaggle/working/light_setup/execution_env.json').read_text()))
source=pathlib.Path('/kaggle/temp/h3_light_source')
argv=[sys.executable,'-u',str(source/'benchmarks/run_dit_hash_ablation.py'),'--source',str(source),
 '--output','/kaggle/working/dit_hash']

os.execve(sys.executable,argv,env)
