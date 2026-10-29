import json,os,pathlib,sys
env=dict(os.environ);env.update(json.loads(pathlib.Path('/kaggle/working/light_setup/execution_env.json').read_text()))
source=pathlib.Path('/kaggle/temp/h3_light_source')
argv=[sys.executable,'-u',str(source/'benchmarks/qualify_turbo_challenge.py'),'--source',str(source),'--output','/kaggle/working/turbo_challenge_cpu']
if False:argv+=['--plan',str(source/'turbo_challenge_plan.json'),'--scratch','/tmp/h3_turbo_1mp_10s_20261029','--e2e-timeout','5400']
os.execve(sys.executable,argv,env)
