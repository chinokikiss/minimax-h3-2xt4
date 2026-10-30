import json,subprocess,sys
from pathlib import Path
repo=Path('C:/Users/Administrator/Documents/project/2xT4 minimax h3');sys.path.insert(0,str(repo))
from benchmarks.collect_kaggle_final_once import collect
root=Path(__file__).resolve().parent
original=subprocess.run
def tls_run(argv,**kwargs):
 assert argv[1:4]==['-m','kaggle','kernels'] and argv[4]=='output'
 command=[argv[0],str(repo/'benchmarks/kaggle_tls12_runner.py'),'--cli',*argv[3:]]
 return original(command,**kwargs)
print(json.dumps(collect(root/'gpu_submission.json',root/'final_v2',run=tls_run)),flush=True)
