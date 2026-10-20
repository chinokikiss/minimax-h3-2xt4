"""Bounded CPU assessment of complete, version-bound generation artifacts."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--runs',type=Path,required=True)
    parser.add_argument('--embedding-revisions',type=Path)
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    env=os.environ.copy();env.update(OMP_NUM_THREADS='8',MKL_NUM_THREADS='8')
    state={'status':'scoring','device':'cpu','scope':'quality proxies, not hardware performance','runs':[]}
    path=args.runs/'local_assessment.json'
    def save():path.write_text(json.dumps(state,indent=2))
    save()
    for run in json.loads((args.runs/'status.json').read_text())['runs']:
        name=run['name'];directory=args.runs/name
        if not (directory/'quality_screen.json').exists():continue
        config=json.loads((args.runs/(name+'.json')).read_text())
        record={'name':name,'status':'running'};state['runs'].append(record);save()
        command=[sys.executable,str(root/'benchmarks/quality_first_video_metrics.py'),
            '--video',str(directory/'native_video.mp4'),'--reference',str(args.runs/'reference.jpg'),
            '--prompt',config['prompt'],'--output',str(directory/'embedding_metrics.json'),'--device','cpu','--embeddings']
        if args.embedding_revisions:command+=['--embedding-revisions',str(args.embedding_revisions)]
        try:
            with (directory/'local_embedding_metrics.log').open('w') as log:
                process=subprocess.run(command,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=900)
            record.update(status='measured' if process.returncode==0 else 'failed',exit_code=process.returncode)
        except subprocess.TimeoutExpired:record['status']='timed_out'
        save()
    state['status']='finished';state['quality_acceptance']='pending blinded ratings; proxies do not establish noninferiority';save()


if __name__=='__main__':main()
