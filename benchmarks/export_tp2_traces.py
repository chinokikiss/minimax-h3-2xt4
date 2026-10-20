"""Export collected reports locally; CPU-only, no new GPU profiling."""
import argparse
import json
from pathlib import Path
import subprocess


def main():
    ap=argparse.ArgumentParser();ap.add_argument('root',type=Path);args=ap.parse_args()
    nsys=Path('C:/Program Files/NVIDIA Corporation/Nsight Systems 2026.1.3/target-windows-x64/nsys.exe')
    assert nsys.exists();commands=[]
    for p in sorted(args.root.glob('*.nsys-rep')):
        out=p.with_suffix('.sqlite')
        with p.with_suffix('.export.log').open('w') as log:
            result=subprocess.run([str(nsys),'export','--type','sqlite','--force-overwrite=true','-o',str(out),str(p)],stdout=log,stderr=subprocess.STDOUT,timeout=60)
        commands.append({'source':p.name,'sqlite':out.name,'exit_code':result.returncode});assert result.returncode==0
    (args.root/'local_export_commands.json').write_text(json.dumps(commands,indent=2));print('CPU exported',len(commands),'Systems reports')


if __name__=='__main__':main()
