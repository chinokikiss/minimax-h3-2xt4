"""Compile SM75 cubins/resource reports on the CPU, never load or launch them."""
import importlib.util
import argparse
import json
from pathlib import Path
import re
import shutil
import subprocess


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--input',type=Path,default=Path('D:/attention_eval/operator_optimization_tp2_20261016/offline_optimization'));args=ap.parse_args()
    out=args.input
    package=Path(importlib.util.find_spec('tilelang').origin).parent
    compiler=shutil.which('nvcc');assert compiler,'CUDA compiler unavailable'
    vswhere=Path('C:/Program Files (x86)/Microsoft Visual Studio/Installer/vswhere.exe')
    host=subprocess.check_output([str(vswhere),'-latest','-products','*','-find','VC/Tools/MSVC/**/bin/Hostx64/x64/cl.exe'],text=True).strip().splitlines()[0]
    includes=[package/'src',package/'3rdparty/cutlass/include']
    assert all(p.exists() for p in includes)
    results=[]
    for source in sorted(out.glob('*.cu')):
        argv=[compiler,'--cubin','-arch=sm_75','-std=c++17','--expt-relaxed-constexpr','-Xcompiler=/Zc:preprocessor,/utf-8',
              '--compiler-bindir',str(Path(host).parent),'-Xptxas=-v,--warn-on-spills',str(source),'-o',str(source.with_suffix('.cubin'))]
        for include in includes:argv+=['-I',str(include)]
        p=subprocess.run(argv,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=90)
        source.with_suffix('.compile.log').write_text(p.stdout)
        match=re.search(r'Used (\d+) registers',p.stdout)
        spill=re.search(r'(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads',p.stdout)
        row={'source':source.name,'target':'sm_75','exit_code':p.returncode,
             'registers_per_thread':int(match[1]) if match else None,
             'stack_bytes':int(spill[1]) if spill else None,'spill_store_bytes':int(spill[2]) if spill else None,
             'spill_load_bytes':int(spill[3]) if spill else None,
             'ptxas_lines':[line for line in p.stdout.splitlines() if 'ptxas' in line],
             'scope':'Offline compile only, not execution or performance'}
        results.append(row);print('SM75_CUBIN',source.stem,p.returncode,row['registers_per_thread'],flush=True)
        if p.returncode:print(p.stdout[-1800:],flush=True);break
    (out/'compilation.json').write_text(json.dumps({'compiler':compiler,'host_compiler':host,'cases':results},indent=2))


if __name__=='__main__':main()
