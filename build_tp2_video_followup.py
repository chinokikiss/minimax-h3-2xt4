"""Close the active Video baseline gap within a six-minute T4 cap."""
import ast
import hashlib
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
old=ROOT/'kaggle_h3-tp2-operator-screen'
code=''.join(json.loads((old/'experiment.ipynb').read_text())['cells'][0]['source'])
code=code.replace("/kaggle/working/tp2_operators","/kaggle/working/video_tp2").replace('time.monotonic()+1140','time.monotonic()+330').replace("'budget_seconds':1140","'budget_seconds':330").replace("'cu130_runtime',420","'cu130_runtime',180")
files=['benchmarks/tp2_video_microchain.py','benchmarks/tp2_timing.py','patches/parallel_vae.py','patches/windowed_tp.py','patches/weight_residency.py','patches/diagnostic_metrics.py']
extra=''
hashes={}
for name in files:
    source=(ROOT/name).read_text(encoding='utf8');ast.parse(source);hashes[name]=hashlib.sha256(source.encode()).hexdigest()
    extra+=f'p=SRC/{name!r};p.parent.mkdir(parents=True,exist_ok=True);p.write_text({source!r})\n'
extra+=f"(OUT/'followup_source_hashes.json').write_text(json.dumps({hashes!r},indent=2))\n"
code=code.replace('save()\ntry:',extra+'save()\ntry:')
start=code.index('        gemm_ok=run(');end=code.index('\nexcept Exception as e:',start)
code=code[:start]+'''        base=[*launcher,str(SRC/'benchmarks/tp2_video_microchain.py'),'--data',str(mounts[0].parent)]
        ok=run([*base,'--output',str(OUT/'unprofiled')],'video',85)
        nsys=shutil.which('nsys') or next((str(p) for p in pathlib.Path('/opt/nvidia').rglob('nsys') if p.is_file()),None)
        if False and ok and nsys:  # timings-only confirmation; v1 has the traces
            run([nsys,'profile','--trace=cuda,nvtx,osrt','--sample=none','--cpuctxsw=none','--capture-range=cudaProfilerApi',
                 '--capture-range-end=repeat','--force-overwrite=true','-o',str(OUT/'video_profile'),*base,'--profile','--output',str(OUT/'profiled')],'video_nsys',65)
    finally:monitor.terminate();monitor.wait(timeout=10)
    state['status']='complete' if ok else 'partial'
'''+code[end:]
ast.parse(code)
folder=ROOT/'kaggle_h3-tp2-video-microchain';folder.mkdir(exist_ok=True)
nb=json.loads((old/'experiment.ipynb').read_text());nb['cells'][0]['source']=code.splitlines(keepends=True)
(folder/'experiment.ipynb').write_text(json.dumps(nb),encoding='utf8')
metadata=json.loads((old/'kernel-metadata.json').read_text());metadata.update(id='chinokiki9/h3-tp2-video-microchain',title='H3 TP2 Video Microchain')
(folder/'kernel-metadata.json').write_text(json.dumps(metadata,indent=2))
print(folder)
