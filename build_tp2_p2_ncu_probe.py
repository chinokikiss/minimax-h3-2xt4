"""One bounded current T4 counter-permission probe, with no runtime upgrade."""
import ast
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent;folder=ROOT/'kaggle_h3-tp2-p2-counter-probe';folder.mkdir(exist_ok=True)
code='''import json,pathlib,shutil,subprocess,sys,time
out=pathlib.Path('/kaggle/working/ncu_probe');out.mkdir(exist_ok=True)
state={'scope':'Current physical-T4 counter access probe only; not attention performance'}
try:
    import torch
    state['hardware']=[{'name':torch.cuda.get_device_name(i),'capability':list(torch.cuda.get_device_capability(i))} for i in range(torch.cuda.device_count())]
    assert len(state['hardware'])==2 and all(d['capability']==[7,5] and 'T4' in d['name'] for d in state['hardware'])
    state['torch']=torch.__version__
    ncu=shutil.which('ncu') or next((str(p) for p in pathlib.Path('/opt/nvidia').rglob('ncu') if p.is_file()),None)
    state['ncu']=ncu
    if not ncu:state['status']='tool_unavailable'
    else:
        state['version']=subprocess.check_output([ncu,'--version'],text=True)
        script=out/'probe.py';script.write_text("import torch\\ntorch.cuda.set_device(0)\\nx=torch.ones(1048576,device='cuda')\\ny=x+x\\ntorch.cuda.synchronize()\\n")
        argv=[ncu,'--metrics','sm__cycles_elapsed.avg,sm__warps_active.avg.pct_of_peak_sustained_active,lts__t_sectors_op_read.sum','--launch-count','1','--force-overwrite','-o',str(out/'counter_access'),sys.executable,str(script)]
        with (out/'probe.log').open('w') as log:
            try:
                p=subprocess.run(argv,stdout=log,stderr=subprocess.STDOUT,timeout=45);state['exit_code']=p.returncode
            except subprocess.TimeoutExpired:state['status']='timeout'
        text=(out/'probe.log').read_text(errors='replace')
        if 'ERR_NVGPUCTRPERM' in text:state['status']='blocked_ERR_NVGPUCTRPERM'
        elif state.get('exit_code')==0:state['status']='counters_accessible'
        else:state.setdefault('status','probe_failed')
except Exception as error:state.update(status='failed',error=type(error).__name__+': '+str(error))
(out/'status.json').write_text(json.dumps(state,indent=2));print('P2_NCU_PROBE',state['status'],flush=True)
'''
# Preserve actual newlines in the child script, not literal backslash-n.
ast.parse(code)
notebook=dict(nbformat=4,nbformat_minor=5,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}},cells=[dict(id='counter-probe',cell_type='code',metadata={},execution_count=None,outputs=[],source=code.splitlines(keepends=True))])
(folder/'experiment.ipynb').write_text(json.dumps(notebook),encoding='utf8')
(folder/'kernel-metadata.json').write_text(json.dumps(dict(id='chinokiki9/h3-tp2-p2-counter-probe',title='H3 TP2 P2 Counter Probe',code_file='experiment.ipynb',language='python',kernel_type='notebook',is_private=True,enable_gpu=True,enable_tpu=False,enable_internet=False,dataset_sources=[],competition_sources=[],kernel_sources=[],model_sources=[]),indent=2))
print(folder)
