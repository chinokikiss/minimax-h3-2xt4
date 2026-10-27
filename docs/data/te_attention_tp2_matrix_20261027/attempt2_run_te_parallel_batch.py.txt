"""Bounded public-shard TE operator run; no private credentials or H3 pipeline."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib.metadata as md
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time
import traceback


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--lengths',type=int,nargs='+',choices=(128,353,1306,4096),default=(128,353,1306,4096))
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    state=dict(status='starting',full_h3_pipeline=False,full_te_encoder=False,trained_block=0,
        inputs='Seeded FP16 residuals /64; not real captured TE activations',cases=[])
    def save():(args.output/'result.json').write_text(json.dumps(state,indent=2))
    stop=threading.Event()
    def heartbeat():
        while not stop.wait(60):print('TE_BATCH_HEARTBEAT',state['status'],flush=True)
    threading.Thread(target=heartbeat,daemon=True).start();save()
    try:
        import torch
        from patches.legacy_runtime import validate_packages,validate_source
        validate_packages();validate_source(args.source/'ComfyUI')
        hardware=[dict(name=torch.cuda.get_device_name(i),capability=list(torch.cuda.get_device_capability(i)),
            uuid=str(torch.cuda.get_device_properties(i).uuid)) for i in range(torch.cuda.device_count())]
        assert len(hardware)==2 and len({d['uuid'] for d in hardware})==2
        assert all('T4' in d['name'] and d['capability']==[7,5] for d in hardware)
        state.update(hardware=hardware,torch=torch.__version__,cuda=torch.version.cuda,
            kitchen=md.version('comfy-kitchen'),tilelang=md.version('tilelang'),
            profiler_tools={name:shutil.which(name) for name in ('nsys','ncu')})
        nvcc=Path(shutil.which('nvcc') or '')
        assert nvcc.is_file() and 'release 12.8' in subprocess.check_output([str(nvcc),'--version'],text=True)
        os.environ.update(CUDACXX=str(nvcc.resolve()),CUDA_HOME=str(nvcc.resolve().parent.parent),
            CUDA_PATH=str(nvcc.resolve().parent.parent),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',
            NCCL_P2P_DISABLE='1',NCCL_MIN_NCHANNELS='4',NCCL_MAX_NCHANNELS='4',NCCL_DEBUG='INFO',
            HF_HUB_DISABLE_XET='1',HF_HUB_DISABLE_PROGRESS_BARS='1',HF_HUB_DOWNLOAD_TIMEOUT='90')
        os.environ.pop('NCCL_PROTO',None)
        state['transport_environment']={name:os.environ.get(name) for name in ('NCCL_P2P_DISABLE','NCCL_MIN_NCHANNELS','NCCL_MAX_NCHANNELS','NCCL_PROTO','NCCL_IB_DISABLE')}
        plan=json.loads((args.source/'benchmarks/review_t4_assets.json').read_text())
        assert plan['revision']=='d3e6a4b131056d2c626a01acf6ba48123a885783' and not plan['token_required']
        assets=Path('/kaggle/temp/h3_te_block_assets');assets.mkdir(exist_ok=True)
        state['status']='download_two_public_te_shards';save();started=time.monotonic()
        from huggingface_hub import hf_hub_download
        def download(rank):
            name=f'te.block00.rank{rank}.safetensors';expected=plan['objects'][name]
            file=Path(hf_hub_download(plan['repository'],name,revision=plan['revision'],token=False,local_dir=assets))
            with file.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
            assert file.stat().st_size==expected['bytes'] and digest==expected['sha256']
            return dict(name=name,**expected)
        with ThreadPoolExecutor(max_workers=2) as pool:objects=list(pool.map(download,(0,1)))
        state['assets']=dict(repository=plan['repository'],revision=plan['revision'],objects=objects,seconds=time.monotonic()-started)
        state['status']='operator_matrix';save()
        command=[sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=2',
            str(Path(__file__).with_name('te_parallel_matrix.py')),'--assets',str(assets),'--output',str(args.output),
            '--lengths',*[str(n) for n in args.lengths]]
        with (args.output/'workers.log').open('w') as stream:
            process=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
            for line in process.stdout:
                stream.write(line);stream.flush()
                if line.startswith(('TE_','Traceback')):print(line,end='',flush=True)
            code=process.wait()
        state['worker_exit_code']=code
        ranks=[json.loads((args.output/f'rank{rank}.json').read_text()) for rank in (0,1)]
        state['ranks']=ranks
        if code or not all(r['status']=='complete' and len(r['cases'])==6*len(args.lengths) for r in ranks):
            raise RuntimeError('TE workers did not complete the matrix')
        comparisons=[]
        for length in args.lengths:
            for backend in ('native','tilelang_group32','kitchen_causal'):
                rows=[[next(c for c in r['cases'] if c['tokens']==length and c['backend']==backend and c['topology']==topology)
                    for topology in ('single_gpu','tp2_nccl')] for r in ranks]
                assert rows[0][1]['output_sha256']==rows[1][1]['output_sha256'],'TP2 owners disagree'
                single=[row[0]['block']['median_wall_ms'] for row in rows]
                tp=max(row[1]['block']['median_wall_ms'] for row in rows)
                replay=max(row[1]['collective_result_replay']['median_wall_ms'] for row in rows)
                comparisons.append(dict(tokens=length,backend=backend,single_gpu_medians_ms=single,
                    tp2_slower_owner_median_ms=tp,replay_slower_owner_median_ms=replay,
                    speedup_vs_gpu0=single[0]/tp,speedup_vs_gpu1=single[1]/tp,
                    replay_latency_reduction_ms=tp-replay))
        state.update(status='complete',executed_lengths=args.lengths,comparisons=comparisons,
            conclusion_scope='Resident block compute/communication only; no end-to-end TE or weight-streaming speedup claim')
    except BaseException as error:
        state.update(status='failed',error=repr(error));(args.output/'exception.txt').write_text(traceback.format_exc());raise
    finally:
        stop.set();save()


if __name__=='__main__':main()
