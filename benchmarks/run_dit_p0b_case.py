"""Physical 2xT4 trained DiT block P0-B A/B, source-bound and fail-closed.

Both arms run the current default production P0-A1, P0-C, P0-2 and P0-3.
Only the experimental FC2 fused_stored preparation is interposed.
"""
import argparse
from datetime import timedelta
import gc
import json
from pathlib import Path
import sys
import time
import traceback

import torch

from benchmarks.run_dit_p0ac_case import inputs,sol_options
from benchmarks.run_dit_vram_case import Memory,balance,baseline_phases
from benchmarks.dit_p0b_t4_candidates import active_p0b,probe_codes
from benchmarks.dit_p0ac_t4_candidates import bytes_compare

VARIANTS=('baseline','p0_b')


@torch.no_grad()
def test_rows(block,rows,device,barrier):
    x,segments,freqs,embedding=inputs(rows,device)
    options=sol_options(rows)
    hits={name:{} for name in VARIANTS}
    def call(name):
        with active_p0b(name=='p0_b',hits[name]):
            return block(x.clone(),embedding,segments,freqs,options)
    funcs={name:(lambda key=name:call(key)) for name in VARIANTS}
    barrier()
    reference=funcs['baseline']().detach().cpu()
    barrier()
    trial=funcs['p0_b']()
    equality=bytes_compare(trial.detach().cpu(),reference)
    del trial,reference
    if not equality['bit_exact'] or not equality['finite']:
        raise AssertionError('Trained-block output failed raw-bit identity')
    timings=balance(funcs,barrier,warmups=3,repeats=9)
    phases={}
    for name in VARIANTS:
        barrier()
        torch.cuda.synchronize(device)
        observer=Memory()
        with baseline_phases(block,observer):
            output=funcs[name]()
            torch.cuda.synchronize(device)
        del output
        phases[name]=observer.rows
    # Whole-call peaks are recorded with explicit pre-call live allocation
    # and alternating order. Differences are valid only if baselines match.
    full_call={}
    for name in ('baseline','p0_b','p0_b','baseline'):
        barrier()
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize(device)
        before=torch.cuda.memory_allocated(device)
        torch.cuda.reset_peak_memory_stats(device)
        start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        start.record()
        output=funcs[name]()
        end.record()
        end.synchronize()
        reading=dict(initial_allocated=before,peak_allocated=torch.cuda.max_memory_allocated(device),
            peak_incremental=torch.cuda.max_memory_allocated(device)-before,
            after_allocated=torch.cuda.memory_allocated(device),cuda_ms=start.elapsed_time(end))
        del output
        full_call.setdefault(name,[]).append(reading)
    if hits['p0_b'].get('p0b_fc2_calls',0)<16:
        raise AssertionError('P0-B FC2 was not really selected for every measured call')
    return dict(rows=rows,attention='sol_v2' if rows==65166 else 'kitchen',
        trained_block_bit_exact=equality,all_bit_exact=True,
        measurements=timings,stage_peak_allocated=phases,
        whole_call_peak_allocated=full_call,runtime_dispatch_hits=hits,
        production_a1_c_default=True)


def main():
    import os
    import torch.distributed as dist
    from safetensors.torch import load_file
    from patches.legacy_runtime import validate_packages,validate_source
    from patches.rank_local_int8 import shell,state_hashes
    from patches.h3_tensor_flow import prepare_tp,TensorFlowPolicy
    from patches.h3_operator_policy import H3OperatorPolicy,configure_block
    from patches import optimized_tp

    parser=argparse.ArgumentParser()
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--weights',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    sys.path[:0]=[str(args.source),str(args.source/'ComfyUI'),str(args.source/'benchmarks')]
    import comfy.cli_args
    rank=int(os.environ['LOCAL_RANK'])
    device=torch.device('cuda',rank)
    torch.cuda.set_device(device)
    torch.set_grad_enabled(False)
    torch.set_num_threads(2)
    torch.manual_seed(20261102)
    torch.backends.cuda.matmul.allow_tf32=False
    out=args.output/f'rank{rank}'
    out.mkdir(parents=True,exist_ok=True)
    result=dict(status='starting',rank=rank,study='p0_b_physical_dual_t4',
        experimental=True,production_p0b_enabled=False,full_pipeline=False,
        hardware=dict(name=torch.cuda.get_device_name(device),torch=str(torch.__version__),
            cuda=torch.version.cuda,capability=list(torch.cuda.get_device_capability(device))))
    try:
        validate_packages()
        validate_source(args.source/'ComfyUI')
        if torch.cuda.get_device_capability(device)!=(7,5) or 'T4' not in result['hardware']['name']:
            raise RuntimeError('Physical Tesla T4 SM75 required on both ranks')
        dist.init_process_group('nccl',timeout=timedelta(seconds=240))
        cpu_group=dist.new_group(backend='gloo',timeout=timedelta(seconds=240))
        barrier=lambda:dist.barrier(group=cpu_group)
        weights=load_file(str(args.weights/f'dit.block00.rank{rank}.safetensors'))
        block=shell('dit',rank)
        block.load_state_dict(weights,assign=True)
        proof=json.loads((args.source/'bundle_provenance'/f'dit.block00.rank{rank}.json').read_text())
        assert state_hashes(block)==proof['state_sha256']
        prepare_tp(block,'dit',TensorFlowPolicy(mode='fp16',residual_exponent=9))
        policy=H3OperatorPolicy(dit_attention='sol',fc2_quantizer='fused_stored',
            column_linear_backend='cuda_int8_tilelang',dit_prepared_attention=True)
        configure_block(block,'dit',policy)
        block.to(device).eval()
        del weights
        optimized_tp.enable()
        assert block.qkv_proj._vram_p0_a1 is True
        assert block.fc1._vram_p0_a1 is True
        assert block.out_proj._vram_p0_c is True
        assert block.out_proj._vram_p0_2 is True
        assert block.fc2._vram_p0_3 is True
        assert block.fc1._vram_p0_3_producer is True
        result['weight_bytes']=sum(t.numel()*t.element_size() for t in block.state_dict().values())
        result['intermediate']=[]
        for rows in (128,6814,59134,65166):
            barrier()
            observed=probe_codes(rows,device,dist,rank)
            result['intermediate'].append(observed)
            if not observed['all_bit_exact']:
                raise AssertionError('P0-B quantized codes/scales differ from production FC2')
        result['per_geometry']=[]
        for rows in (59134,65166):
            policy=H3OperatorPolicy(dit_attention='kitchen' if rows==59134 else 'sol',
                fc2_quantizer='fused_stored',column_linear_backend='cuda_int8_tilelang',
                dit_prepared_attention=True)
            configure_block(block,'dit',policy)
            observed=test_rows(block,rows,device,barrier)
            result['per_geometry'].append(observed)
            (out/'result.json').write_text(json.dumps(result,indent=2))
        result['all_bit_exact']=all(p['all_bit_exact'] for p in result['intermediate']) and all(
            p['all_bit_exact'] for p in result['per_geometry'])
        result['status']='complete'
    except BaseException as exc:
        result.update(status='failed',error=repr(exc))
        (out/'exception.txt').write_text(traceback.format_exc())
        raise
    finally:
        (out/'result.json').write_text(json.dumps(result,indent=2))
        if dist.is_initialized():dist.destroy_process_group()
        print('DIT_P0B_RANK_FINAL',rank,result['status'],flush=True)


if __name__=='__main__':
    main()
