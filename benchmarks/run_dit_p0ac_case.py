"""Physical dual-T4 P0-A1/P0-C isolation on trained block0 shards.

Current production P0-2/P0-3 remain enabled. CPU-first frozen study,
no installed package edits, no automatic Kaggle submission or status polling.
All output and timing claims are block-level rather than full H3 E2E.
"""
import argparse
from datetime import timedelta
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback
from types import SimpleNamespace

import torch

from benchmarks.run_dit_vram_case import balance, Memory, baseline_phases
from benchmarks.dit_p0ac_t4_candidates import active_variant, assert_hits, build_p0a1, bytes_compare, probe_p0c


VARIANTS = ("baseline", "p0_a1", "p0_c", "p0_a1_p0_c")


def inputs(rows, device):
    from benchmarks.dit_vram_candidates import row_labels
    regions=[(0,2392,7),(2392,3892,6),(3892,4702,5),(4702,rows,0)]
    segments=[(lo,hi,tag) for lo,hi,tag in regions if lo<rows]
    row_labels(rows,segments,device,modulation_rows=9)
    x=torch.randn(rows,5376,device=device,dtype=torch.float16) * (1./512.)
    angle=(torch.arange(64,device=device,dtype=torch.float32)[None,:]*.01+
           torch.arange(rows,device=device,dtype=torch.float32)[:,None]*.0001)
    c,s=angle.cos(),angle.sin()
    freqs=torch.stack((c,-s,s,c),-1).reshape(1,rows,1,64,2,2).half()
    clocks=torch.tensor([.5,.75,1.],device=device,dtype=torch.float32)
    time_embedding=clocks[:,None].pow(torch.arange(8,device=device)[None,:])
    return x,segments,freqs,time_embedding


def sol_options(rows):
    # The 65,166-row completed 0.85-MP case has 4,974 conditioning+audio
    # tokens and 60,192 video tokens = 18*3,344. Do not invent a 59K Sol layout.
    if rows!=65166:return {}
    layout=SimpleNamespace(seq_len=rows,
        segments=((0,4164,'text'),(4164,4974,'audio'),(4974,rows,'video')),
        signature=('synthetic_t4',18))
    return dict(minimax_h3_layout=layout,_h3_nfe=1,_h3_sol_topology_v2=True,
                _h3_sol_tau=1.,_h3_sol_sync_free_lut=True,
                _h3_sol_collect_route_stats=False,_h3_sol_temporal_radius=1,
                _h3_sol_local_radius_tokens=64,_h3_sol_protect_frame_boundaries=True)


def test_rows(block, rows, device, barrier, out):
    x, segments, freqs, embedding = inputs(rows, device)
    options = sol_options(rows)
    hit_counters = {name: {} for name in VARIANTS}
    candidates = {
        name: build_p0a1(hit_counters[name]) for name in VARIANTS
        if "p0_a1" in name
    }

    def call(name):
        with active_variant(name, hit_counters[name], candidates.get(name)):
            return block(x.clone(), embedding, segments, freqs, options)

    functions = {name: (lambda key=name: call(key)) for name in VARIANTS}
    barrier()
    torch.cuda.synchronize()
    reference = functions["baseline"]().detach().cpu()
    precision = {}
    for name in VARIANTS[1:]:
        barrier()
        output = functions[name]()
        precision[name] = bytes_compare(output.detach().cpu(), reference)
        del output
    del reference

    # Compile/warm up every path before any steady-state measurements.
    timings = balance(functions, barrier, warmups=2, repeats=5)
    phases = {}
    for name in VARIANTS:
        barrier()
        observer = Memory()
        with baseline_phases(block, observer):
            output = functions[name]()
            torch.cuda.synchronize()
            del output
        phases[name] = observer.rows

    for name in VARIANTS:
        assert_hits(hit_counters[name], name)
    return dict(
        rows=rows, attention="sol_v2" if rows == 65166 else "kitchen",
        residual_exponent=9, precision=precision,
        measurements=timings, stage_peak_allocated=phases,
        runtime_dispatch_hits=hit_counters,
        all_bit_exact=all(v["bit_exact"] for v in precision.values()),
        all_finite=all(v["finite"] for v in precision.values()),
    )


def main():
    import os
    import torch.distributed as dist
    from safetensors.torch import load_file
    parser=argparse.ArgumentParser()
    parser.add_argument("--source",type=Path,required=True)
    parser.add_argument("--weights",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    sys.path[:0]=[str(args.source),str(args.source/"ComfyUI"),str(args.source/"benchmarks")]
    import comfy.cli_args
    from patches.legacy_runtime import validate_packages,validate_source
    from patches.rank_local_int8 import shell,state_hashes
    from patches.h3_tensor_flow import prepare_tp,TensorFlowPolicy
    from patches.h3_operator_policy import H3OperatorPolicy,configure_block
    from patches import optimized_tp
    from benchmarks.dit_p0ac_t4_candidates import build_p0a1
    build_p0a1({})
    rank=int(os.environ["LOCAL_RANK"])
    device=torch.device("cuda",rank)
    torch.cuda.set_device(device)
    torch.set_grad_enabled(False)
    torch.set_num_threads(2)
    torch.manual_seed(20261102)
    torch.backends.cuda.matmul.allow_tf32=False
    out=args.output/f"rank{rank}"
    out.mkdir(parents=True,exist_ok=True)
    result=dict(status="starting",rank=rank,experiment="p0_a1_p0_c",experimental=True,full_pipeline=False,
                hardware=dict(name=torch.cuda.get_device_name(device),torch=str(torch.__version__),
                    cuda=torch.version.cuda,capability=list(torch.cuda.get_device_capability(device))))
    try:
        validate_packages()
        validate_source(args.source/"ComfyUI")
        if torch.cuda.get_device_capability(device)!=(7,5) or "T4" not in result["hardware"]["name"]:
            raise RuntimeError("Two physical SM75 T4 devices required")
        dist.init_process_group("nccl",timeout=timedelta(seconds=240))
        cpu_group=dist.new_group(backend="gloo",timeout=timedelta(seconds=240))
        barrier=lambda:dist.barrier(group=cpu_group)
        weights=load_file(str(args.weights/f"dit.block00.rank{rank}.safetensors"))
        block=shell("dit",rank)
        block.load_state_dict(weights,assign=True)
        proof=json.loads((args.source/"bundle_provenance"/f"dit.block00.rank{rank}.json").read_text())
        assert state_hashes(block)==proof["state_sha256"]
        prepare_tp(block,"dit",TensorFlowPolicy(mode="fp16",residual_exponent=9))
        # Match current 1/512 Sol-v2 production at 65K and retained 59K
        # Kitchen reference benchmark for the second requested shape.
        policy=H3OperatorPolicy(dit_attention="sol",fc2_quantizer="fused_stored",
            column_linear_backend="cuda_int8_tilelang",dit_prepared_attention=True)
        configure_block(block,"dit",policy)
        block.to(device).eval()
        del weights
        optimized_tp.enable()
        result["weight_bytes"]=sum(t.numel()*t.element_size() for t in block.state_dict().values())
        # Fail closed if a supposedly frozen baseline changes its lifetime policy.
        assert block.out_proj._vram_p0_2 is True
        assert block.fc2._vram_p0_3 is True
        assert block.fc1._vram_p0_3_producer is True
        result["p0c_real_sm75_probe"] = []
        for probe_rows in (128, 6814, 59134, 65166):
            barrier()
            result["p0c_real_sm75_probe"].append(probe_p0c(probe_rows, device, dist))
        result["per_geometry"]=[]
        for rows in (59134,65166):
            if rows==59134:
                configure_block(block,"dit",H3OperatorPolicy(dit_attention="kitchen",
                    fc2_quantizer="fused_stored",column_linear_backend="cuda_int8_tilelang",
                    dit_prepared_attention=True))
            else:configure_block(block,"dit",policy)
            case = test_rows(block,rows,device,barrier,out)
            result["per_geometry"].append(case)
            (out/"result.json").write_text(json.dumps(result,indent=2))
        result["all_bit_exact"] = all(
            row["all_bit_exact"] and row["all_finite"]
            for row in result["per_geometry"]
        )
        result["p0c_all_intermediate_exact"] = all(
            row["all_bit_exact"] for row in result["p0c_real_sm75_probe"]
            if row["rows"] in (59134, 65166)
        )
        result["status"]="complete"
    except BaseException as e:
        result.update(status="failed",error=repr(e))
        (out/"exception.txt").write_text(traceback.format_exc())
        raise
    finally:
        (out/"result.json").write_text(json.dumps(result,indent=2))
        if dist.is_initialized():dist.destroy_process_group()
        print("DIT_P0_RANK_FINAL",rank,result["status"],flush=True)


if __name__=="__main__":
    main()
