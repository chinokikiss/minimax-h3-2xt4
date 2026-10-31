"""Matched complete-call T4 benchmark: Sol All-INT8 versus Kitchen dense INT8."""
import argparse
import json
import os
from pathlib import Path
import random
import statistics
import time
import torch

LENGTHS = (512, 1024, 2048, 4096, 6814, 8192, 12288, 16384, 32768, 51220)
HEADS = 28
DIM = 128


def save(path, state):
    path=Path(path); tmp=path.with_suffix(".tmp"); tmp.write_text(json.dumps(state,indent=2)+"\n"); tmp.replace(path)


def measure(name, fn, tokens):
    warmups = 3 if tokens <= 8192 else 2 if tokens <= 16384 else 1
    samples = 9 if tokens <= 8192 else 7 if tokens <= 16384 else 3 if tokens <= 32768 else 2
    for _ in range(warmups):
        x=fn(); del x
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    baseline=torch.cuda.memory_allocated(); event=[]; wall=[]; densities=[]
    for _ in range(samples):
        a=torch.cuda.Event(enable_timing=True); b=torch.cuda.Event(enable_timing=True)
        tick=time.perf_counter(); a.record(); out=fn(); b.record(); b.synchronize()
        event.append(a.elapsed_time(b)); wall.append((time.perf_counter()-tick)*1000)
        if hasattr(out,"density"): densities.append(float(out.density)); payload=out.output
        else: payload=out
        if payload.shape != (1,HEADS,tokens,DIM) or payload.dtype != torch.float16 or not torch.isfinite(payload).all():
            raise AssertionError(f"{name} output contract failed")
        del out,payload
    med=statistics.median(event)
    return dict(name=name,event_ms=event,wall_ms=wall,event_median_ms=med,
        event_p10_ms=sorted(event)[max(0,int(.1*len(event))-1)],event_p90_ms=sorted(event)[min(len(event)-1,int(.9*len(event)))],
        tokens_per_second=tokens/(med/1000.0),baseline_allocated=baseline,
        peak_allocated=torch.cuda.max_memory_allocated(),peak_temporary=torch.cuda.max_memory_allocated()-baseline,
        route_density_median=statistics.median(densities) if densities else None,
        samples=samples,warmups=warmups)


def run(output, library, rank=0, min_length=0):
    torch.cuda.set_device(rank); torch.set_grad_enabled(False)
    assert torch.cuda.get_device_capability(rank)==(7,5) and "T4" in torch.cuda.get_device_name(rank)
    os.environ["H3_SOL_ALL_INT8_LIBRARY"]=str(Path(library).resolve())
    import comfy_kitchen as ck
    from .backend import attention as sol
    lengths=tuple(length for length in LENGTHS if length >= int(min_length))
    if not lengths:
        raise ValueError("min_length excludes every benchmark length")
    state=dict(status="running",gpu=torch.cuda.get_device_name(rank),rank=rank,torch=torch.__version__,cuda=torch.version.cuda,
        scope="Synthetic complete raw-input attention API; no H3 model or generation",heads=HEADS,head_dim=DIM,
        lengths=list(lengths),sample_policy='9/7/3/2 by length; 3/2/1 warmups',cases=[])
    save(output,state)
    try:
        for length in lengths:
            torch.manual_seed(20261030 + length)
            q=torch.randn(1,HEADS,length,DIM,device=rank,dtype=torch.float16)*.2
            k=torch.randn_like(q); v=torch.randn_like(q)
            cases={
                "ck_int8_dense": lambda: ck.int8_attention(q,k,v),
                "sol_all_int8_tau1": lambda: sol(q,k,v,tau=1.0),
            }
            row=dict(length=length,rounds=[]);state["cases"].append(row);save(output,state)
            order=list(cases); random.Random(20261030+length).shuffle(order)
            for round_id,names in enumerate((order,list(reversed(order)))):
                for name in names:
                    result=measure(name,cases[name],length)
                    row["rounds"].append(dict(round=round_id,**result));save(output,state)
            def med(name):
                vals=[r["event_median_ms"] for r in row["rounds"] if r["name"]==name]
                return statistics.median(vals)
            ck_ms=med("ck_int8_dense"); sol_ms=med("sol_all_int8_tau1")
            row["summary"]=dict(ck_median_ms=ck_ms,sol_median_ms=sol_ms,
                speedup_vs_ck=ck_ms/sol_ms,sol_latency_change_pct=(sol_ms/ck_ms-1)*100)
            save(output,state)
            del q,k,v,cases
            torch.cuda.empty_cache()
        state["status"]="completed"
    except Exception as exc:
        state.update(status="failed",error=repr(exc)); raise
    finally: save(output,state)
    return state


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--output",type=Path,required=True);p.add_argument("--library",type=Path,required=True);p.add_argument("--rank",type=int,default=0);p.add_argument("--min-length",type=int,default=0)
    a=p.parse_args();run(a.output,a.library,a.rank,a.min_length)
