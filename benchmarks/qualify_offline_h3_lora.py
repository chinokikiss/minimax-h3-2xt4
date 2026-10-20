"""Independent Native offline byte reference and full block output checks."""
import hashlib
import json
from pathlib import Path
import simulate_weight_residency as setup
import torch
import retest
import te_dit_study as study
from logical_dual_gpu import LogicalDualGPU,isolate_cublas_workspaces
from patches.h3_lora import H3LoRAStore
from patches.windowed_tp import CheckpointReader,prepare_tp_block
from patches.optimized_tp import enable,disable


def main():
    import comfy_kitchen as ck
    from comfy_kitchen.tensor.int8 import TensorWiseINT8Layout as Layout
    import patches.tp_dit as dit
    ck.backends.cuda._max_dynamic_shared_memory_per_block=lambda _:49152
    torch.set_grad_enabled(False);torch.backends.cuda.matmul.allow_tf32=False;isolate_cublas_workspaces()
    root=Path(__file__).resolve().parents[1];cases=[]
    original=CheckpointReader(setup.PATHS['dit'])
    merged=CheckpointReader(root/'semantic_results/dmad_lora/merged_block0.safetensors')
    store=H3LoRAStore((('D:/minimaxH3/loras/dmad_minimax_h3_4step_full_critic_comfyui.safetensors',1.),),original)
    target='token_refiner.blocks.0.attn.out_proj'
    dense=original.read_tensor(target+'.weight').cuda().float()
    for a,b,scale in store.terms(target):dense.add_(torch.mm(b.cuda().float(),a.cuda().float())*scale)
    dense_reader=CheckpointReader(root/'semantic_results/dmad_lora/merged_refiner_out.safetensors')
    exact=torch.equal(dense.to(torch.bfloat16).cpu(),dense_reader.read_tensor(target+'.weight'))
    cases.append(dict(label=target,native_merged_bytes_exact=exact,passed=exact))
    print('OFFLINE_DENSE_BYTES',target,exact,flush=True);del dense;torch.cuda.empty_cache()
    for target in sorted(t for t in store.targets if t.startswith('blocks.0.')):
        q=original.read_tensor(target+'.weight').cuda();s=original.read_tensor(target+'.weight_scale').cuda()
        params=Layout.Params(scale=s,orig_dtype=torch.float32,orig_shape=tuple(q.shape),is_weight=True,convrot=True,convrot_groupsize=256)
        value=Layout.dequantize(q,params)
        # Independent ordinary A/B formula; production merge uses Native LoRAAdapter.
        for a,b,scale in store.terms(target):value.add_(torch.mm(b.cuda().float(),a.cuda().float())*scale)
        qr,pr=Layout.quantize(value,per_channel=True,convrot=True,convrot_groupsize=256,stochastic_rounding=0)
        exact=torch.equal(qr.cpu(),merged.read_tensor(target+'.weight')) and torch.equal(pr.scale.cpu(),merged.read_tensor(target+'.weight_scale'))
        cases.append(dict(label=target,native_merged_bytes_exact=exact,passed=exact))
        print('OFFLINE_BYTES',target,exact,flush=True);del q,s,params,value,qr,pr;torch.cuda.empty_cache()
    saved=dit.dist
    try:
        native=retest.load_native_block('dit',merged.path)
        blocks=[prepare_tp_block(merged,'dit',0,r).cuda() for r in (0,1)]
        for count in (33,257,512):
            xs,args=study.inputs('dit',original.path,count,1,'cuda');x=xs[0];a=args[0]
            ref=native(x.clone(),*a)
            for mode in ('scalemax','scalemax_streamed'):
                enable(mode);comm=LogicalDualGPU(180);dit.dist=comm
                outs=comm.run([lambda r=r:blocks[r](x.clone(),*a) for r in (0,1)])
                score=retest.metrics(outs[0],ref)
                cases.append(dict(label=f'block0_S{count}_{mode}',**score,passed=torch.equal(outs[0],ref) and torch.equal(outs[0],outs[1])))
                print('OFFLINE_BLOCK',count,mode,score['is_bit_exact'],flush=True)
                del outs
            del xs,args,x,a,ref;torch.cuda.empty_cache()
    finally:dit.dist=saved;disable()
    result=dict(passed=all(c['passed'] for c in cases),physical_t4=False,cases=cases,
        scope='Native full-width offline merge/requantization versus independently merged bytes and Native merged block; not online-bypass equivalence',
        source_sha256={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in
            ('patches/h3_lora.py','patches/tp_dit.py','patches/optimized_tp.py','benchmarks/merge_h3_lora.py','benchmarks/qualify_offline_h3_lora.py')})
    (root/'semantic_results/h3_offline_lora.json').write_text(json.dumps(result,indent=2))
    if not result['passed']:raise RuntimeError('Offline Native equivalence failed')


if __name__=='__main__':main()
