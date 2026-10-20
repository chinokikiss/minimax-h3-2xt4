"""Independent ordinary A/B reference for Native offline INT8 LoRA merging."""
import json
import torch


def verify_weight(reader,store,target,actual,actual_scale,device='cuda:0'):
    from comfy_kitchen.tensor.int8 import TensorWiseINT8Layout as Layout
    q=reader.read_tensor(target+'.weight').to(device)
    dtype=q.dtype
    if dtype==torch.int8:
        metadata=json.loads(bytes(reader.read_tensor(target+'.comfy_quant').tolist()).decode())
        if metadata!={'format':'int8_tensorwise','convrot':True,'convrot_groupsize':256}:
            raise ValueError('Unsupported reference quantization')
        scale=reader.read_tensor(target+'.weight_scale').to(device)
        params=Layout.Params(scale=scale,orig_dtype=torch.float32,orig_shape=tuple(q.shape),
            is_weight=True,convrot=True,convrot_groupsize=256)
        weight=Layout.dequantize(q,params)
        del scale,params
    else:
        weight=q.float()
    del q
    for down,up,strength in store.terms(target):
        # Independent formula, not production LoRAAdapter.calculate_weight.
        delta=torch.mm(up.to(device,dtype=torch.float32),down.to(device,dtype=torch.float32))
        weight.add_(delta*strength)
        del delta,down,up
    if dtype==torch.int8:
        expected,params=Layout.quantize(weight,per_channel=True,convrot=True,
            convrot_groupsize=256,stochastic_rounding=0)
        exact=torch.equal(expected.cpu(),actual) and torch.equal(params.scale.cpu(),actual_scale)
    else:
        exact=actual_scale is None and torch.equal(weight.to(dtype).cpu(),actual)
    if not exact:raise AssertionError('Independent Native offline bytes differ: '+target)
    return True
