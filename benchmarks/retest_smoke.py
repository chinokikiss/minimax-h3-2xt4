"""Small real-kernel smoke test; never used as a two-T4 benchmark result."""
import json
import torch
from retest import metrics, summary, validate_runtime, native_call
import comfy.ops
from comfy.ldm.minimax.model import DiTBlock
validate_runtime()

torch.manual_seed(42)
ops = comfy.ops.mixed_precision_ops({"mixed_ops": True}, torch.float16)
block = DiTBlock(256, 4, 64, 512, 8, 1e-5, 1e-5,
                 apply_silu=False, adaln_dtype=torch.float32,
                 device="cpu", dtype=torch.float16, operations=ops)
sd = {}
for name, shape in {"attn.qkv_proj": (768,256), "attn.out_proj": (256,256),
                    "mlp.fc1": (1024,256), "mlp.fc2": (256,512)}.items():
    sd[name + ".weight"] = torch.randint(-8,9,shape,dtype=torch.int8)
    sd[name + ".weight_scale"] = torch.full((shape[0],1),.0005)
    sd[name + ".comfy_quant"] = torch.tensor(list(json.dumps({"format":"int8_tensorwise", "convrot":True, "convrot_groupsize":256}).encode()),dtype=torch.uint8)
for name, size in {"norm1":256,"norm2":256,"attn.q_norm":64,"attn.k_norm":64}.items():
    sd[name + ".weight"] = torch.ones(size,dtype=torch.float16)
sd["adaln_proj.linear.weight"] = torch.randn(4608,8)*.001
sd["adaln_proj.linear.bias"] = torch.randn(4608)*.001
block.load_state_dict(sd, strict=True)
block = block.cuda().eval().requires_grad_(False)
x = torch.randn(16,256,device="cuda",dtype=torch.float16)
t = torch.randn(1,8,device="cuda",dtype=torch.float16)
with torch.inference_mode():
    angle = torch.randn(1,16,1,24,device="cuda")
    c,s = angle.cos(),angle.sin()
    rope = torch.stack([c,-s,s,c],-1).reshape(1,16,1,24,2,2).half()
    a = block(x.clone(),t,[(0,16,0)],rope)
    b = block(x.clone(),t,[(0,16,0)],rope)
assert metrics(a,b)["is_bit_exact"]
assert not metrics(torch.zeros(5),torch.zeros(5))["valid"]
assert summary([1.,2.,3.])["median_ms"] == 2.

fp32_ops = comfy.ops.mixed_precision_ops({"mixed_ops":True},torch.float32)
fp32_block = DiTBlock(256,4,64,512,8,1e-5,1e-5,apply_silu=False,
                     adaln_dtype=torch.float32,device="cpu",dtype=torch.float32,operations=fp32_ops)
fp32_block.load_state_dict(sd,strict=True)
fp32_block = fp32_block.to("cuda:0").eval().requires_grad_(False)
with torch.inference_mode():
    fp32_ref = native_call("dit",fp32_block,x.float(),t.float(),[(0,16,0)],rope.float())
assert torch.isfinite(fp32_ref).all()
assert fp32_ref.dtype == torch.float32
if torch.cuda.device_count()==2:
    # Deliberately leave the current device at zero, then call a device-one block.
    fp32_block1 = DiTBlock(256,4,64,512,8,1e-5,1e-5,apply_silu=False,
                          adaln_dtype=torch.float32,device="cpu",dtype=torch.float32,operations=fp32_ops)
    fp32_block1.load_state_dict(sd,strict=True)
    fp32_block1 = fp32_block1.to("cuda:1").eval().requires_grad_(False)
    torch.cuda.set_device(0)
    with torch.inference_mode():
        fp32_other = native_call("dit",fp32_block1,x.float().to("cuda:1"),t.float().to("cuda:1"),
                                 [(0,16,0)],rope.float().to("cuda:1"))
    assert torch.cuda.current_device()==0, "Device guard did not restore the caller's context"
    assert metrics(fp32_other.cpu(),fp32_ref.cpu())["rel_l2"]<=1e-6
    del fp32_block1,fp32_other
    print("PASS: two-GPU native extension device guard and FP32 output agreement")
del fp32_block,fp32_ref

from comfy.text_encoders.llama import Qwen3VL_32BConfig, TransformerBlock, precompute_freqs_cis
from comfy.ldm.modules.attention import optimized_attention_for_device
config = Qwen3VL_32BConfig(hidden_size=256, intermediate_size=512,
                         num_attention_heads=4, num_key_value_heads=2)
config.head_dim = 64
config.rope_dims = [12,10,10]
te_block = TransformerBlock(config,index=0,device="cpu",dtype=torch.float16,ops=ops)
te_sd = {}
for name,shape in {"self_attn.q_proj":(256,256),"self_attn.k_proj":(128,256),
                   "self_attn.v_proj":(128,256),"self_attn.o_proj":(256,256),
                   "mlp.gate_proj":(512,256),"mlp.up_proj":(512,256),
                   "mlp.down_proj":(256,512)}.items():
    te_sd[name+".weight"] = torch.randint(-8,9,shape,dtype=torch.int8)
    te_sd[name+".weight_scale"] = torch.full((shape[0],1),.01)
    te_sd[name+".comfy_quant"] = sd["attn.qkv_proj.comfy_quant"].clone()
for name,size in {"input_layernorm":256,"post_attention_layernorm":256,
                  "self_attn.q_norm":64,"self_attn.k_norm":64}.items():
    te_sd[name+".weight"] = torch.ones(size,dtype=torch.float16)
te_block.load_state_dict(te_sd,strict=True)
te_block = te_block.cuda().eval().requires_grad_(False)
x = torch.randn(1,16,256,device="cuda",dtype=torch.float16)
freq = precompute_freqs_cis(64,torch.arange(16,device="cuda").unsqueeze(0),config.rope_theta,
                            rope_dims=config.rope_dims,device=x.device,interleaved_mrope=True)
mask = torch.full((16,16),torch.finfo(x.dtype).min/4,device="cuda",dtype=x.dtype).triu(1)
attn = optimized_attention_for_device(x.device,mask=True,small_input=True)
with torch.inference_mode():
    a = te_block(x.clone(),mask,freq,attn)[0]
    changed = x.clone()
    changed[:,8:] += 10
    b = te_block(changed,mask,freq,attn)[0]
assert torch.equal(a[:,:8],b[:,:8]), "Future tokens leaked through the native causal mask"
print("PASS: native INT8 DiT (FP16/FP32) and TE blocks, causal attention/RoPE, API compatibility, metrics and timings")
