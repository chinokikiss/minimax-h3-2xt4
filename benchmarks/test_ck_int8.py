import torch
import comfy_kitchen

print("Testing comfy_kitchen int8_linear...")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

# Test dimensions for Qwen3-VL-32B
B, S = 1, 128
H = 5120
I = 25600

x = torch.randn(B * S, H, dtype=torch.float16, device=device)
w = torch.randint(-128, 127, (I, H), dtype=torch.int8, device=device)
scale = torch.tensor(0.001, dtype=torch.float32, device=device)
bias = None

try:
    out = comfy_kitchen.int8_linear(
        x, w, scale, bias, torch.float16,
        convrot=True, convrot_groupsize=256
    )
    print("int8_linear with convrot successful! Output shape:", out.shape)
except Exception as e:
    print("int8_linear error:", e)
