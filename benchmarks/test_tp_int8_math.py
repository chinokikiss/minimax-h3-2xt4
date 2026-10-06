import torch
import comfy_kitchen

device = "cuda" if torch.cuda.is_available() else "cpu"
B, S = 1, 64
H = 5120
I = 25600

x = torch.randn(B * S, H, dtype=torch.float16, device=device)
w = torch.randint(-128, 127, (I, H), dtype=torch.int8, device=device)
scale = torch.tensor(0.001, dtype=torch.float32, device=device)

# 1. Full Column Parallel check
out_full = comfy_kitchen.int8_linear(x, w, scale, None, torch.float16, convrot=True, convrot_groupsize=256)

w_0 = w[:I//2, :]
w_1 = w[I//2:, :]
out_col_0 = comfy_kitchen.int8_linear(x, w_0, scale, None, torch.float16, convrot=True, convrot_groupsize=256)
out_col_1 = comfy_kitchen.int8_linear(x, w_1, scale, None, torch.float16, convrot=True, convrot_groupsize=256)
out_col_cat = torch.cat([out_col_0, out_col_1], dim=-1)

diff_col = torch.max(torch.abs(out_full - out_col_cat)).item()
print(f"Column Parallel diff: {diff_col}")

# 2. Row Parallel check (e.g., down_proj: in_features = 25600, out_features = 5120)
w_down = torch.randint(-128, 127, (H, I), dtype=torch.int8, device=device)
x_mlp = torch.randn(B * S, I, dtype=torch.float16, device=device)
out_down_full = comfy_kitchen.int8_linear(x_mlp, w_down, scale, None, torch.float16, convrot=True, convrot_groupsize=256)

x_mlp_0 = x_mlp[:, :I//2]
x_mlp_1 = x_mlp[:, I//2:]
w_down_0 = w_down[:, :I//2]
w_down_1 = w_down[:, I//2:]

out_down_0 = comfy_kitchen.int8_linear(x_mlp_0, w_down_0, scale, None, torch.float16, convrot=True, convrot_groupsize=256)
out_down_1 = comfy_kitchen.int8_linear(x_mlp_1, w_down_1, scale, None, torch.float16, convrot=True, convrot_groupsize=256)
out_down_sum = out_down_0 + out_down_1

diff_row = torch.max(torch.abs(out_down_full - out_down_sum)).item()
print(f"Row Parallel diff: {diff_row}")
