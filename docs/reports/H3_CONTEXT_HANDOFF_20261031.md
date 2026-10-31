# MiniMax-H3 2×T4 Infra — Context Handoff

> **截至：2026-10-31；工作区：`C:\Users\Administrator\Documents\project\2xT4 minimax h3`；分支：`main`。**
>
> **用途：** 直接粘贴到新 Chat，继续 MiniMax-H3 Ref2VA 的 Kaggle 2×Tesla T4 推理后端、Sol/Kitchen Attention、全链路 E2E 测试与源码清理。本文区分 **[事实/已验证]、[已提交/未验证运行]、[设计/未验证]、[退役]、[待办]**。历史 Kaggle Notebook 是冻结的当时源码，不能当作今天工作区源码。
>
> **执行边界：** 用户明确要求 **不主动轮询 Kaggle 状态、不抓实时日志、不自动重试/补交 GPU 任务**；只有用户再次明确触发失败调查/结果收集时，才进行一次版本绑定的终态日志与产物下载。不得把“提交成功”写成“GPU 运行成功”，也不得把 Attention Rel-L2 等同于视频主观质量。

## 1. 项目目标与不可变约束

- **[事实]** 仓库：`chinokikiss/minimax-h3-2xt4`；目标是让 MiniMax-H3 **Ref2VA** 在 **Kaggle 2×Tesla T4（SM75、16 GiB/卡、无 NVLink）** 上运行，尽可能降低长序列生成时间与 VRAM 峰值，同时保持生成质量、参考图一致性、时序连续性与数值可控性。
- **[事实]** 核心路径：参考图/提示词 → TE（含 vision）→ DiT TP2 视频与音频联合去噪 → Video VAE / Audio VAE → 音视频导出。长序列主 Attention 的 `N=65166, D=128`；TP rank 实际可有 28 个头。实验常针对每 rank 的 1 或 4 个真实 head 重放。
- **[事实]** 采用 **权重流式窗口、lookahead 预取、CPU/VRAM 分级驻留与接入预算**。用户已经实现这些机制，不得在没有实测等待证据时声称 DiT 长时间等待 Host 或利用率必然低。
- **[事实]** 基座 TP2 采用 INT8/ConvRot 相关数据路径；LoRA 优先**离线烘焙并形成独立 TP2 shard/bundle**，避免 E2E 多次运行中的在线 LoRA 烘焙开销。分片源码、模型版本、输入与 LoRA 必须 hash/pin；离线 bake 与 E2E wall time **分开报告**。
- **[事实]** 受限于 T4/SM75、PCIe 与分布式通讯；生产内核不要无依据升级 CUDA、Torch、Kitchen、TileLang 或同步 ComfyUI 最新源码。历史内核接口曾因版本漂移出现兼容问题。
- **[原则]** 比较实验必须同一真实输入、相同 source binding、只改声明变量。对跨 GPU 的实验只比数值语义，不拿 T4 与本机 SM120 的毫秒直接算加速比。完整输出必须有 batch、rank、per-NFE、VAE、导出与质量证据。

## 2. 工作环境 / 仓库操作

| 层次 | 当前事实 |
|---|---|
| 主工作区 | Windows：`C:\Users\Administrator\Documents\project\2xT4 minimax h3` |
| 本地 Python | `C:\ProgramData\miniconda3\envs\comfyui\python.exe` |
| 本机 GPU | RTX 5060 Ti 16GB，SM120 |
| 本机现查版本 | PyTorch **2.12.1+cu132**；`comfy-kitchen==0.2.37`；`tilelang==0.1.13` |
| Kaggle 已验证版本 | 物理 T4，PyTorch **2.11.0+cu128**，CUDA/nvcc **12.8**，Kitchen **0.2.37**，TileLang **0.1.13**，Sol native ABI **12**；以各 job 绑定的冻结环境为准 |
| Kaggle CLI（本机） | 当次 CLI 回执版本 **2.2.4**，有新版提醒但**不要为了提示随意升级** |
| Git | `main`；**大量未提交的本地修改、删除及 untracked 文件**。绝不可 `git reset --hard`、`git clean -fd`、大范围覆盖、擅自 commit/push |

通过 **CatDesk Workspace** 操作用户本地 Windows 仓库，不能把运行在模型 sandbox 的路径当作用户本地文件。需要访问时先读取 `catdesk_instruction`；工具连接故障时不得虚构“已修改”。应优先使用 CatDesk 专用 read/edit/delete，长测试用 start/poll **仅轮询本地命令进程**，不得借此轮询 Kaggle GPU 状态。

## 3. 生产推理架构与关键源码

**[事实/现态]** 主要模块如下（均为相对仓库根目录路径）：

| 模块 | 作用 |
|---|---|
| `custom_nodes/minimax_h3_tp2/__init__.py` | ComfyUI Loader 参数、服务缓存、用户可选 Attention 后端 |
| `patches/comfy_tp_runtime.py` | `RuntimeConfig`、两张 T4 的持久 worker、权重/VAE 生命周期、AOT、per-NFE 分发 |
| `patches/h3_operator_policy.py` | DiT/TE operator policy、`sol_v2_routing`、保护前缀、准备/执行 Attention、量化 row/column |
| `patches/attention_schedule.py` | 1-based NFE 后端与 `sol@tau` 调度解析 |
| `patches/sol_attention.py` | 自研生产 SM75 Sol v2 All-INT8、编译缓存、prepared Attention、路线统计 |
| `patches/sol_sm75_backend.py` | Sol ABI12 ctypes/native bridge、Q/K/V quant、K-centroid、workspace 等 |
| `patches/sol_attention_native/` | SM75 CUDA/C++ 实现；重要：`sla_sm75_sparse.cu`、`sla_sm75_fp16_kernel.cuh`、`sol_sm75_preprocess.cuh` |
| `patches/kitchen_attention.py`, `patches/kitchen_te_attention.py` | Kitchen dense INT8、TE causal Attention |
| `patches/parallel_vae.py`, `patches/staged_vae.py`, `patches/video_decode_planner.py` | Video VAE 双 rank 查询/时序解码、调度、回退 |
| `patches/windowed_tp.py`、`patches/tp2_bundle.py`、`patches/tp2_disk_bake.py` | TP2 权重流式窗口、缓存/预取、bundle 与离线烘焙 |
| `patches/h3_operator_aot.py`, `patches/h3_tensor_flow.py` | SM75 AOT 与残差/激活精度边界 |
| `patches/ptd/{bank,runtime,nodes}.py` | PTD/RefAsymOPD bank 与 sampling 接入；不能视为已质量验收 |
| `benchmarks/ref2va_e2e_run.py`、`benchmarks/run_h3_l1_turbo_e2e.py` | E2E 观测器、per-NFE dispatch、完整链路执行及验收 |
| `docs/guides/{SOL_ATTENTION,H3_OPERATOR_INTEGRATION}.md` | 当前生产策略文档 |
| `docs/data/h3_sol_v2_active_source_identity.json` | **当前五份 Sol 核心源码 SHA256**；替代已删的 math-fix 回滚备份，不是历史实验源码哈希 |

### 3.1 Attention / 层级分发

- **TE**：默认 `kitchen_causal`，`native` 可选；TE causal/padding/additive mask 需要正确性约束。
- **DiT**：默认 Kitchen INT8；可选 `sol`（仅 SM75/T4、明确 fail-closed）或 Native；按 `dit_attention_schedule` 在 1–8 NFE 切换，`sol@tau` 可覆盖全局 `sol_tau`。典型生产对照：`1-3:kitchen,4-8:sol@1.0`。DiT token refiner 仍走 Native SDPA；不要误套 Sol。
- **Video VAE — 最新决定**：**只允许 `video_attention='kitchen'` 或 `'native'`**；已从 UI、RuntimeConfig、`parallel_vae`、专属 AOT 与观测器移除 Video VAE Sol 分支。默认 Kitchen INT8，Native 强制 PyTorch SDPA；`native_temporal_pairs` 与 Query 分区仍独立存在。**不要恢复 Sol-VAE 的 D64 兼容 wrapper**。
- **Audio VAE**：Native、单 GPU，和 DiT/Video VAE backend 不同。
- **Sol v2 路由边界特别注意**：给定 `[text 0:2528, ref1 2528:3344, ref2 3344:4164, audio 4164:4974, video 4974:65166]`，旧函数 `protected_prefix()` 返回 **4164**（仅纯 conditioning）；当前 v2 `sol_v2_routing()` 返回 **4974**（conditioning + generated audio），同时限定 Video topology `start_token=4974`、`frame_stride_tokens=836`、`temporal_radius=1`、`local_radius_tokens=64`、frame boundary protection。**两函数不可混同。** native Q64/K64 对不齐边界作向外取整保护。
- **Sol ABI12**：D128 和原生 D64 specialization、prepared Attention、INT8 Q/K/V、K-centroid 路由、sync-free LUT、local-Q/full-KV 原语；D64 的 native 支持**不意味着生产 Video VAE 仍启用 Sol**。
- **仍保留但非默认**：`H3_SOL_WORKSPACE_POOL=1` 的可复用显存池、`H3_SOL_LUT_BITMAP=1` 的 maskless Bitmap LUT；应独立验收，未获得长序列稳定速度收益。
- **[本次退役] 分片 KC 阈值融合**：`H3_SOL_FUSED_KC_THRESHOLD`、`_fused_kc_enabled`、`compute_kc_stats_threshold_fused` 的定义及 full-Q/local-Q 分发已从当前生产源码移除。当前 `sol_sm75_preprocess.cuh` 两条路径都回到 **`reduce_kc_stats` + `compute_diag_threshold`**，正常 K-centroid 与 KC 路由**不能删除**。不再把旧 fusion 宏当生产配置。
- **[本次退役] Sol math-parity 精度补丁**：试过多处数值改动，但真实 H3 双 rank 的相对误差未获得稳健改善；先前恢复原 Sol v2，后又删除该精度补丁的报告、失败实验/备份目录及独立脚本。**当前源码不是数学补丁版**。不要从原已删 `docs/data/h3_sol_pre_math_restore_20261031` 读取 manifest。

### 3.2 其他仍有效的工程决策

- **权重/I/O**：权重流式窗口和 lookahead 已实施；离线 bake 保存与完整性验证后优先按生命周期清理多余同步、重复 sha、`empty_cache` 与 `gc.collect`，但不得无 benchmark 证据删掉真正的 correctness gate。
- **量化/计算**：TileLang `tilelang_scaled_fp16` row transport、`cuda_int8_tilelang` column、`tilelang_fp32` RMS/RoPE、`fused_stored`/可选 `fused_recompute` SwiGLU + ConvRot，FP32 数值岛用于防溢出；不要将 RoPE/softmax 等敏感阶段一概强制 FP16。
- **LoRA**：`lightx2v/Minimax-h3-Turbo` 常规 **8-step** 模型经离线烘焙使用；PTD/RefAsymOPD 是另一个 opt-in 方向，必须先确认 pruned AdaLN curves 与 PTD 对原始权重的依赖兼容，不能假设可无损移植。用户倾向独立 Notebook 离线 bake LoRA 后挂 Dataset，减少在线烘焙/重复上传。

## 4. 标准长序列 E2E 输入（可复现）

| 字段 | 冻结对照值 |
|---|---|
| Model | MiniMax-H3 Ref2VA，TP2 base：`cnmds/minimax-h3-tp2-int8-561a6211` |
| 分辨率 | **1216×704 = 856,064 pixels = 0.856064 MP** |
| Native 帧 / export | 243 帧 decoded；240 帧 × 24 FPS = **10.0 秒导出** |
| NFE / Sampling | **8**，Euler / simple，CFG **1.0** |
| Shifts | Video **6.0** / Audio **3.0** |
| Prompt / refs | 固定二次元：神社、紫发巫女、白狐灵；**两张参考图**，内容及 SHA256 用原 request 的 pin，勿改写 |
| LoRA | `lightx2v/Minimax-h3-Turbo` / `minimax_h3_ref2v_turbo_8step_v1.0_768p_comfyui_bf16.safetensors` |
| LoRA commit / scale | `30f485ed874aa44c8183348ad361c5eabad318da` / 1.0，208 个目标，offline baked |
| 原比较 seed | `20261020` |
| **最新实验 seed** | **`987654321`**，其他外部输入不变 |
| PackedLayout | 65,166 total tokens；Video 60,192 tokens，72 latent frames × 836 tokens |

权威 workload 文件：`docs/data/h3_turbo_085mp_10s_20261029/gpu_v2_final/turbo_085mp_e2e/retry_request.json` 及对应 `generation/result.json`；冻结环境：`docs/data/h3_turbo_085mp_10s_20261029/provider_v2_source/`。不要从本文重建 prompt 文本或 reference SHA256，以 request 原文件为准。

## 5. 已完成并实际验证的 E2E 结果

**[事实/已完成]** 以下数字是不同时间 Kaggle 运行记录的 wall times；可以作为历史参考，但**不是严格只改一个算子的实验因果估计**。尤其 2026-10-30 的旧 E2E 仍使用 **Video VAE Sol**，这是历史行为，**不代表今天生产线允许**。

| 2026-10-30 实验 | DiT NFE | Video VAE | Sampler | Foreground | E2E worker（离线 bake 不含） |
|---|---|---|---:|---:|---:|
| All Kitchen 参考 | 全 Kitchen | Kitchen | **1348.776 s** | **1639.157 s** | **1703.296 s** |
| Old Sol Hybrid v4 | 1,8 Kitchen；2–7 Sol | Sol | **1191.284 s** | **1497.299 s** | **1549.299 s** |
| Sol v2 Production E2E v1 | 1,8 Kitchen；2–7 Sol | Sol | **1191.718 s** | **1458.364 s** | **1506.675 s** |

Sol v2 Production 相对 All Kitchen 的 sampler 时间少 **157.058 s（11.64%）**，foreground 少 **180.793 s（11.03%）**；不同作业的 TE/I/O/视频解码时差**不能归因于 DiT Sol 内核**。其每 rank DiT 调度实测 **100 Kitchen + 300 Sol**；8/8 denoise + sampler 完成，Video VAE Native temporal pairs 无 fallback，2016 Sol 视频 Attention 调用/rank（历史版），输出 pixels `[1,243,704,1216,3]`、waveform `[1,324000,2]`，有效 10 s 音视频导出。离线 bake 独立 **328.539 s**。

更多结果与制品：`docs/reports/H3_SOL_V2_PRODUCTION_E2E_20261030_RESULTS.md`、`docs/reports/H3_SOL_HYBRID_085MP_10S_T4_20261030_RESULTS.md`。历史 Run 的 latent SHA256 不一致只能证明**不 bitwise identical**，无法判断哪个画面更好；人工视频评分、时序连贯/参考角色一致性仍待完成。

## 6. 2026-10-31 已提交但未回收终态的长序列 GPU 任务

**[已提交/未验证运行]** 以下仅 Kaggle provider **accepted**；除非用户明确要求，不做状态查询、自动下载或重跑。

| Job / URL | Version | Seed | DiT 1–8 NFE | Video VAE | 当前可证明 |
|---|---:|---:|---|---|---|
| [KC1–3 / Sol4–8 / VAE KC 历史混合](https://www.kaggle.com/pamcks/h3-kc1-3-sol4-8-vae-kc-long-e2e-20261031) | **v4** | 20261020 | 1–3 Kitchen，4–8 Sol v2 τ=1 | Kitchen | accepted；**存在旧 `id_no` 可能复用了 Notebook 身份**，不可假装新独立 Notebook |
| [KC All 长序列对照](https://www.kaggle.com/code/pamcks/h3-kc-all-e2e-20261031) | **v1** | 20261020 | 全部 Kitchen | Kitchen | accepted，独立 v1 |
| [最新改 Seed 的混合长序列](https://www.kaggle.com/code/pamcks/h3-kc-sol-seed987654321-e2e) | **v1** | **987654321** | **1–3 Kitchen，4–8 Sol v2 τ=1** | **Kitchen** | **accepted，独立 v1；当前最高优先级待验收** |

- **最新 seed job**：提交 UTC `2026-10-31T06:18:49Z`，申请 2×T4，Notebook SHA256 `a9cff3db6bc213b349aa2bfe7d657a078d6011f8390f0437a25c0eba2d8ca083`。本地提交前 **35/35** checks PASS。Kaggle receipt `status=accepted`、`version=1`、`final_collection_count=0`；没有任何运行完成证明。
- 最新 Job 的 frozen notebook overlay **只在原 20261020/20261021 seed allowlist 上增补 987654321**，不改主工作区 `benchmarks/turbo_challenge_contract.py`；外部输入与前一混合 E2E 相比仅 seed（以及非语义 `name/source_note`）不同。**Seed 数值差很大不意味着视觉差异程度必然更大**。
- 预期而非已验证：每 rank DiT **150 Kitchen + 250 Sol** 调用（50 blocks × 8 NFE），正常 Video VAE Kitchen 解码。
- **重要版本边界**：以上已提交 Notebook 使用提交当时**冻结的 Sol v2 源码**，包含关闭状态的 KC fusion 编译宏；今天工作区已经**彻底退役该融合代码**。不要以当前 SHA256 覆盖历史 notebook 的 source binding，也不要对已接收 v1 作静默更新。

关键本地路径：
- `docs/reports/H3_KC_SOL_SEED987654321_LONG_E2E_SUBMISSION_20261031.md`
- `docs/data/h3_kc_sol_seed987654321_long_e2e_20261031/{request.json,gpu_submission.json,preflight.log}`
- `docs/data/h3_kc_sol_seed987654321_long_e2e_20261031/gpu_job/{experiment.ipynb,source_binding.json,kernel-metadata.json}`
- `benchmarks/{build_h3_kc_sol_seed987654321_long_e2e.py,test_h3_kc_sol_seed987654321_job.py,submit_h3_kc_sol_seed987654321_long_e2e.py}`
- 另两份 Notebook 的 `docs/data/h3_kc_sol_8step_long_e2e_20261031` 和 `docs/data/h3_kc_all_8step_long_e2e_20261031`。

## 7. 真实 H3 Attention：Sol v2 vs ComfyUI 内置 Sol 语义实验

**[事实/已完成，虽然 Kaggle Notebook 标记 Failed]**

- 实验 Notebook：[H3 Real Sol Comfy Capture v1](https://www.kaggle.com/code/pamcks/h3-real-sol-comfy-capture-20261031)。
- Kaggle 失败点 **不是 GPU 计算**：2 个 T4 rank 均完成实际 first-DiT Attention 的真实 **post-RMS/post-RoPE** QKV 截获及旧 Sol v2 输出保存；旧 batch 汇总器误要求 `rank['summary']`，但新 schema 没有该字段，于最终汇总抛出 **`KeyError: 'summary'`**。已在本机取回有效数据，**无需重跑 GPU**。
- 每 rank 取真实 head0：FP16 `[1,1,65166,128]` 的 Q/K/V + T4 Sol v2 输出，`rank0_real_qkv.pt` 与 `rank1_real_qkv.pt` 各 **66,732,515 bytes**，文件和 Tensor SHA256 全通过。
- 本机 RTX 5060 Ti SM120 对**相同序列化 QKV**使用 low-level `comfy_kitchen.sol_attn`，每 rank 4 组设置；共 **8 组成功重放**。Comfy Sol 要求 SM80+，T4 Sol 是 SM75；不可能不移植就做同 GPU 原生 A/B，不能比较跨硬件耗时。
- **ComfyUI Sol 相对 T4 Sol v2 的 overall Rel-L2：**

| Comfy Sol 低层设置 | Rank 0 | Rank 1 |
|---|---:|---:|
| τ=1.0，匹配保护前缀，token_aug=0 | **1.3055%** | **2.6213%** |
| τ=1.0，Comfy H3 sink_q，extra=0 | 4.2120% | 3.4228% |
| τ=1.0，Comfy H3 sink_q，extra=256 | 3.4064% | 2.9910% |
| τ=1.3，Comfy H3 sink_q，extra=256 | 5.2645% | 4.7708% |

- 抽取 **85 个真实 Query**，使用**完整真实 K/V**算局部查询 FP32 Dense，作为共同数值参考：

| 输出 vs sampled FP32 Dense，Rel-L2 | Rank 0 | Rank 1 |
|---|---:|---:|
| T4 Sol v2，τ=1 | **5.3483%** | **2.8443%** |
| Comfy Sol，匹配 sink，τ=1，extra0 | 6.0101% | 4.0387% |
| Comfy Sol，H3 sink，τ=1，extra256 | 7.2490% | 8.8262% |
| Comfy Sol，H3 sink，τ=1.3，extra256 | 9.2438% | 9.8780% |

- **结论边界**：在这两个头、第一层、85 个非随机/非全覆盖采样 Query 上，旧 Sol v2 比所测 Comfy 低层 Sol 配置更贴近 FP32 Dense；**不是完整视频质量优劣的证明**，也未测 Comfy H3 特有的 `sol_attn_chunked` QKV projection 分块/跨步 kmean/vscale 复用。Sink/extra/tau 参数会显著改变结果。
- 保存路径：`docs/reports/H3_REAL_SOL_COMFY_V1_FAILURE_RECOVERY_RESULTS_20261031.md`，`docs/data/h3_real_sol_comfy_capture_20261031/{recovered_v1_result.json,local_comfy_replay/result.json}`；真实 `.pt` 在 `failed_v1_payloads/real_h3_sol_comfy_capture/generation/rank_results/`；工具 `benchmarks/{real_h3_cross_backend_capture.py,replay_real_h3_comfy_sol.py,recover_real_h3_cross_backend_result.py}`。**原 provider Failed 状态不得篡改**。

## 8. 其他已验证的 Sol 实验与已淘汰方向

### 8.1 Bitmap LUT（保留 opt-in，不要误删）

- **[事实]** `docs/reports/H3_SOL_BITMAP_LUT_AB_20261031.md`：Kaggle 物理 T4、CUDA12.8、Sol ABI12；D64 与 65,166-token D128 synthetic 两组 **ExactMask、RowCount、LUT 有效逻辑序列完全一致；输出 Rel-L2 = 0**。
- D128 原始 int32 LUT → Bitmap：GPU temp peak **189,895,168 B → 169,445,376 B**（**−10.77%，约 19.50 MiB**）；完整调用 **139.4679 → 140.0346 ms**（**+0.41% 变慢**）；主 execute **+4.50% 变慢**。**说明节省显存，但不提高这组长序列速度**。
- 这只是 synthetic-shaped operator A/B，不是两 rank Real H3 E2E。因收益不稳，**默认仍旧 int32 LUT；bitmap 用 `H3_SOL_LUT_BITMAP=1` opt-in**。不要从“route 完全相同”推断完整视频与 Dense 同质。

### 8.2 分片 KC 阈值融合（已退役）

- **[事实/退役]** `compute_kc_stats_threshold_fused` 分片 KC 统计 + 阈值融合在 physical T4 v2 实验中保持 route exact，但 **D128 基本无稳定收益**：约 −0.011%、+0.258%、+0.024% 的不同分区 full-call 变化（测试噪声量级），D64 proxy 曾有约 +7.30% 小工作负载优势，**不足以推动通用生产化**。
- 用户要求彻底移除。**当前 native/CUDA/编译缓存开关均已清理**；相关专属构建器、probe、报告及数据/备份目录已删除。保留 baseline 的 KC statistics + diagonal threshold。**不要重建该融合分支**。

### 8.3 数学精度“优化一大堆但没提升”（已退役）

- **[历史事实/原证据已按用户要求删除]** 对真实 H3 相对 Kitchen 的 Sol 输出 Rel-L2 约 **4.5%**，math parity 修改没有在两 rank 上持续改善；部分阶段 Rank1 性能回退，结果不支持推广。
- 回滚到旧 Sol v2 后，本轮删除报告、math-smoke/Math-parity 实验目录、补丁备份、回滚脚本。保留**今天生产五文件源码 SHA256**在 `docs/data/h3_sol_v2_active_source_identity.json`。不应假定可再从旧备份恢复 math-fix。
- 历史冻结 Notebook 中可能仍含融合或 math-fix 代码，这是**历史证据而非当前生产选项**，不要随意用新源码覆写它们。

## 9. 本轮源码清理与验证（2026-10-31）

**[本地已执行]**
1. `Video VAE Sol` 的 Loader 选项、RuntimeConfig 允许值、`parallel_vae` 的全量/分区 Sol Attention、Video AOT 与相关 E2E 断言已移除；默认 Kitchen，Native 可选。
2. Sol native `compute_kc_stats_threshold_fused` 和两处 `#if H3_SOL_FUSED_KC_THRESHOLD` 已删；`patches/sol_attention.py` 的宏、开关与编译 cache key 分支已删。普通 `reduce_kc_stats` 和 `compute_diag_threshold` 两条路径继续存在。
3. Bitmap probe 从融合实验脚本中分离为 `benchmarks/sol_bitmap_route_probe.py`，保留 Bitmap / workspace 非融合实验能力。
4. 删除 math-parity/restore 目录与报告、KC fusion 各实验目录/报告/专属脚本；替换依赖旧 rollback manifest 的验证为当前 source-identity JSON；更新生产文档及旧 builder 中的 Video VAE backend。
5. **两次本地回归**：先 **44/44 PASS**，再扩展 **50/50 PASS**（包含 Sol、VAE、配置、E2E wiring/source closure）；相关 Python `py_compile` PASS。**未在本次清理后重新执行物理 T4/CUDA12.8 E2E**。
6. `git status` 当前有大量此前已存在或本轮形成的修改/删除/untracked（包括训练/推理相关文件），**未提交 commit/未 push**。不得误以为所有 git diff 都是此次清理所致。

**注意：** 已提交并仍冻结的旧 Notebook 不受本地清理影响；它们对应提交时 source binding，允许保留已淘汰功能的快照，不代表当前代码能启用。若在清理后重建旧实验，需要**新 source binding/测试结果**，不要要求与旧 binding 字节相同。

## 10. 尚未解决的问题与后续优先级

### P0 — 仅在用户主动触发时收集/报告最新长序列终态

优先处理最新 `pamcks/h3-kc-sol-seed987654321-e2e` **v1**。只做一次**version-bound 终态产物**下载/解读，不轮询。核对 batch/generation `status`、GPU 身份、hash、TP2 数量、`8/8` NFE、每 rank **150/250** DiT calls、Video VAE Kitchen dispatch、音视频解码/导出、各阶段 wall time、sampler、offline bake 分离及 artifacts。若失败，先读取失败终态日志，不立刻补交新 GPU。

用 `benchmarks/kaggle_tls12_runner.py` 包装 `benchmarks/download_kaggle_evidence.py`，按实际 Kaggle URL slug 和 **`--version 1`** 绑定下载；先核对该工具当时的 CLI 参数/下载规则，再调用。目标 evidence 目录：`docs/data/h3_kc_sol_seed987654321_long_e2e_20261031/gpu_final_v1/`。同理另两个 submitted 对照应按实际 URL/v4、v1 下载，不可错读 intended metadata ID。

### P1 — 三组可解释的对照

在**相同 seed 20261020**的前提下比较全 Kitchen 与 1–3 Kitchen/4–8 Sol，Video VAE 同为 Kitchen；检验 sampler/E2E、per-NFE 延迟与质量。然后对比 `seed=20261020` 与 `987654321` 的混合路径，观察是否出现特定 seed 触发的角色、构图、运动或音视频对齐崩坏。**不能只凭两次 seed 判定普遍崩坏概率**，必要时预注册样本数量和判别尺度。

### P2 — 数值/画面质量验收

对生成视频逐项评估：参考人物/白狐身份保持、背景结构、伞/手部/眼睛、运动连贯、跨帧/跨秒跳变、音频节奏与同步。可用盲评或固定描述性指标；主观效果必须配合实际导出媒体，不能由 SHA256 或 Attention 采样 Rel-L2 替代。若怀疑 Sol 中长程相关性问题，应优先做**同真实 QKV、相同 Q/K/V、不同 NFE/层**对照；现有真实捕获仅 first attention + one head/rank。

### P3 — 生产稳定性与性能

- 对清理后的最新源码另起小型物理 T4 编译/回归测试，确保无 Sol-VAE 调用、不引用融合宏，Sol ABI12 编译成功、DiT 混合策略无回退；已有 **50/50 CPU** 是静态/局部保证，不能冒充物理 GPU 验收。
- Benchmark `H3_SOL_WORKSPACE_POOL` 的**实际 cold/steady/resident peak**，不能把长期驻留内存藏在峰值基线里；默认不启用。
- Bitmap 已证实 D128 内核速度轻微退化，优先找真正影响 worst-rank latency 的瓶颈，或开展更合理的 live buffer lifetime/重计算设计；不能把理论内存节省当直接 E2E 加速。
- 继续检查权重窗口、lookahead、TE page-cache/O_DIRECT、Kaggle disk-bake；**只有对应日志与测量**才可声称 Host I/O 或 cache 是瓶颈。PCIe/无 NVLink 的 TP2 通讯成本须显式量化。

### P4 — PTD / LoRA 长期工作流（仍有效，但不是本次 Sol 结果）

- `taobao-content-ai/MiniMax-H3-PTD-RefAsymOPD-LoRAs` 迁移目标：offline bake 多份 DiT / per-hop LoRA bank，并与流式权重窗口及现有 TP2 sampler 整合。
- **关键未决**：当前 TP2 base 的 pruned AdaLN curves 与 PTD 参考实现拒绝该 base 变体，需原始非 curve 权重或显式验证近似；不能宣称 bitwise 兼容。
- 用户倾向将高成本的普通/PTD LoRA 全部离线准备为独立 Kaggle Dataset，运行时挂载并按窗口加载。PTD 适配是否提升视频质量需要独立 E2E 证明。

## 11. 新 Chat 的执行规则

1. **先读用户当前指令与本快照，不要重复询问 seed/模型尺寸/NFE/设备**；除非用户明确改变其中某项。
2. **严守 Kaggle 不轮询规则**；用户说“任务失败”“结果出来了”时视为一次性收集已知终态证据的授权，不等于无限次查询或自动补交。
3. **用证据等级表述**：提交回执 ≠ GPU 完成；CPU 测试 ≠ SM75 CUDA 编译；单层 QKV 输出误差 ≠ E2E 质量；数据 SHA 不同 ≠ 视频崩坏；不同 Kaggle E2E wall time ≠ 纯内核因果改善。
4. **不恢复退役功能**：Video VAE Sol、分片 KC 统计阈值融合、无效 math-parity 补丁/备份均已按用户要求清理。保留 DiT Sol v2、Kitchen VAE、workspace pool/bitmap opt-in，以及真正需要的正常 KC 路由。
5. **保持历史实验可追溯**：provider URL、notebook version、request.json、source binding、源指纹与核验结果相互对应。不要用最新源码重写已提交 notebook。
6. **不破坏现有 workspace**：先查 git status 和定向文件；不 reset、不 clean、不盲删任意其他实验；若未获授权不 commit/push。
7. **后续输出习惯**：用户要求“报告”时重点写**实测 E2E 输出与计时**而不是泛谈瓶颈；工程方案应标注已做/计划/未经证实并保留路径与验证门槛。

---

**一行接续提示：** 当前主要目标是**用户触发后**收集并比较 seed=`987654321` 的 `KC NFE1–3 / Sol v2 NFE4–8 / Video VAE Kitchen` 完整 0.85MP/10s E2E 结果；生产源码已移除 Video VAE Sol、分片 KC 阈值融合及无效 math-fix 备份，保留 DiT Sol ABI12/正常 KC 路由、Bitmap/Pool opt-in，最新 CPU 回归 50/50 PASS，**尚缺本轮清理后的物理 T4 全链路验证**。
