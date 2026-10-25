"""Publish compact operator evidence; never loads a model or launches CUDA."""
import argparse,hashlib,json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser();p.add_argument('artifacts',type=Path);args=p.parse_args();base=args.artifacts
    load=lambda name:json.loads((base/name).read_text(encoding='utf-8'))
    full=load('screen_s7895.json');lora=load('screen_s513_lora.json');fold=load('norm_fold.json');fixture=load('fixture.json')
    assert full['status']==lora['status']=='complete'
    for state in (full,lora):
        for path,sha in state['source_sha256'].items():
            assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==sha,path
        for case in state['cases']:
            if 'ranks' in case:assert case['output_sha256'][0]==case['output_sha256'][1]
        for name in ('fp32_control','fp32_gauge_c16','fp32_fold_c16','fp32_normboost64_control'):
            case=next(r for r in state['cases'] if r['name']==name)
            assert case['output_sha256']==state['reference_sha256'],name
    rows={x['name']:x for x in full['cases']};old=load('screen_s7895_v1.json')
    repeated=[r['name'] for r in old['cases'] if r['name'] in rows]
    repeat_equal=all(r['output_sha256']==rows[r['name']]['output_sha256'] for r in old['cases'] if r['name'] in rows)
    assert repeat_equal
    data=dict(date='2026-10-20',status='exploratory_not_promoted',hardware_scope=full['scope'],
        local_artifacts=str(base),fixture=fixture,full_sequence=full,lora_subset=lora,norm_gain_fold=fold,
        repeat=dict(shared_cases=len(repeated),output_sha256_identical=repeat_equal,
                    scope='two fresh local processes, same device/fixture; not independent T4 sessions'),
        result_sha256={n:hashlib.sha256((base/n).read_bytes()).hexdigest() for n in ('screen_s7895.json','screen_s513_lora.json','norm_fold.json','fixture.json')})
    target=ROOT/'docs/data/fp16_residual_fix_local_20261020.json'
    target.write_text(json.dumps(data,indent=2),encoding='utf-8')
    control=rows['fp32_control']['observations'][0]
    lines=['# FP16 residual fix and weight folding: local operator study','',
        '**Result: worth retaining as an isolated candidate, not ready for a default change.** Scaled FP16 removes a real first-block overflow, but it does not remove residual-update absorption or ordinary FP16 rounding. Fixed scale folding can avoid changing INT8 weight bytes. Full channel-wise RMSNorm-gain folding adds requantization error.','',
        '## Scope and reproducibility','',
        '- Actual hardware: one RTX 5060 Ti, SM120; Torch 2.12.1+cu132, Kitchen 0.2.37 and the frozen conda environment. All 94 package versions match the preserved snapshot. No package or source API was updated.',
        '- The current pinned Native front end reconstructs block-0 inputs from retained real Tokyo challenge conditioning, reference latents and initial noise at sigma=1. The source workload records 512x512, 90 frames, seed 20261013. This is the challenge fixture, not the earlier 640x352 workload.',
        '- Only conditioning projection/text-refiner/input-packing operators execute during preparation. Execution is intercepted before the first main DiT block. No TE or VAE inference, 50-block DiT, sampler trajectory, full generation or Kaggle submission is run.',
        '- Full retained shape is **[7895,5376]**: 417 text, 266 reference-image, 300 audio and 6912 generated-video tokens. All 7895 tokens participate in the main block-0 screen. A separate LoRA test takes 513 order-preserving, uniformly selected tokens; that subset changes attention context and is explicitly not full-sequence validation.',
        '- Two logical TP ranks execute real rank-owned trained weights on the same GPU with synchronized mock collectives. Rank agreement is checked, but neither those collectives nor SM120 timings qualify T4/NCCL performance.',
        '- The active SM75-only TileLang row backend falls back to Native on this GPU. These are current logical TP2/Native-Kitchen arithmetic probes, not execution tests of FP16 TileLang SM75 kernels. Current packed/SwiGLU/owned-FP32 kernels also have FP32 guards; an FP16 deployment needs separate dispatch/kernel work.',
        f'- There are 18 complete-sequence and 18 LoRA-subset configurations. {len(repeated)} shared configurations reproduced their output hashes across two fresh full-sequence processes. Compile/loading and diagnostic reductions are not timed; no speed or measured pipeline-VRAM claim is made.','',
        '## Exact gauge, including epsilon','',
        'For a positive constant `c = 2**k`, store `y = x/c`. A pre-normalized residual block becomes:','',
        '```text','RMSNorm_eps(x) = RMSNorm_(eps/c^2)(y)',
        'h = AdaLN(RMSNorm_(eps/c^2)(y))',
        'y_next = y + gate * branch(h) / c','```','',
        'The same epsilon transformation applies to LayerNorm; its affine beta stays unchanged. RMSNorm, AdaLN, timestep curves, Q/K norm/RoPE and residual-update arithmetic remain FP32 islands in the prototype. The FP16-stream variant rounds at storage boundaries. The FP16-branch variant additionally uses FP16 projection/attention/MLP inputs and outputs, with INT32 base GEMM SUM retained.',
        '', 'Leaving epsilon unchanged is wrong near zero. Storing adjusted epsilon in FP16 is also wrong: `1e-5 / 256^2` rounds to zero in FP16. Nonlinear SiLU, AdaLN shifts, output biases and LoRA contributions cannot be scaled away without compensation.',
        '', 'The FP32 gauge control, FP32 output-scale fold and FP32 scalar norm-gain fold all match the current TP2 block **byte-for-byte** in the tested full-sequence and LoRA-subset cases. This validates the transformation controls for these cases, not FP16 equivalence or universal floating-point identities.',
        '', '## Overflow, underflow and lost updates','',
        f'The unscaled input maximum is {fixture["range"]["maximum"]:.6f}. After attention, the baseline residual maximum is {control["attention_residual"]["maximum"]:.6f}; after the MLP it is **{control["mlp_residual"]["maximum"]:.6f}**, beyond FP16\'s maximum 65504. Exactly one baseline output element exceeds that range. Unscaled FP16 storage produces one infinity after the MLP in each logical rank.',
        '', '| Residual divisor | Nonfinite output elements/rank | Initial nonzero values cast to zero | Attention updates rounded away | MLP updates rounded away | Block relative L2 |',
        '|---:|---:|---:|---:|---:|---:|']
    for c in (1,4,16,64,256,4096):
        r=rows[f'fp16_store_c{c}'];o=r['observations'][0];score=r['ranks'][0]
        err='not finite' if score['relative_l2'] is None else f'{score["relative_l2"]*100:.4f}%'
        lines.append(f'| {c} | {score["nonfinite"]} | {r["input_cast"]["cast_nonzero_to_zero"]:,} | {o["attention_residual"]["rounded_away_updates"]:,} | {o["mlp_residual"]["rounded_away_updates"]:,} | {err} |')
    a=rows['fp16_store_c16']['observations'][0]['attention_residual'];m=rows['fp16_store_c16']['observations'][0]['mlp_residual']
    lines+=['',f'At c=16, approximately **{a["rounded_away_updates"]/a["nonzero_updates"]*100:.3f}%** of finite nonzero attention updates and **{m["rounded_away_updates"]/m["nonzero_updates"]*100:.3f}%** of MLP updates leave the stored residual unchanged. The corresponding FP32 control counts are only {control["attention_residual"]["rounded_away_updates"]} and {control["mlp_residual"]["rounded_away_updates"]}. These counters use each candidate\'s own finite FP32 increment, not a semantic importance score or a matched Native-update mask.',
        '', 'Power-of-two scaling changes exponent range, not the mantissa budget or update/residual ratio. It therefore does not solve ordinary FP16 update absorption. Larger divisors trade overflow headroom for more subnormal/zero values: c=4096 loses 3295 initially nonzero values versus 12 at c=16. There is no globally selected divisor from this one-layer, one-step screen.',
        '', 'The L2 column is implementation diagnostics only. It is not a generation distribution/semantics acceptance gate, and finite output is insufficient for promotion. Later layers/steps and error accumulation remain unmeasured.',
        '', '## Weight Fold A: fixed residual output compensation','',
        'Instead of dividing a dequantized row result by c, divide the FP32 `out_proj` and `fc2` weight-scale vectors by c. INT8 weights, activation quantization and INT32 collective payloads are unchanged. Online LoRA requires its separate post-B strength divided by c as well; adapter order and A/B GEMM shapes stay unchanged. A schema with row bias is rejected until that bias has its own compensation.',
        '', 'Folding happens before the output cast, so it is not universally byte-equivalent to dividing an already-rounded FP16 result. In the full-sequence c=16 case, folded/unfolded FP16-branch output hashes match. At c=256 they differ; the LoRA subset can also differ at c=16. Those distinctions are retained, not hidden behind an equality claim.',
        '', '## Weight Fold B: scalar norm-gain redistribution','',
        'A safer fold than changing every input-channel gain is a common power-of-two factor p:','',
        '```text','norm_gain_new = p * norm_gain',
        'AdaLN_shift_new = p * AdaLN_shift   # includes learned weight AND bias rows',
        'following_QKV_or_FC1_weight_scale_new = old_scale / p',
        'following_online_LoRA_post_B_strength_new = old_strength / p','```','',
        'Modulation scale/gate outputs remain unchanged. Compensating the shifts ensures the whole modulated input is multiplied by p; scaling only the norm gain is wrong for DiT. Compensation occurs before downstream nonlinearities. The implementation rejects an AdaLN adapter wrapper rather than silently omitting its row-wise correction.',
        '', 'With p=64, the FP32 whole-block control is byte-identical in these fixtures. In the FP16 TE projection probe, input casts that lose nonzero values fall from 2 to 0 and subnormal outputs from 6755 to 114. Relative projection L2 barely changes (0.33357% to 0.33353%): underflow prevention is useful range engineering, not a general precision improvement.',
        '', 'In the full-sequence FP16 DiT flow, adding the scalar-gain fold changes block L2 from 0.26682% to 0.26723%, with no nonfinite values in either. Reduced norm-output underflow alone does not establish better block accuracy or generation quality.',
        '', '## Weight Fold C: full channel-wise RMSNorm gain','',
        'The separate TE probe uses learned layer-0 RMSNorm gain and 512 Q-projection output rows with full input width 5120, on 257 seeded synthetic normalized inputs. It does not represent real TE activation distributions.',
        '', 'Checkpoint INT8 weights live in a ConvRot basis. To fold a nonuniform gain correctly, Native-dequantize/inverse-rotate the weight, multiply input columns in the original basis, then Native-requantize the full input width. Directly multiplying rotated INT8 columns by gamma is generally wrong because diagonal gamma and Hadamard rotation do not commute.',
        '', f'Dense algebraic folding has relative L2 {fold["dense_fold"]["relative_l2"]:.3e}. After Native INT8 requantization, the measured projection difference is **{fold["int8_fold"]["relative_l2"]*100:.4f}%**. This is a different quantized model, not a free exact fold. Static full-gain folding through dynamic DiT AdaLN shifts is rejected by the helper.',
        '', '## Memory meaning and next decision','',
        'The [7895,5376] residual payload is 161.91 MiB in FP32 and 80.95 MiB in FP16, per tensor/rank. That is a byte ledger only. The prototype intentionally materializes FP32 normalization/update islands and stores diagnostic observations; no 50% whole-block or pipeline-VRAM saving follows.',
        '', 'Retain c=4/c=16 as calibration candidates and fixed-output/scalar-gain folding as separate switches. Keep FP32 residual accumulation as the conservative comparator. Before any runtime option is introduced, calibrate real later-layer/later-step ranges and update absorption, validate online LoRA/AdaLN/final-head compensation, then implement a guarded SM75 FP16 preparation path and measure it on physical T4. Generation distribution/semantic acceptance remains a later full-pipeline gate, not something inferred here.',
        '', '## Reproduce','',
        'Run from the repository root with the existing conda `comfyui` Python. No setup/install command is required:','',
        '```powershell',
        '& C:\ProgramData\miniconda3\envs\comfyui\python.exe benchmarks/run_fp16_residual_fix.py prepare',
        '& C:\ProgramData\miniconda3\envs\comfyui\python.exe benchmarks/run_fp16_residual_fix.py screen --tokens 0',
        '& C:\ProgramData\miniconda3\envs\comfyui\python.exe benchmarks/run_fp16_residual_fix.py screen --tokens 513 --lora D:/minimaxH3/loras/dmad_minimax_h3_4step_full_critic_comfyui.safetensors',
        '& C:\ProgramData\miniconda3\envs\comfyui\python.exe benchmarks/run_fp16_residual_fix.py norm-fold',
        '& C:\ProgramData\miniconda3\envs\comfyui\python.exe -m unittest benchmarks.test_fp16_residual_fix',
        '```','',
        'The default cached-input location is local historical evidence; a clean clone needs those private fixture files or another explicitly supplied `--inputs` directory. Artifact tensors, prior source/output versions and logs stay under `D:/attention_eval/fp16_residual_fix_20261020`.',
        '', 'Source: [experimental gauge/folding helpers](../../benchmarks/fp16_residual_fix.py), [operator driver](../../benchmarks/run_fp16_residual_fix.py), [analytical tests](../../benchmarks/test_fp16_residual_fix.py), [compact results](../data/fp16_residual_fix_local_20261020.json). Ten new analytical/contract tests plus eight existing pipeline/runtime tests pass. Runtime source, model files, environment versions and defaults are unchanged.',
        '', 'Mathematical references: [PyTorch RMSNorm definition](https://docs.pytorch.org/docs/stable/generated/torch.nn.RMSNorm.html), [NVIDIA floating-point formats](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/mathematical-functions.html). These explain the formula/format, not qualification of a different Torch/CUDA version. Execution uses the frozen local packages.','']
    (ROOT/'docs/reports/FP16_RESIDUAL_FIX_LOCAL_20261020.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps(dict(configurations=len(full['cases'])+len(lora['cases']),repeat_cases=len(repeated),results=str(target)),indent=2))


if __name__=='__main__':main()
