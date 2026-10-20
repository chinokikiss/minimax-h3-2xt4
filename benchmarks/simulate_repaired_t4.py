"""Execute the exact planned inputs with two logical ranks before submission."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
for path in [ROOT, ROOT / 'ComfyUI', ROOT / '.retest-deps']:
    sys.path.insert(0, str(path))
import hashlib
import json
import time
import torch
from benchmarks.logical_dual_gpu import LogicalDualGPU, isolate_cublas_workspaces
from benchmarks.retest import load_native_block, load_video_native, metrics, native_call, passes, write_json
from comfy.text_encoders.llama import Qwen3VL_32BConfig, precompute_freqs_cis
import patches.tp_qwen3vl as te_tp
import patches.sp_int8_qwen3vl as te_sp
import patches.tp_dit as dit_tp
import patches.sp_int8_dit as dit_sp
import patches.tp_video_vae as video_parallel


def main():
    start = time.perf_counter()
    fixtures = ROOT / 'repaired_fixtures'
    specs = json.loads((fixtures / 'fixture_manifest.json').read_text())
    gate = {'rel_l2_max': 1e-5, 'cosine_min': .999999, 'max_abs_max': .005}
    result = {'scope': 'exact proposed inputs; two independent ranks on one 5060 Ti; no physical performance claim',
              'gate': gate, 'cases': [], 'passed': False}
    target = ROOT / 'semantic_results/repaired_t4_inputs.json'
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    isolate_cublas_workspaces()
    comm = LogicalDualGPU()
    for module in [te_tp, te_sp, dit_tp, dit_sp, video_parallel]:
        module.dist = comm
    def record(kind, mode, name, out, ref):
        score = metrics(out, ref)
        score.update(suite=kind, mode=mode, workload=name, passed=passes(score, gate))
        result['cases'].append(score)
        write_json(target, result)
        print(kind, mode, name, 'PASS' if score['passed'] else 'FAIL', score['rel_l2'], flush=True)
        if not score['passed']:
            raise RuntimeError('Planned workload failed local semantic comparison')
    with torch.no_grad():
        for kind in ['te', 'dit']:
            native = load_native_block(kind, fixtures / specs['checkpoints'][kind]['file'])
            for mode in ['tp', 'sp']:
                cls = ((te_tp.TPTransformerBlockINT8 if mode == 'tp' else te_sp.SPTransformerBlockINT8)
                       if kind == 'te' else (dit_tp.TPDiTBlockINT8 if mode == 'tp' else dit_sp.SPDiTBlockINT8))
                ranks = [cls(rank=r, world_size=2, device='cuda:0').cuda().eval().requires_grad_(False) for r in range(2)]
                for rank in ranks:
                    rank.load_from_full_block(native)
                for row in specs['inputs'][kind]:
                    data = torch.load(fixtures / row['file'], weights_only=True)
                    x = data['input'].cuda()
                    if kind == 'te':
                        config = Qwen3VL_32BConfig()
                        freq = precompute_freqs_cis(128, torch.arange(x.shape[1], device=x.device)[None],
                            config.rope_theta, rope_dims=config.rope_dims, device=x.device, interleaved_mrope=True)
                        mask = torch.full((x.shape[1], x.shape[1]), torch.finfo(x.dtype).min / 4,
                            device=x.device, dtype=x.dtype).triu(1)
                        ref = native_call(kind, native, x)
                        chunks = list(torch.tensor_split(x, 2, 1)) if mode == 'sp' else [x, x]
                        outputs = comm.run([lambda r=r: ranks[r](chunks[r].clone(), mask, freq) for r in range(2)])
                        out = torch.cat(outputs, 1) if mode == 'sp' else outputs[0]
                    else:
                        t = data['t_emb'].cuda()
                        rope = data['rope'].cuda()
                        segments = data['segments']
                        ref = native_call(kind, native, x, t, segments, rope)
                        chunks = list(torch.tensor_split(x, 2, 0)) if mode == 'sp' else [x, x]
                        outputs = comm.run([lambda r=r: ranks[r](chunks[r].clone(), t, segments, rope) for r in range(2)])
                        out = torch.cat(outputs, 0) if mode == 'sp' else outputs[0]
                    record(kind, mode, row['name'], out, ref)
                    del x, ref, out, outputs, chunks, data
                del ranks
            del native
            torch.cuda.empty_cache()
        checkpoint = Path('D:/minimaxH3') / specs['checkpoints']['video']['file']
        for mode in ['tp', 'sp']:
            native, sd = load_video_native(checkpoint, 'cuda:0')
            ranks = []
            patcher = video_parallel.patch_video_vae_tp if mode == 'tp' else video_parallel.patch_video_vae_sp
            for r in range(2):
                model, unused = load_video_native(checkpoint, 'cuda:0')
                del unused
                ranks.append(patcher(model, sd, r, 2, torch.device('cuda:0')))
            del sd
            for row in specs['inputs']['video']:
                z = torch.load(fixtures / row['file'], weights_only=True)['input'].cuda()
                ref = native._decode_pixels(z)
                outputs = comm.run([lambda r=r: ranks[r]._decode_pixels(z.clone()) for r in range(2)])
                record('video', mode, row['name'], outputs[0], ref)
                record('video', mode + '_rank_agreement', row['name'], outputs[1], outputs[0])
                del z, ref, outputs
            del native, ranks, model
            torch.cuda.empty_cache()
    result.update(passed=True, seconds=time.perf_counter() - start,
                  fixture_manifest_sha256=hashlib.sha256((fixtures / 'fixture_manifest.json').read_bytes()).hexdigest(),
                  source_sha256={p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in set((ROOT / 'patches').glob('*.py')) | {
                        Path(__file__), ROOT / 'benchmarks/retest.py', ROOT / 'benchmarks/repaired_t4.py',
                        ROOT / 'benchmarks/fixture_inputs.py'}})
    write_json(target, result)
    print('ALL PASS', len(result['cases']), result['seconds'], flush=True)


if __name__ == '__main__':
    main()
