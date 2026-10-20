"""Source-derived storage ledger for one unique H3 TP2 block, not a GPU profile.

No torch import, runtime hooks, or inference changes. Each T is an explicit
allocation/release/alias boundary. Unknown library workspace is separate.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class Ledger:
    def __init__(self, rows, time_values):
        self.rows = rows
        self.time_values = time_values
        self.live = {}
        self.storages = {}
        self.events = []

    def event(self, action, name, shape=None, itemsize=4, note=''):
        delta = 0
        if action == 'alloc':
            assert name not in self.live
            count = 1
            for size in shape:
                count *= size
            delta = count * itemsize
            self.live[name] = dict(shape=shape, itemsize=itemsize, bytes=delta)
            # Names used repeatedly for small per-segment temporaries are
            # disambiguated by allocation T in the lifetime inventory.
            self.storages[str(len(self.events))] = dict(name=name, **self.live[name], birth_T=len(self.events), release_T=None)
        elif action == 'free':
            delta = -self.live.pop(name)['bytes']
            owner = next(v for v in reversed(list(self.storages.values())) if v['name'] == name and v['release_T'] is None)
            owner['release_T'] = len(self.events)
        self.events.append(dict(T=len(self.events), action=action, storage=name,
                                delta_bytes=delta, live_bytes=sum(v['bytes'] for v in self.live.values()),
                                live_mib=sum(v['bytes'] for v in self.live.values()) / 2**20,
                                live_storages=list(self.live), note=note))

    def alloc(self, name, width, itemsize=4):
        self.event('alloc', name, (self.rows, width), itemsize)

    def free(self, *names):
        for name in names:
            self.event('free', name)

    def alias(self, name, note):
        self.event('alias', name, note=note)


def column(l, prefix, width):
    # K=5376 FP32 ConvRot fallback; SM75 fused INT8 output route, CK 0.2.37.
    l.alloc(prefix + '.rotation', 5376)
    l.alloc(prefix + '.q', 5376, 1)
    l.alloc(prefix + '.scale', 1)
    l.free(prefix + '.rotation')
    l.alloc(prefix + '.unused_out', width)
    l.alloc(prefix + '.output', width)
    l.event('mark', prefix + '.turing_gemm', note='unused_out and helper output coexist')
    l.free(prefix + '.unused_out', prefix + '.q', prefix + '.scale')


def lora_column(l, prefix, ranks, output_width):
    for index, rank in enumerate(ranks):
        p = prefix + '.lora' + str(index)
        l.alloc(p + '.hidden', rank)
        l.alloc(p + '.B_delta', output_width)
        l.free(p + '.hidden')
        l.event('mark', p + '.scale_add', note='in-place; no second B output')
        l.free(p + '.B_delta')


def row(l, prefix, local_width, swiglu, backend, padding_lifetime='released', owned=False):
    # The row input is owned by the block/caller throughout this invocation.
    if owned:
        l.alloc(prefix + '.owned_activation', local_width)
        l.alloc(prefix + '.owned_rotation', local_width)
        l.free(prefix + '.owned_activation')
        l.alloc(prefix + '.maximum', 1)
        l.event('mark', prefix + '.MAX', note='same global FP32 row MAX')
        l.alloc(prefix + '.q', local_width, 1)
        l.alloc(prefix + '.scale', 1)
        l.free(prefix + '.owned_rotation', prefix + '.maximum')
    elif backend == 'packed':
        l.alloc(prefix + '.padded', 2 * local_width)
        l.alloc(prefix + '.full_rotation', 2 * local_width)
        if swiglu and padding_lifetime == 'before':
            l.alias(prefix + '.inner_del_padded', 'NOT freed: global_row_quantize caller still owns padded')
        else:
            l.free(prefix + '.padded')
        l.alloc(prefix + '.maximum', 1)
        l.event('mark', prefix + '.MAX', note='in-place NCCL MAX; external NCCL workspace excluded')
        l.alloc(prefix + '.q', local_width, 1)
        l.alloc(prefix + '.scale', 1)
        l.free(prefix + '.full_rotation', prefix + '.maximum')
        if swiglu and padding_lifetime == 'before':
            l.free(prefix + '.padded')
    else:
        if swiglu:
            l.alloc(prefix + '.activated', local_width)
        l.alloc(prefix + '.padded', 2 * local_width)
        if swiglu:
            l.free(prefix + '.activated')
        l.alloc(prefix + '.full_rotation', 2 * local_width)
        l.free(prefix + '.padded')
        l.alloc(prefix + '.owned_rotation', local_width)
        l.free(prefix + '.full_rotation')
        l.alloc(prefix + '.abs', local_width)
        l.alloc(prefix + '.maximum', 1)
        l.free(prefix + '.abs')
        l.event('mark', prefix + '.MAX', note='in-place NCCL MAX')
        l.alloc(prefix + '.sentinel', 256)
        l.alloc(prefix + '.cat', local_width + 256)
        l.alloc(prefix + '.q_with_sentinel', local_width + 256, 1)
        l.alloc(prefix + '.scale', 1)
        l.free(prefix + '.cat')
        l.alloc(prefix + '.q', local_width, 1)
        l.free(prefix + '.q_with_sentinel', prefix + '.sentinel', prefix + '.maximum', prefix + '.owned_rotation')
    l.alloc(prefix + '.int32_partial', 5376)
    l.event('mark', prefix + '.SUM', note='in-place Native INT32 SUM; no separate receive tensor here')
    l.alloc(prefix + '.output', 5376)
    l.free(prefix + '.int32_partial', prefix + '.q', prefix + '.scale')


def lora_row(l, prefix, ranks, local_width, swiglu, mode):
    if not ranks:
        return
    if swiglu:
        l.alloc(prefix + '.lora_activation', local_width)
        l.alias(prefix + '.gate_up', 'views retain packed FC1; activation is separate')
    stage = swiglu and mode == 'partial' and l.rows * sum(ranks) * 4 <= 32 * 2**20 and sum(ranks) < local_width
    if mode == 'gather':
        l.alloc(prefix + '.gather', 2 * local_width)
        if swiglu:
            l.free(prefix + '.lora_activation')
        l.alloc(prefix + '.reordered_full_input', 2 * local_width)
        l.free(prefix + '.gather')
    if stage:
        for i, rank in enumerate(ranks):
            l.alloc(prefix + '.hidden' + str(i), rank)
            l.event('mark', prefix + '.A_SUM' + str(i))
        l.free(prefix + '.lora_activation')
    for i, rank in enumerate(ranks):
        hidden = prefix + '.hidden' + str(i)
        if not stage:
            l.alloc(hidden, rank)
            if mode == 'partial':
                l.event('mark', prefix + '.A_SUM' + str(i))
            if swiglu and mode == 'partial' and i == len(ranks) - 1:
                l.free(prefix + '.lora_activation')
        l.alloc(prefix + '.B_delta', 5376)
        l.free(hidden)
        l.event('mark', prefix + '.scale_add' + str(i))
        l.free(prefix + '.B_delta')
    if mode == 'gather':
        l.free(prefix + '.reordered_full_input')


def build(args):
    l = Ledger(args.rows, args.time_values)
    l.event('mark', 'entry', note='residual x, RoPE, time basis and weights are external baseline')
    l.event('alloc', 'adaln', (args.time_values, 18 * 5376))
    for i, rank in enumerate(args.adaln_lora_ranks):
        p = 'adaln.lora' + str(i)
        l.event('alloc', p + '.hidden', (args.time_values, rank))
        l.event('alloc', p + '.B_delta', (args.time_values, 18 * 5376))
        l.free(p + '.hidden')
        l.event('mark', p + '.scale_add')
        l.free(p + '.B_delta')
    l.alias('adaln.chunk', 'six views; do not count six extra tensors')
    l.alloc('norm1', 5376)
    l.alias('h', 'same storage as norm1; modulation is in-place')
    l.event('alloc', 'modulation.scale_plus_one', (5376,))
    l.free('modulation.scale_plus_one')
    column(l, 'qkv', 10752)
    lora_column(l, 'qkv', args.qkv_lora_ranks, 21504)
    l.alias('q,k,v', 'views of qkv.output; V keeps all of that allocation alive')
    l.alloc('rope.Q', 3584)
    l.alloc('rope.K', 3584)
    l.alias('q_heads,k_heads,v_heads', 'transpose/unsqueeze views, no allocation')
    # CK 0.2.37 dense INT8 attention, D128, no mask, B1/H28.
    s = args.rows
    cta = 128 if s > 1024 else 64
    pad = ((s + cta - 1) // cta) * cta
    l.alloc('attention.Q8', 3584, 1)
    l.alloc('attention.K8', 3584, 1)
    l.event('alloc', 'attention.Q_scale', (28, ((s + 127) // 128) * 32))
    l.event('alloc', 'attention.K_scale', (28, ((s + cta - 1) // cta) * 4))
    l.event('alloc', 'attention.V8', (3584, pad), 1)
    l.event('alloc', 'attention.V_scale', (3584,))
    l.alloc('attention.bf16_output', 3584, 2)
    l.event('alloc', 'attention.anchors', (28,))
    l.alloc('attention.fp32_output_heads_layout', 3584)
    l.free('attention.Q8', 'attention.K8', 'attention.Q_scale', 'attention.K_scale',
           'attention.V8', 'attention.V_scale', 'attention.bf16_output', 'attention.anchors')
    l.alloc('attention.output_token_layout', 3584)
    l.free('attention.fp32_output_heads_layout')
    row(l, 'O', 3584, False, args.row_backend, args.padding_lifetime)
    lora_row(l, 'O', args.lora_ranks, 3584, False, args.row_lora)
    l.free('attention.output_token_layout')
    l.event('mark', 'attention.residual_add', note='x.addcmul_; no new SxH output')
    l.free('norm1', 'qkv.output', 'rope.Q', 'rope.K', 'O.output')
    l.alloc('norm2', 5376)
    l.alias('h2', 'same storage as norm2')
    l.event('alloc', 'modulation.scale_plus_one', (5376,))
    l.free('modulation.scale_plus_one')
    column(l, 'FC1', 14336)
    selected = args.selective_b and all(rank == 128 for rank in args.lora_ranks)
    lora_column(l, 'FC1', args.lora_ranks, 14336 if selected else 28672)
    l.free('norm2')
    l.alias('mlp_act', 'same storage as FC1.output; not a separate SwiGLU result')
    row(l, 'FC2', 7168, True, args.row_backend, args.padding_lifetime, args.fc2_rotation=='owned_fp32')
    lora_row(l, 'FC2', args.lora_ranks, 7168, True, args.row_lora)
    l.event('mark', 'mlp.residual_add', note='FC1 output remains live through residual update and return')
    l.free('FC2.output', 'FC1.output', 'adaln')
    l.event('mark', 'return_x', note='same input storage; block-created live storage is now zero')
    assert not l.live
    return l


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--rows', type=int, default=8192)
    p.add_argument('--time-values', type=int, default=3)
    p.add_argument('--lora-ranks', type=int, nargs='*', default=[], help='Per-adapter ranks for FC1/FC2/O (DMAD:128)')
    p.add_argument('--qkv-lora-ranks', type=int, nargs='*', default=[], help='QKV ranks, specified separately (DMAD:384)')
    p.add_argument('--adaln-lora-ranks', type=int, nargs='*', default=[], help='Optional AdaLN adapters; DMAD has none')
    p.add_argument('--row-lora', choices=['partial', 'gather'], default='partial')
    p.add_argument('--row-backend', choices=['packed', 'native'], default='packed')
    p.add_argument('--padding-lifetime', choices=['before', 'released'], default='released',
                   help='before reproduces the historical caller-held padding; released follows the lifetime fix')
    p.add_argument('--fc2-rotation', choices=['native','owned_fp32'], default='native')
    p.add_argument('--selective-b', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--step', type=int, help='Print exact named-storage inventory at this event T')
    p.add_argument('--output', type=Path, required=True, help='JSON and same-stem CSV; compact static ledger only')
    args = p.parse_args()
    if args.rows < 2 or args.time_values < 1 or any(r < 1 for r in args.lora_ranks + args.qkv_lora_ranks + args.adaln_lora_ranks):
        p.error('Requires S>=2, positive time values and positive LoRA ranks')
    ledger = build(args)
    sources = ['patches/tp_dit.py', 'patches/h3_operator_policy.py', 'patches/h3_lora.py',
               'patches/tilelang_swiglu_pad.py', 'patches/tilelang_row_quant.py', 'patches/optimized_tp.py',
               'ComfyUI/comfy/ldm/minimax/model.py',
               'semantic_results/selective_b_t4_v1/kitchen_0_2_37_cuda.py',
               'semantic_results/dit_tensor_ledger/ck_0_2_37_sage_attention.py',
               'semantic_results/dit_tensor_ledger/ck_0_2_37_int8_utils.py']
    data = dict(kind='analytical named-storage ledger, not physical allocator measurement',
                configuration={k: v for k, v in vars(args).items() if k not in ('output', 'step')},
                assumptions=['FP32, TP2, B1, H5376, local heads28/D128, trained rank widths supplied explicitly',
                             'Kitchen 0.2.37 dense attention; SM75 packed row path or selected Native fallback',
                             'uniform scalar AdaLN segment indices; no per-token advanced-index modulation',
                             'no autograd, extra patch callbacks, graphs, or activation retention by caller hooks',
                             'warm cached Hadamard; its 256x256 FP32 buffer belongs to persistent baseline',
                             'no masks in DiT attention; residual x, outer forward activations and weights are external',
                             'hidden RMSNorm/BLAS/NCCL/extension workspaces and allocator rounding excluded',
                             'free means last Python owner released; CUDA deferred reuse is an additional term'],
                source_sha256={name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in sources},
                input_residual_bytes=args.rows * 5376 * 4,
                storages=list(ledger.storages.values()),
                events=ledger.events, peak=max(ledger.events, key=lambda e: e['live_bytes']))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, indent=2), encoding='utf8')
    with args.output.with_suffix('.csv').open('w', newline='', encoding='utf8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(ledger.events[0]))
        writer.writeheader()
        for event in ledger.events:
            writer.writerow({**event, 'live_storages': '|'.join(event['live_storages'])})
    chosen = ledger.events[args.step] if args.step is not None else data['peak']
    inventory = [s for s in data['storages'] if s['birth_T'] <= chosen['T'] < s['release_T']]
    print(json.dumps(dict(event=chosen, tensors=inventory), indent=2))


if __name__ == '__main__':
    main()
