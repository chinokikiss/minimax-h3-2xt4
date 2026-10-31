"""Synthetic Sol route and bitmap-LUT equivalence probe, not KC fusion."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import time

import torch
from patches import sol_attention
from patches import sol_sm75_backend as native


def inputs(shape, seed):
    b, h, length, dim = shape
    result = []
    for idx in range(3):
        g = torch.Generator(device="cpu").manual_seed(seed + idx)
        x = torch.randn((b, length, h, dim), dtype=torch.float16, generator=g)
        result.append(x.cuda().transpose(1, 2))
    return result


def gpu_ms(fn, repeats=3, warmup=1):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    values = []
    for _ in range(repeats):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        result = fn()
        end.record()
        end.synchronize()
        values.append(float(start.elapsed_time(end)))
        del result
    return dict(median_ms=statistics.median(values), samples_ms=values)


def valid_lut_sha256(lut, counts, *, bitmap=False, key_blocks=None):
    """Hash logical ordered K indices, never uninitialized tail slots.

    Bitmap and int32-LUT variants must produce identical hashes for precisely
    the same sequence of exact K blocks, not merely the same unordered mask.
    """
    packed = lut.detach().cpu().reshape(-1, lut.shape[-1])
    sizes = counts.reshape(-1)
    digest = hashlib.sha256()
    for row in range(sizes.numel()):
        count = int(sizes[row])
        if not bitmap:
            digest.update(packed[row, :count].numpy().tobytes())
            continue
        if key_blocks is None or packed.shape[1] != (key_blocks+31)//32:
            raise ValueError("Invalid bitmap LUT shape")
        ordered = []
        for word_idx in range(packed.shape[1]):
            pending = int(packed[row,word_idx]) & 0xffffffff
            while pending:
                bit = (pending & -pending).bit_length() - 1
                block = word_idx*32+bit
                if block >= key_blocks:
                    raise AssertionError("Bitmap LUT includes padded key blocks")
                ordered.append(block)
                pending &= pending-1
        if len(ordered) != count:
            raise AssertionError("Bitmap popcount and RowCount differ")
        import array
        digest.update(array.array("i",ordered).tobytes())
    return digest.hexdigest()


def decode_bitmap_mask(bitmap, key_blocks):
    """Recover the logical exact mask for A/B comparison only (CPU)."""
    words=bitmap.detach().cpu().to(torch.int64)&0xffffffff
    shifts=torch.arange(32,dtype=torch.int64)
    return ((words.unsqueeze(-1)>>shifts)&1).reshape(
        *words.shape[:-1],words.shape[-1]*32
    )[..., :key_blocks].to(torch.uint8).contiguous()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    torch.set_grad_enabled(False)
    torch.cuda.set_device(0)
    assert torch.cuda.is_available() and torch.cuda.get_device_capability(0) == (7, 5)
    torch.backends.cuda.matmul.allow_tf32 = False
    bitmap = os.environ.get("H3_SOL_LUT_BITMAP", "0") == "1"
    assert not bool(os.environ.get("H3_SOL_WORKSPACE_POOL", "0") == "1")
    library = sol_attention.prepare_native()
    payload = dict(
        variant="bitmap" if bitmap else "baseline",
        abi=int(native._load_library().star7_sla_sm75_abi_version()),
        library=library, gpu=torch.cuda.get_device_name(0), cases={})
    for name, shape, seed in [
        ("d64_vae", (1, 8, 1029, 64), 20261031),
        ("real_shape_d128", (1, 4, 65166, 128), 20261040),
    ]:
        q, k, v = inputs(shape, seed)
        if name == "d64_vae":
            topo=dict(kind="video_vae",start_token=0,end_token=1024,
                      frame_stride_tokens=256,temporal_radius=1,
                      local_radius_tokens=64,protect_frame_boundaries=False)
            kwargs=dict(protected_tokens=5,protected_start=1024,allow_d64=True,
                        tau=1.,topology=topo,sync_free_lut=True,
                        collect_route_stats=False)
        else:
            topo=dict(kind="dit",start_token=4974,end_token=65166,
                      frame_stride_tokens=836,temporal_radius=1,
                      local_radius_tokens=64,protect_frame_boundaries=True)
            kwargs=dict(protected_tokens=4974,protected_start=0,allow_d64=False,
                        tau=1.,topology=topo,sync_free_lut=True,
                        collect_route_stats=False)
        prepare = lambda: sol_attention.prepare_attention(q,k,v,**kwargs)
        timing_prepare = gpu_ms(prepare)
        prepared = prepare()
        exact_mask = (decode_bitmap_mask(prepared["lut"],(shape[-2]+63)//64)
                      if bitmap else prepared["exact_mask"].detach().cpu().clone())
        row_count = prepared["row_count"].detach().cpu().clone()
        lut_sha256 = valid_lut_sha256(
            prepared["lut"], row_count, bitmap=bitmap,
            key_blocks=(shape[-2]+63)//64)
        prepared_lut_shape=tuple(prepared["lut"].shape)
        execute=lambda:sol_attention.execute_prepared(prepared)
        timing_execute=gpu_ms(execute)
        output=execute().detach().cpu().clone()
        torch.cuda.synchronize()
        del prepared
        torch.cuda.empty_cache()
        complete=lambda: sol_attention.attention(q,k,v,**kwargs)
        timing_full=gpu_ms(complete)
        torch.cuda.synchronize()
        before=torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        test_output=complete()
        torch.cuda.synchronize()
        peak=int(max(0,torch.cuda.max_memory_allocated()-before))
        torch.testing.assert_close(test_output.cpu(),output,atol=0,rtol=0)
        torch.save(dict(output=output,exact_mask=exact_mask,row_count=row_count),
                   args.out/(name+".pt"))
        payload["cases"][name]=dict(shape=list(shape),prepare=timing_prepare,
            execute=timing_execute,complete=timing_full,
            peak_temporary_bytes=peak,lut_sha256=lut_sha256,
            lut_shape=list(prepared_lut_shape),route_density=float(
                row_count.sum().item()/(row_count.numel()*exact_mask.shape[-1])))
        del q,k,v,output,exact_mask,row_count,test_output
        torch.cuda.empty_cache()
    (args.out/"result.json").write_text(json.dumps(payload,indent=2))
    print(json.dumps(payload,indent=2),flush=True)


if __name__=="__main__":
    main()
