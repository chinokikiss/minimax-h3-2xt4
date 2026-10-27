"""Lossless TE column-GEMM layout ablations, without persistent weight caches.

Runtime dispatch remains unchanged. This explicit benchmark changes only the
column accumulator call; row GEMMs, quantizers and dequantization stay intact.
"""
from contextlib import contextmanager
from unittest.mock import patch

from comfy_kitchen.backends.eager import quantization as eager

VARIANTS = ('legacy_copy', 'torch_view', 'kitchen_direct')


def legacy_copy(codes, weight):
    return eager._int8_matmul_accumulate(codes, weight.T.contiguous())


def torch_view(codes, weight):
    return eager._int8_matmul_accumulate(codes, weight.T)


def kitchen_direct(codes, weight):
    from patches.optimized_tp import native_int_mm
    return native_int_mm(codes, weight)


def accumulator(variant):
    if variant not in VARIANTS:
        raise ValueError('Unknown column layout experiment')
    return globals()[variant]


@contextmanager
def column_layout(variant):
    from patches import dit_column_linear as column
    # Patch only the TE/DiT column boundary, never Kitchen's global GEMM helper:
    # the two row projections must remain identical across the paired runs.
    with patch.object(column, '_column_accumulate', accumulator(variant)):
        yield


def balanced_order(repetition):
    offset = repetition % len(VARIANTS)
    return VARIANTS[offset:] + VARIANTS[:offset]
