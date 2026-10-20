"""CPU-only SM75 lowering of external-global-max and packing candidates."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import tilelang
from tilelang.engine import lower
from tilelang import tvm
from patches import tilelang_row_quant as candidate


def inspect(func, **kwargs):
    target = tvm.target.Target(kwargs['target'])
    with target, tvm.transform.PassContext(config=kwargs['pass_configs']): return lower(func, target=target)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--output', type=Path, required=True); args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True); tilelang.compile = inspect; rows = []
    for width in (3584, 7168):
        for rank in (0, 1):
            cases = [('quant', candidate.quantizer, (6814, width, width * 2, width * rank)),
                     ('max', candidate.maximum_kernel, (6814, width, width * 2, width * rank)),
                     ('pack', candidate.packing_kernel, (6814, width, 2, rank))]
            for name, function, values in cases:
                artifact = function(*values); source = args.output / f'{name}_k{width}_r{rank}.cu'
                source.write_text(artifact.kernel_source)
                rows.append(dict(name=source.stem, bytes=source.stat().st_size))
                print('STATIC_SM75', source.name, flush=True)
    (args.output / 'lowering.json').write_text(json.dumps(dict(scope='CPU codegen only, no GPU execution', cases=rows), indent=2))


if __name__ == '__main__': main()
