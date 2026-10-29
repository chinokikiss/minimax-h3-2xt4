"""Explicit PTD overlay and deletion set for future frozen-source deliveries."""
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
RETIRED=('patches/ptd/auxiliary.py','patches/ptd/refiner_layout.py',
         'benchmarks/test_ptd_auxiliary.py','benchmarks/analyze_ptd_aux_split.py')
BENCHMARKS=('test_ptd_tp2.py','test_ptd_queue.py','test_ptd_cpu_schedule.py','test_ptd_overlay.py',
            'ptd_worker_probe.py','run_ptd_tp2_t4.py','ptd_delivery_sources.py')


def apply(files):
    """Historical parent capsules stay immutable; retired code cannot reappear."""
    for name in RETIRED: files.pop(name,None)
    paths=sorted(path for path in (ROOT/'patches/ptd').iterdir() if path.is_file())
    paths += [ROOT/'benchmarks'/name for name in BENCHMARKS]
    names=[]
    for path in paths:
        name=path.relative_to(ROOT).as_posix()
        files[name]=path.read_bytes(); names.append(name)
    assert not set(RETIRED)&files.keys()
    return tuple(names)
